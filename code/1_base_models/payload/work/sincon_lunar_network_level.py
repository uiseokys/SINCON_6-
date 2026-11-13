"""Compare literal lunar-holiday phases with local ordinary Sunday demand."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sincon_network_level_audit import STABLE_EXCLUSIONS
from sincon_munhak_ordinary import WARM
from sincon_songdo_schedule import save_submission
from sincon_station_state import KEY
from sincon_weather_sensitivity import score
from sincon_weekly_variability import protected_days


BANDS = [6, 10, 16, 21]
DAY = KEY[:2]
CAP_SCALE = .80


def calendars(work):
    with ZipFile(work / "sincon_local_repro.zip") as bundle:
        calendar = pd.read_csv(bundle.open("external_inputs/calendar_features.csv"), parse_dates=["date"])
    centers = calendar.loc[calendar.holiday_name.isin(["설날", "추석"]), ["date", "holiday_name"]]
    phases = pd.concat([centers.assign(date=centers.date + pd.Timedelta(days=offset), phase=offset)
                        for offset in (-1, 0, 1)], ignore_index=True).sort_values("date")
    if not phases.date.is_unique:
        raise ValueError("Each literal lunar holiday day must have a unique phase.")
    return phases


def effects(rows, phases, protected, value):
    holidays, events = protected
    eligible = ~rows.station_id.isin(STABLE_EXCLUSIONS) & ~pd.MultiIndex.from_frame(rows[DAY]).isin(events)
    panel = rows.loc[eligible, [*KEY, value]].pivot(index="date", columns=["station_id", "hour"], values=value)
    sundays = panel.loc[(panel.index.day_of_week == 6) & ~panel.index.isin(holidays)]
    hours = panel.columns.get_level_values("hour").to_numpy()
    band = np.asarray(BANDS)[np.searchsorted(BANDS, hours, side="right") - 1]
    results = []
    for row in phases.loc[phases.date.isin(panel.index)].itertuples(index=False):
        distance = np.abs((sundays.index - row.date).days.to_numpy())
        normal = sundays.loc[distance <= 42]
        if len(normal) < 3:
            continue
        normalized = panel.loc[row.date] - normal.median(axis=0)
        shift = {str(h): float(normalized.loc[band == h].median()) for h in BANDS}
        if not np.isfinite(list(shift.values())).all():
            raise ValueError("Every holiday phase needs an ordinary-hour anchor.")
        results.append({"date": row.date, "phase": int(row.phase), "family": row.holiday_name,
                        "ordinary_sundays": len(normal), "bands": shift})
    return results


def proposal(history, target, phases, cutoff, protected):
    if history.date.max() > cutoff or target.date.min() <= cutoff or "y_log" in target:
        raise ValueError("Use only older realized boardings and unlabeled future forecasts.")
    prior = effects(history, phases, protected, "observed")
    future = effects(target, phases, protected, "baseline")
    allow = ~target.station_id.isin(STABLE_EXCLUSIONS) & ~pd.MultiIndex.from_frame(target[DAY]).isin(protected[1])
    delta = np.zeros(len(target))
    band = np.asarray(BANDS)[np.searchsorted(BANDS, target.hour, side="right") - 1]
    report = {}
    for day in future:
        donors = [row for row in prior if row["phase"] == day["phase"]]
        if not donors:
            continue
        desired = np.median([[row["bands"][str(h)] for h in BANDS] for row in donors], axis=0)
        incumbent = np.array([day["bands"][str(h)] for h in BANDS])
        shift = .5 * (desired - incumbent).clip(-.20, .20)
        for hour, amount in zip(BANDS, shift):
            mask = allow & target.date.eq(day["date"]) & (band == hour)
            delta[mask] = amount
        report[str(day["date"].date())] = {"phase": day["phase"], "ordinary_sundays": day["ordinary_sundays"],
                                          "donors": [str(row["date"].date()) for row in donors],
                                          "prior_relative_log_profile": desired.tolist(),
                                          "base_relative_log_profile": incumbent.tolist(), "adjustment": shift.tolist()}
    return delta, report


def verified_lunar_phases(rows, change, phases):
    return np.where(rows.date.isin(phases.loc[phases.phase.le(0), "date"]), CAP_SCALE * change, 0.)


def combined_replay(args, tabicl=None):
    archive = pd.read_pickle(args.archive)
    if tabicl is None and not args.tabicl_oof:
        return archive
    keys = ["label", "origin", *KEY]
    if tabicl is None:
        tabicl = pd.read_pickle(args.tabicl_oof)
    tabicl = tabicl.rename(columns={"log_delta": "tabicl"})
    retail = pd.read_csv(args.retail_oof, parse_dates=["date", "origin"]).rename(columns={"log_delta": "retail"})
    known = pd.MultiIndex.from_frame(archive[keys])
    if any(not pd.MultiIndex.from_frame(part[keys]).isin(known).all() for part in (tabicl, retail)):
        raise ValueError("Each adjustment must match the exact forecast keys.")
    updated = archive.merge(tabicl, on=keys, how="left", validate="one_to_one").merge(retail, on=keys, how="left", validate="one_to_one")
    covered = updated.label.isin(tabicl.label.unique())
    if updated.loc[covered, "tabicl"].isna().any():
        raise ValueError("TabICL coverage must include every key in its supplied validation cases.")
    return updated.assign(baseline=updated.baseline + updated.tabicl.fillna(0.) + updated.retail.fillna(0.))


def export(args, test, baseline, delta):
    from sincon_tabicl_final_delta import protect_previous_layers

    manifest = json.loads(args.incumbent_report.read_text())
    if hashlib.sha256(args.baseline.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("The incumbent checksum must match the validated combined prediction.")
    _, layers = protect_previous_layers(test, baseline.boardings, args.work_dir, args.team)
    preserved = pd.DataFrame(layers).any(axis=1).to_numpy()
    changed = delta != 0.
    if (np.any(preserved & changed) or set(test.loc[changed, "date"].dt.strftime("%Y-%m-%d")) != {"2026-09-24", "2026-09-25"}
            or not np.isfinite(delta).all() or np.max(np.abs(delta)) > .080000001):
        raise ValueError("The adjustment must be bounded and restricted to the verified Chuseok phases.")
    values = baseline.boardings.to_numpy(copy=True)
    values[changed] = np.expm1(np.log1p(values[changed]) + delta[changed])
    save_submission(args.submission, baseline, test, values)
    return {"submission": str(args.submission.resolve()), "rows": len(test), "changed_rows": int(changed.sum()),
            "protected_event_rows": int(preserved.sum()), "sha256": hashlib.sha256(args.submission.read_bytes()).hexdigest()}


def run(args):
    archive = combined_replay(args)
    history = pd.read_csv(args.team / "data/train.csv", parse_dates=["date"], usecols=[*KEY, "boardings"])
    history["observed"] = np.log1p(history.pop("boardings"))
    phases = calendars(args.work_dir)
    protected = protected_days(args.work_dir)
    observed = history.set_index(KEY).observed
    report, forecasts = {}, {}
    for label, block in archive.groupby("label", sort=False):
        origin = block.origin.iloc[0]
        if not np.allclose(observed.reindex(pd.MultiIndex.from_frame(block[KEY])), block.y_log, atol=1e-8, rtol=0.):
            raise ValueError(f"Archived official labels differ in {label}.")
        base = block.baseline.to_numpy()
        change, detail = (proposal(history.loc[history.date.le(origin)], block[[*KEY, "baseline"]], phases, origin, protected)
                          if block.date.isin(phases.date).any() else (np.zeros(len(block)), {}))
        selected = verified_lunar_phases(block, change, phases)
        truth = block.y_log.to_numpy()
        rows = {}
        for name, scope in (("all", np.ones(len(block), dtype=bool)), ("adjusted", change != 0.)):
            if scope.any():
                rows[name] = score(truth[scope], base[scope], base[scope] + change[scope], block.date[scope])
        for date, entry in detail.items():
            day = block.date.eq(date) & (change != 0.)
            if day.any():
                entry.update(score(truth[day], base[day], base[day] + change[day], block.date[day]))
                entry["eve_and_center"] = score(truth[day], base[day], base[day] + selected[day], block.date[day])
        report[label] = {"origin": str(origin.date()), "dates": detail, **rows,
                         "eve_and_center": score(truth, base, base + selected, block.date)}
        forecasts[label] = block.assign(change=selected)
    warm = pd.concat([forecasts[label].loc[forecasts[label].date.dt.month.eq(month)] for label, month in WARM])
    ac = pd.concat([forecasts["A"], forecasts["C"]])
    for label, block in (("disjoint_A_C", ac), ("disjoint_warm", warm)):
        if block.duplicated(KEY).any():
            raise ValueError("Pooled reports need distinct dates and station-hours.")
        report[label] = {"eve_and_center": score(block.y_log.to_numpy(), block.baseline.to_numpy(),
                                                  (block.baseline + block.change).to_numpy(), block.date)}
    test = pd.read_csv(args.team / "data/test.csv", parse_dates=["date"], usecols=[*KEY, "example_id"])
    baseline = pd.read_csv(args.baseline, float_precision="round_trip")
    if len(test) != 157_194 or not test.example_id.equals(baseline.example_id):
        raise ValueError("Final sensitivities must use the official test grid in order.")
    test["baseline"] = np.log1p(baseline.boardings)
    change, report["test"] = proposal(history, test[[*KEY, "baseline"]], phases, pd.Timestamp("2026-06-30"), protected)
    report["test_rows"] = int(np.count_nonzero(change))
    delta = verified_lunar_phases(test, change, phases)
    report["test_eve_and_center_rows"] = int(np.count_nonzero(delta))
    if args.submission:
        if any(report[label]["eve_and_center"]["candidate"] >= report[label]["eve_and_center"]["reference"]
               for label in ("B", "C", "disjoint_warm", "disjoint_A_C")):
            raise ValueError("The adopted phases must improve both different historical holiday blocks.")
        report["export"] = export(args, test, baseline, delta)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "archive", "baseline", "results"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--tabicl-oof", type=Path)
    parser.add_argument("--retail-oof", type=Path)
    parser.add_argument("--incumbent-report", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    if bool(args.tabicl_oof) != bool(args.retail_oof) or bool(args.submission) != bool(args.incumbent_report):
        parser.error("Supply both validation overlays and pair any export with its incumbent report.")
    report = run(args)
    args.results.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for label, block in report.items():
        print(label, block.get("eve_and_center", block) if isinstance(block, dict) else block)


if __name__ == "__main__":
    main()
