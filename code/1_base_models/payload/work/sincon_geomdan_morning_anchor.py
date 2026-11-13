"""Compare recent commuting-hour anchors for Geomdan Lake Park station."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_3107_mature_hour_profile import NEIGHBORS, STATION, ordinary, weekday_group
from sincon_geomdan_opening_control import OPENED
from sincon_lunar_network_level import calendars, combined_replay, proposal, verified_lunar_phases
from sincon_munhak_ordinary import WARM
from sincon_songdo_schedule import save_submission
from sincon_spectator_month_split import gain_summary, month_halves
from sincon_station_state import KEY
from sincon_weather_sensitivity import score
from sincon_weekly_variability import protected_days


STATIONS = (STATION, *NEIGHBORS)
METHODS = ("relative_morning", "relative_peak_share", "mean_of_anchors")
WINDOW = 56
CAP = .25
STRENGTH = .5


def profiles(rows, column, protected):
    local = rows.loc[rows.station_id.isin(STATIONS) & rows.hour.between(6, 15), [*KEY, column]].copy()
    local["band"] = np.where(local.hour.lt(10), "morning", "midday")
    local["counts"] = np.expm1(local[column].clip(lower=0.))
    levels = np.log1p(local.groupby(["date", "band", "station_id"]).counts.sum().unstack(["band", "station_id"]))
    if levels.empty:
        return pd.DataFrame(columns=["relative_morning", "relative_peak_share", "ordinary", "weekday"])
    morning = levels["morning"]
    peak_share = morning - levels["midday"]
    return pd.DataFrame({"relative_morning": morning[STATION] - morning[list(NEIGHBORS)].mean(axis=1),
                         "relative_peak_share": peak_share[STATION] - peak_share[list(NEIGHBORS)].mean(axis=1),
                         "ordinary": ordinary(levels.index, protected), "weekday": weekday_group(levels.index)})


def forecast(history, target, origin, protected):
    if history.date.max() > origin or target.date.min() <= origin or "y_log" in target:
        raise ValueError("Training station levels must predate the forecast rows.")
    past = profiles(history, "observed", protected)
    actual = past.loc[past.ordinary & (past.index > origin - pd.Timedelta(days=WINDOW))
                      & (past.index >= OPENED + pd.Timedelta(days=14))]
    future = profiles(target, "baseline", protected)
    future = future.loc[future.ordinary].assign(month=lambda f: f.index.to_period("M"))
    anchor = list(METHODS[:2])
    proposals = []
    for (month, weekday), block in future.groupby(["month", "weekday"]):
        earlier = actual.loc[actual.weekday.eq(weekday), anchor].dropna()
        if len(earlier) < 4:
            same_weekend = actual.weekday.ge(2) == (weekday >= 2)
            earlier = actual.loc[same_weekend, anchor].dropna()
        if len(earlier) < 4:
            continue
        change = STRENGTH * (earlier.median() - block[anchor].median()).clip(-CAP, CAP)
        proposals.append(pd.DataFrame([change.to_dict()] * len(block), index=block.index).assign(
            sample_days=len(earlier), month=str(month), weekday=int(weekday)))
    keys = target.date
    scope = target.station_id.eq(STATION) & target.hour.between(6, 9)
    day_delta = pd.concat(proposals) if proposals else pd.DataFrame(columns=anchor)
    if not day_delta.index.is_unique:
        raise ValueError("Each target date must have one past-based commuting level.")
    adjustments = {method: np.where(scope, keys.map(day_delta[method]).fillna(0.), 0.) for method in anchor}
    adjustments[METHODS[2]] = (adjustments[anchor[0]] + adjustments[anchor[1]]) / 2
    return adjustments, {"donor_days": len(actual), "target_days": len(day_delta),
                         "median_deltas": day_delta[anchor].median().dropna().to_dict()}


def evaluate(frame):
    local = frame.station_id.eq(STATION) & frame.hour.between(6, 9)
    results = {}
    for label, block in (("all", frame), ("station_3107_06_09", frame.loc[local])):
        actual, base = block.y_log.to_numpy(), block.baseline.to_numpy()
        results[label] = {method: {**score(actual, base, base + block[method].to_numpy(), block.date),
                                    "changed_rows": int(block[method].ne(0.).sum())} for method in METHODS}
    return results


def split_check(frame):
    actual, baseline = frame.y_log.to_numpy(), frame.baseline.to_numpy()
    correction = frame.relative_peak_share.to_numpy()
    values = frame[["date"]].assign(old=(actual - baseline)**2,
                                      new=(actual - (baseline + correction).clip(min=0.))**2, n=1).groupby("date").sum()
    halves = month_halves(values.index, 10_000, 116).astype(float)
    errors, counts = values[["old", "new"]].to_numpy(), values.n.to_numpy()
    sampled, rows = halves @ errors, halves @ counts
    left = np.sqrt(sampled / rows[:, None])
    right = np.sqrt((errors.sum(axis=0) - sampled) / (counts.sum() - rows)[:, None])
    gains = np.column_stack((left[:, 0] - left[:, 1], right[:, 0] - right[:, 1]))
    return {**gain_summary(gains.ravel()), "both_improve_share": float(np.mean(np.all(gains > 1e-12, axis=1)))}


def export(args, test, predictions, changes, protected):
    baseline_digest = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    expected = json.loads(args.incumbent_report.read_text())["export"]["sha256"]
    if baseline_digest != expected:
        raise ValueError("The tested adjustment requires the checked Chuseok incumbent.")
    rows = changes != 0.
    affected = test.loc[rows]
    scope = affected.station_id.eq(STATION) & affected.hour.between(6, 9) & ~affected.date.isin(protected[0])
    if (not rows.any() or not scope.all() or np.any(pd.MultiIndex.from_frame(affected[KEY[:2]]).isin(protected[1]))
            or not np.isfinite(changes).all() or np.abs(changes).max() > STRENGTH * CAP + 1e-12):
        raise ValueError("Only bounded ordinary morning adjustments to station 3107 can be exported.")
    values = predictions.boardings.to_numpy(copy=True)
    values[rows] = np.expm1((np.log1p(values[rows]) + changes[rows]).clip(min=0.))
    save_submission(args.submission, predictions, test, values)
    return {"submission": str(args.submission.resolve()), "rows": len(test), "changed_rows": int(rows.sum()),
            "changed_dates": int(affected.date.nunique()), "station_id": STATION,
            "log_delta_range": [float(changes[rows].min()), float(changes[rows].max())],
            "incumbent_sha256": baseline_digest, "sha256": hashlib.sha256(args.submission.read_bytes()).hexdigest()}


def run(args):
    archive = combined_replay(args)
    protected, phases = protected_days(args.work_dir), calendars(args.work_dir)
    raw = pd.read_csv(args.team / "data/train.csv", parse_dates=["date"], usecols=[*KEY, "boardings"])
    history = raw.assign(observed=np.log1p(raw.pop("boardings")))
    truth = history.set_index(KEY).observed
    forecasts, report = {}, {}
    for label, block in archive.groupby("label", sort=False):
        origin = block.origin.iloc[0]
        if not np.allclose(truth.reindex(pd.MultiIndex.from_frame(block[KEY])), block.y_log, atol=1e-8, rtol=0.):
            raise ValueError(f"Archived counts differ from official historical observations in {label}.")
        earlier = history.loc[history.date.le(origin)]
        inputs = block[[*KEY, "baseline"]]
        if block.date.isin(phases.date).any():
            lunar, _ = proposal(earlier, inputs, phases, origin, protected)
            block = block.assign(baseline=block.baseline + verified_lunar_phases(block, lunar, phases))
            inputs = block[[*KEY, "baseline"]]
        offsets, support = forecast(earlier, inputs, origin, protected)
        forecasts[label] = block.assign(**offsets)
        report[label] = {"origin": str(origin.date()), **support, **evaluate(forecasts[label])}
    warm = pd.concat([forecasts[label].loc[forecasts[label].date.dt.month.eq(month)] for label, month in WARM])
    ac = pd.concat([forecasts["A"], forecasts["C"]])
    for label, frame in (("disjoint_A_C", ac), ("disjoint_warm", warm)):
        if frame.duplicated(KEY).any():
            raise ValueError("Combined scores require distinct station-hours.")
        report[label] = {**evaluate(frame), "month_halves": split_check(frame)}
    test = pd.read_csv(args.team / "data/test.csv", parse_dates=["date"], usecols=[*KEY, "example_id"])
    predictions = pd.read_csv(args.baseline, float_precision="round_trip")
    if len(test) != 157_194 or not test.example_id.equals(predictions.example_id):
        raise ValueError("The final sensitivity must follow the official test grid.")
    inputs = test[KEY].assign(baseline=np.log1p(predictions.boardings))
    offsets, support = forecast(history, inputs, pd.Timestamp("2026-06-30"), protected)
    report["target_sensitivity"] = {**support, "changed_rows": {m: int(np.count_nonzero(d)) for m, d in offsets.items()}}
    if args.submission:
        heldout = [report[label]["all"]["relative_peak_share"] for label in ("A", "B", "C", "disjoint_A_C", "disjoint_warm")]
        if any(case["candidate"] >= case["reference"] for case in heldout):
            raise ValueError("The commuting contrast must improve across both seasonal and disjoint validation.")
        report["export"] = export(args, test, predictions, offsets["relative_peak_share"], protected)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "archive", "baseline", "tabicl_oof", "retail_oof", "results"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--incumbent-report", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    if bool(args.submission) != bool(args.incumbent_report):
        parser.error("Supply the incumbent report whenever exporting an adjusted forecast.")
    report = run(args)
    args.results.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for label in ("A", "B", "C", "disjoint_A_C", "disjoint_warm"):
        print(label, report[label]["all"], flush=True)
    if "export" in report:
        print(json.dumps(report["export"]), flush=True)


if __name__ == "__main__":
    main()
