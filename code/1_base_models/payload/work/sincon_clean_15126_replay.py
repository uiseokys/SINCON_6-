"""Replay the released adjustments starting from the validation-only local stack."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sincon_festival_experiment import local_days, transferred
from sincon_festival_fact_check import predict, read_panel, source_effects
from sincon_overlay_candidate import deltas, trained_inputs
from sincon_songdo_schedule import END, START, STATIONS, aligned_submission, organizer_schedule, save_submission
from sincon_station_state import KEY, WINDOWS
from sincon_weather_sensitivity import context, score
from sincon_weekly_variability import protected_days
from sincon_xgb_conditioned import read_pair
from sincon_xgb_weather_candidate import adjustments
from sincon_zoo_transfer import verified_archive


ORIGIN = pd.Timestamp("2026-06-30")
N_ROWS = 157_194


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_change(values, delta, active):
    if not np.isfinite(delta[active]).all():
        raise ValueError("The selected changes must be finite.")
    values[active] = np.expm1(np.maximum(np.log1p(values[active]) + delta[active], 0.))


def summary(delta, active):
    return {"changed_rows": int(active.sum()), "max_abs_log_change": float(np.abs(delta[active]).max())}


def event_replay(team, grid, values, sources):
    events = local_days(sources["festival_events"])
    history = read_panel(team, sources["external_archive"])
    conditional = aligned_submission(sources["conditional_stack"], grid)
    corrected, _ = organizer_schedule(events)
    old, _ = transferred(history, grid[KEY], events, conditional)
    new, _ = transferred(history, grid[KEY], corrected, conditional)
    songdo = grid.station_id.isin(STATIONS) & grid.date.between(START, END) & grid.hour.isin([21, 22])
    change = new - old
    if songdo.sum() != 36 or np.any(np.abs(change[songdo]) <= 1e-12):
        raise ValueError("The Songdo organizer schedule must update 36 rows.")
    apply_change(values, change, songdo)
    report = {"songdo_organizer_21_22": summary(change, songdo)}

    donors = source_effects(history, events, ORIGIN)
    scopes = {
        "pungmul_friday_only": grid.date.eq("2026-10-02") & grid.station_id.isin([3119, 3120]) & grid.hour.ge(21),
        "bicof_actual_program": grid.date.between("2026-09-18", "2026-09-20") & grid.station_id.eq(3759),
    }
    event_rows = songdo.copy()
    for (variant, scope), expected in zip(scopes.items(), (6, 54), strict=True):
        if scope.sum() != expected or (scope & event_rows).any():
            raise ValueError(f"Unexpected or overlapping rows in {variant}.")
        target = grid.loc[scope, KEY].assign(conditional_stack=conditional[scope])
        delta = predict(target, events, donors, variant) - predict(target, events, donors, "original")
        if not np.isfinite(delta).all() or np.any(np.abs(delta) <= 1e-12):
            raise ValueError(f"Unexpected zero adjustment in {variant}.")
        values[scope] = np.expm1(np.maximum(np.log1p(values[scope]) + delta, 0.))
        event_rows |= scope
        report[variant] = summary(delta, np.ones(len(delta), dtype=bool))
    return event_rows.to_numpy(), report


def official_sources(args):
    work, team = args.work_dir, args.team
    return {
        "official_train": team / "data/train.csv",
        "official_test": team / "data/test.csv",
        "official_metadata": team / "data/metaData.csv",
        "local_stack": args.local_stack or work / "submission_local_stack.csv",
        "conditional_stack": args.conditional_stack or work / "submission_conditional_stack.csv",
        "festival_events": work / "sincon_festival_station_days.csv",
        "external_archive": work / "sincon_local_repro.zip",
        "baseball": work / "sincon_kbo_game_data.csv",
        "football": work / "sincon_bucheon_football_data.csv",
        "local_historical_abc": args.local_historical_abc or work / "sincon_local_validation.csv.gz",
        "final_xgb": args.final_xgb,
        **{f"local_historical_{path.name[7:17]}": path for path in sorted(args.earlier.glob("origin_*.csv.gz"))},
    }


def restore_history(bundle, directory, team):
    """Restore the published cache's omitted labels from official training rows."""
    truth = pd.read_csv(team / "data/train.csv", usecols=[*KEY, "boardings"])
    if truth.date.max() != str(ORIGIN.date()) or truth.duplicated(KEY).any():
        raise ValueError("The cached forecasts require official historical counts.")
    labels = np.log1p(truth.set_index(KEY).boardings)
    with ZipFile(bundle) as archive:
        for name in sorted(archive.namelist()):
            path = Path(name)
            if path.parent != Path("cache") or not path.match("origin_*.csv.gz"):
                continue
            rows = pd.read_csv(archive.open(name), compression="gzip")
            keys = pd.MultiIndex.from_frame(rows[KEY])
            actual = labels.reindex(keys).to_numpy()
            if not keys.is_unique or not np.isfinite(actual).all() or not rows.date.gt(rows.origin).all():
                raise ValueError(f"Invalid historical prediction keys in {name}.")
            rows.assign(y_log=actual).to_csv(directory / path.name, index=False, float_format="%.9f",
                                             compression={"method": "gzip", "mtime": 0})


def validation(args, raw, inputs, protected, fitted):
    archive, _ = verified_archive(args.work_dir, args.team, args.local_historical_abc)
    episodes = {fold: (archive.loc[archive.fold.eq(fold)], pd.Timestamp(span[0]))
                for fold, span in WINDOWS.items()}
    if args.recent is not None:
        june = pd.Timestamp("2025-06-30")
        local = args.work_dir / "sincon_summer_replay_validation.csv.gz"
        episodes["summer"] = read_pair(local, args.june_xgb, june), june
        for name in ("2025-07-31", "2025-08-31", "2025-09-30", "2026-03-31", "2026-04-30"):
            origin = pd.Timestamp(name)
            file = f"origin_{name}.csv.gz"
            local = (args.earlier if origin.year == 2025 else args.recent) / file
            alternate = args.april_xgb if name == "2026-04-30" else args.extra_xgb / file
            episodes[name] = read_pair(local, alternate, origin), origin
    truth = raw.set_index(KEY).observed
    scored, report = {}, {}
    for name, (rows, origin) in episodes.items():
        rows = rows.reset_index(drop=True)
        actual = truth.reindex(pd.MultiIndex.from_frame(rows[KEY])).to_numpy()
        if rows.duplicated(KEY).any() or rows.date.min() <= origin or not np.allclose(actual, rows.y_log, atol=1e-8, rtol=0):
            raise ValueError(f"Invalid fixed-origin validation labels for {name}.")
        raw_log = rows.local_stack.to_numpy().clip(min=0.)
        xgb_weather = adjustments(rows, raw, origin, inputs, protected)["mature_xgb_weather"]
        overlay, support = deltas(rows[[*KEY, "local_stack"]], raw_log, origin, protected, fitted)
        before_overlay = np.maximum(raw_log + xgb_weather, 0.)
        forecast = np.maximum(before_overlay + overlay["combined"], 0.)
        scored[name] = rows[KEY].assign(label=name, origin=origin, y_log=actual, local_stack=raw_log,
                                        baseline=before_overlay, combined=forecast)
        report[name] = {"origin": str(origin.date()), "rows": len(rows), "days": int(rows.date.nunique()), **support,
                        "local_to_xgb_weather": score(actual, raw_log, before_overlay, rows.date),
                        "local_to_combined": score(actual, raw_log, forecast, rows.date),
                        "xgb_weather_to_combined": score(actual, before_overlay, forecast, rows.date)}
    groups = {"disjoint_A_C": pd.concat([scored["A"], scored["C"]], ignore_index=True)}
    if args.recent is not None:
        warm = (("2025-07-31", 8), ("2025-08-31", 9), ("2025-09-30", 10))
        groups["disjoint_warm"] = pd.concat([scored[name].loc[scored[name].date.dt.month.eq(month)]
                                              for name, month in warm], ignore_index=True)
    for name, rows in groups.items():
        if rows.duplicated(KEY).any():
            raise ValueError(f"Overlapping dates for {name}.")
        actual, local, baseline, combined = (rows[key].to_numpy() for key in ("y_log", "local_stack", "baseline", "combined"))
        report[name] = {"rows": len(rows), "days": int(rows.date.nunique()),
                        "local_to_combined": score(actual, local, combined, rows.date),
                        "xgb_weather_to_combined": score(actual, baseline, combined, rows.date)}
    if args.validation_archive:
        pd.concat(scored.values(), ignore_index=True).to_pickle(args.validation_archive)
    return report


def run(args):
    sources = official_sources(args)
    if args.earlier_archive is not None:
        sources["portable_history_archive"] = args.earlier_archive
    earlier_count = sum(name.startswith("local_historical_2025") for name in sources)
    if earlier_count == 0:
        raise ValueError("The late-hour correction needs the historical rolling forecast archive.")
    grid = pd.read_csv(sources["official_test"], parse_dates=["date"], usecols=[*KEY, "example_id"])
    raw = pd.read_csv(sources["official_train"], parse_dates=["date"], usecols=[*KEY, "boardings"])
    raw["observed"] = np.log1p(raw.pop("boardings"))
    if (len(grid) != N_ROWS or not grid.example_id.is_unique or grid.duplicated(KEY).any()
            or grid.date.min() != pd.Timestamp("2026-07-01") or grid.date.max() != pd.Timestamp("2026-10-31")
            or grid.date.nunique() != 123 or raw.date.max() != ORIGIN):
        raise ValueError("The clean replay requires the full official test grid and June 30 train cutoff.")

    local = pd.read_csv(sources["local_stack"], float_precision="round_trip")
    local_log = aligned_submission(sources["local_stack"], grid)
    if not local.example_id.equals(grid.example_id):
        raise ValueError("The starting predictions must match the official test ID order.")
    values = local.boardings.to_numpy(copy=True)
    event_rows, report = event_replay(args.team, grid, values, sources)
    event_values = values[event_rows].copy()

    xgb = pd.read_csv(sources["final_xgb"], parse_dates=["date", "origin"], usecols=[*KEY, "origin", "xgb"])
    forecasts = grid[KEY].assign(local_stack=local_log).merge(xgb, on=KEY, how="left", validate="one_to_one")
    if len(xgb) != len(grid) or not forecasts[KEY].equals(grid[KEY]) or not forecasts.origin.eq(ORIGIN).all():
        raise ValueError("The XGBoost forecast must match the official keys and origin.")
    inputs = context(args.work_dir, args.team)
    protected = protected_days(args.work_dir)
    xgb_weather = adjustments(forecasts, raw, ORIGIN, inputs, protected)["mature_xgb_weather"]
    xgb_rows = ~event_rows & (np.abs(xgb_weather) > 1e-12)
    apply_change(values, xgb_weather, xgb_rows)
    report["mature_xgb_weather"] = summary(xgb_weather, xgb_rows)

    fitted = trained_inputs(raw, args.work_dir, args.earlier, *protected, local_validation=sources["local_historical_abc"])
    overlay, support = deltas(forecasts[[*KEY, "local_stack"]], local_log, ORIGIN, protected, fitted)
    overlay_rows = ~event_rows & (np.abs(overlay["combined"]) > 1e-12)
    apply_change(values, overlay["combined"], overlay_rows)
    report["geomdan_and_late"] = {**summary(overlay["combined"], overlay_rows), **support}
    if not np.array_equal(values[event_rows], event_values):
        raise ValueError("Later adjustments changed the protected event forecasts.")

    if args.submission.exists():
        written = pd.read_csv(args.submission, float_precision="round_trip")
        if (list(written.columns) != ["example_id", "boardings"] or not written.example_id.equals(grid.example_id)
                or not np.array_equal(written.boardings, values)):
            raise FileExistsError(f"Existing predictions differ from the clean replay: {args.submission}")
    else:
        save_submission(args.submission, local, grid, values)
    result = {"origin": str(ORIGIN.date()), "rows": len(values), "protected_event_rows": int(event_rows.sum()),
              "changed_rows_from_local_stack": int(np.count_nonzero(values != local.boardings)), "stages": report,
              "source_sha256": {name: checksum(path) for name, path in sources.items()},
              "submission": str(args.submission.resolve()), "submission_sha256": checksum(args.submission)}
    if args.validate:
        result["validation_scope"] = ("The replay has the original event, XGBoost, weather, and overlay order. "
                                      "The dated final event corrections are not simulated in these historical scores.")
        result["validation"] = validation(args, raw, inputs, protected, fitted)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("submission", "submission_sha256", "stages")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "earlier", "final_xgb", "submission", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--validate", action="store_true")
    for name in ("validation-archive", "recent", "extra-xgb", "june-xgb", "april-xgb",
                 "local-stack", "conditional-stack", "local-historical-abc"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    additional = (args.recent, args.extra_xgb, args.june_xgb, args.april_xgb)
    if any(value is not None for value in additional) and (not args.validate or any(value is None for value in additional)):
        parser.error("Extended validation requires --validate, --recent, --extra-xgb, --june-xgb, and --april-xgb.")
    args.earlier_archive = args.earlier if args.earlier.is_file() else None
    if args.earlier_archive is None:
        run(args)
    else:
        with TemporaryDirectory() as temporary:
            args.earlier = Path(temporary)
            restore_history(args.earlier_archive, args.earlier, args.team)
            run(args)


if __name__ == "__main__":
    main()
