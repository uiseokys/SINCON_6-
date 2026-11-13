"""Reproduce the held-out TabICL blend as a keyed final-origin log adjustment."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from tabicl import TabICLRegressor

from sincon_joint_candidate import setup
from sincon_network_level_audit import STABLE_EXCLUSIONS, ordinary
from sincon_songdo_schedule import CHAMPION
from sincon_station_state import KEY
from sincon_tabicl_daily_deviation import DAY, DISAGREEMENT_CAP, WEIGHT, deviations
from sincon_tabicl_experiment import forecast, station_days
from sincon_weekly_variability import protected_days


CUTOFF = pd.Timestamp("2026-06-30")
LAST = pd.Timestamp("2026-10-31")
BASELINE = "submission_station_profiles_event_program_level_neutral.csv"
LAYERS = (
    ("three_schedule_corrections", CHAMPION, "submission_local_stack_level_station_event_schedule.csv", 96),
    ("spectator_fixtures", "submission_station_profile_munhak_geomdan.csv", "submission_station_profiles_spectator.csv", 15),
    ("canceled_concerts", "submission_station_profiles_spectator.csv", "submission_station_profiles_cancelled.csv", 72),
    ("songdo_venue", "submission_station_profiles_cancelled.csv", "submission_station_profiles_event_audit.csv", 24),
    ("event_timing_and_program", "submission_station_profiles_event_audit.csv", "submission_station_profiles_event_program.csv", 20),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aligned_counts(path, grid):
    counts = pd.read_csv(path, float_precision="round_trip")
    if (list(counts) != ["example_id", "boardings"] or not counts.example_id.equals(grid.example_id)
            or not np.isfinite(counts.boardings).all() or counts.boardings.lt(0.).any()):
        raise ValueError(f"{path.name} must have valid counts in official test order.")
    return counts.boardings


def protect_previous_layers(grid, baseline, work, team):
    masks = {}
    for layer, prior_name, next_name, expected in LAYERS:
        prior_dir = team if prior_name == CHAMPION else work
        prior = aligned_counts(prior_dir / prior_name, grid)
        next_ = aligned_counts(work / next_name, grid)
        changed = (prior - next_).abs().gt(1e-9)
        if changed.sum() != expected or not np.array_equal(baseline[changed], next_[changed]):
            raise ValueError(f"The incumbent does not retain all {expected} {layer} row edits.")
        masks[layer] = changed
    retained = pd.DataFrame(masks).any(axis=1)
    holidays, event_days = protected_days(work)
    # Protect complete station-days so the blend never changes a retained hourly shape.
    event_days = event_days.union(pd.MultiIndex.from_frame(grid.loc[retained, DAY]))
    return (holidays, event_days), masks


def stage(args):
    team, work = args.team, args.work_dir
    train_path = team / "data/train.csv"
    test_path = team / "data/test.csv"
    metadata_path = team / "data/metaData.csv"
    baseline_path = work / BASELINE
    scoreboard = work / "sincon_kbo_game_data.csv"
    grid = pd.read_csv(test_path, usecols=[*KEY, "example_id"], parse_dates=["date"])
    full_grid = pd.MultiIndex.from_product([
        pd.date_range(CUTOFF + pd.Timedelta(days=1), LAST), sorted(grid.station_id.unique()), range(6, 24),
    ], names=KEY)
    key = pd.MultiIndex.from_frame(grid[KEY])
    if (len(grid) != 157_194 or not grid.example_id.is_unique or not key.is_unique
            or not key.sort_values().equals(full_grid)):
        raise ValueError("The final adjustment requires the full 123-day official test grid.")
    if digest(baseline_path) != json.loads((work / "sincon_network_level_audit_results.json").read_text())["export"]["sha256"]:
        raise ValueError("The incumbent must match the level-neutral release checksum.")
    incumbent = aligned_counts(baseline_path, grid)
    protected, masks = protect_previous_layers(grid, incumbent, work, team)
    data, _, _, _, stations, _, _ = setup(team, args.external, scoreboard, work / "mobility_monthly",
                                          work / "sincon_station_addresses.csv", 3)
    dated = data.loc[data.boardings.notna(), "date"]
    if (len(dated) != 688_176 or dated.min() != pd.Timestamp("2025-01-01") or dated.max() != CUTOFF
            or data.loc[data.date.gt(CUTOFF), "boardings"].notna().any()):
        raise ValueError("TabICL can use official observations only through 2026-06-30.")
    table, daily = station_days(data)
    model = TabICLRegressor(model_path=args.checkpoint, allow_auto_download=False, n_estimators=2,
                            device="cpu", n_jobs=3, random_state=2026)
    alt = forecast(table, daily, CUTOFF, LAST, stations.index, model, 1).set_index(DAY).tabicl
    alternative = alt.reindex(pd.MultiIndex.from_frame(grid[DAY])).to_numpy()
    if len(alt) != 8_733 or not np.isfinite(alternative).all():
        raise ValueError("Every final station-day needs a finite TabICL forecast.")
    future = grid[KEY].assign(incumbent=np.log1p(incumbent), alternative=alternative)
    delta = deviations(future, protected) - future.incumbent.to_numpy()
    eligible = ordinary(future, protected) & ~future.station_id.isin(STABLE_EXCLUSIONS)
    changed = delta != 0.
    if not changed.any() or np.any(changed & ~eligible):
        raise ValueError("The final delta must change ordinary eligible rows only.")
    daily_delta = grid[DAY].assign(delta=delta).groupby(DAY).delta.agg(["min", "max", "mean"])
    group_delta = daily_delta.reset_index().assign(month=lambda f: f.date.dt.month,
                                                    weekday=lambda f: f.date.dt.day_of_week)
    if (np.max(daily_delta["max"] - daily_delta["min"]) > 1e-12
            or group_delta.groupby(["station_id", "month", "weekday"])["mean"].sum().abs().max() > 1e-12):
        raise ValueError("Final station-days and month-weekday groups must retain their hourly shapes and means.")
    result = grid[["example_id", *KEY]].assign(log_delta=delta)
    encoded = result.to_csv(index=False, date_format="%Y-%m-%d", float_format="%.17g").encode("utf-8")
    with NamedTemporaryFile(dir=args.delta.parent, delete=False) as temporary:
        temporary.write(gzip.compress(encoded, mtime=0))
    Path(temporary.name).replace(args.delta)
    return {
        "cutoff": str(CUTOFF.date()), "last": str(LAST.date()),
        "checkpoint": "TabICL v2", "ensemble_members": 2, "random_state": 2026,
        "weight": WEIGHT, "disagreement_cap": DISAGREEMENT_CAP, "train_rows": len(dated), "delta_rows": len(result),
        "changed_rows": int(np.count_nonzero(changed)), "changed_station_days": int(daily_delta["mean"].ne(0.).sum()),
        "max_abs_log_delta": float(np.abs(delta).max()), "rms_log_delta": float(np.sqrt(np.square(delta).mean())),
        "protected_prior_rows": {layer: int(mask.sum()) for layer, mask in masks.items()},
        "distinct_protected_prior_rows": int(pd.DataFrame(masks).any(axis=1).sum()),
        "excluded_stations": list(STABLE_EXCLUSIONS), "delta": str(args.delta.resolve()),
        "sha256": {name: digest(path) for name, path in
                   {"delta": args.delta, "incumbent": baseline_path, "train": train_path,
                    "test": test_path, "metadata": metadata_path, "external": args.external,
                    "checkpoint": args.checkpoint, "scoreboard": scoreboard}.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "external", "checkpoint", "delta", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.delta.name != "sincon_tabicl_final_log_delta.csv.gz" or args.report.name != "sincon_tabicl_final_log_delta_results.json":
        parser.error("Specify the TabICL delta and report filenames.")
    result = stage(args)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
