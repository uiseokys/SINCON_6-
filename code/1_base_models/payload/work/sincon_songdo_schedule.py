"""Replay the final forecast with the organizer's nine-day Songdo Beer schedule."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from sincon_apply_final_calibration import calibrate
from sincon_complementary_replay import FORECAST_ORIGIN
from sincon_daily_network import prepare_data
from sincon_festival_experiment import local_days, transferred
from sincon_housing_growth import forecast_changes, read_exposure
from sincon_station_state import KEY, read_team


DAY = KEY[:2]
START = pd.Timestamp("2026-08-22")
END = pd.Timestamp("2026-08-30")
STATIONS = (3138, 3139)
SOURCE = "https://www.songdobeer.com/"
CHAMPION = "submission_local_stack_level_station.csv"


def organizer_schedule(events):
    current = events.series.eq("songdo_beer") & events.date.dt.year.eq(2026)
    prior = events.loc[current]
    expected = pd.MultiIndex.from_product([pd.date_range("2026-08-26", END), STATIONS], names=DAY)
    if not pd.MultiIndex.from_frame(prior[DAY]).sort_values().equals(expected):
        raise ValueError("The frozen input must contain both stations on the original five days.")
    dates = pd.DataFrame({"date": pd.date_range(START, END)})
    template = prior.sort_values("date").drop_duplicates("station_id").drop(columns="date")
    corrected = template.merge(dates, how="cross").assign(
        event_start=START, days=len(dates), status="organizer", source=SOURCE,
    )
    corrected["day_index"] = (corrected.date - START).dt.days + 1
    # Preserve the source's past-edition total. The official dates give a
    # denominator for that proxy, while most actual daily attendance is unknown.
    corrected["attend_day"] = corrected.previous_visitors / corrected.days
    updated = pd.concat([events.loc[~current], corrected], ignore_index=True)
    return updated, pd.MultiIndex.from_frame(corrected[DAY]).difference(expected)


def aligned_submission(path, grid):
    data = pd.read_csv(path, float_precision="round_trip")
    if (list(data.columns) != ["example_id", "boardings"] or not data.example_id.is_unique
            or len(data) != len(grid)):
        raise ValueError(f"Invalid submission format: {path}")
    values = data.set_index("example_id").boardings.reindex(grid.example_id).to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Missing or invalid predictions: {path}")
    return np.log1p(values)


def save_submission(path, champion, keys, forecasts):
    if path.exists():
        raise FileExistsError(path)
    values = pd.Series(forecasts, index=keys.example_id).reindex(champion.example_id).to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("The final forecasts must be finite nonnegative counts.")
    with TemporaryDirectory(dir=path.parent) as directory:
        pending = Path(directory) / path.name
        champion.assign(boardings=values).to_csv(pending, index=False)
        written = pd.read_csv(pending, float_precision="round_trip")
        if not written.example_id.equals(champion.example_id) or not np.array_equal(written.boardings, values):
            raise ValueError("The exported CSV does not preserve test IDs and predictions.")
        pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "external"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--variant", choices=("full_schedule", "late_schedule", "late_missing", "hours_21_22"))
    args = parser.parse_args()
    if bool(args.submission) != bool(args.variant):
        parser.error("Use --submission and --variant together to export an evaluated variant.")
    work, team = args.work_dir, args.team
    origin = FORECAST_ORIGIN
    panel, _, _ = prepare_data(team, args.external, work / "sincon_kbo_game_data.csv")
    train, meta, calendar, _ = read_team(team, args.external)
    test = pd.read_csv(team / "data/test.csv", parse_dates=["date"])
    grid = test.sort_values(KEY).reset_index(drop=True)
    history = panel.loc[panel.date.le(origin)]
    target = panel.loc[panel.date.gt(origin)].assign(boardings=np.nan)
    if (len(grid) != 157_194 or not grid.example_id.is_unique or not target[KEY].reset_index(drop=True).equals(grid[KEY])
            or train.date.max() != origin):
        raise ValueError("Unexpected final test grid or training cutoff.")

    events = local_days(work / "sincon_festival_station_days.csv")
    corrected_events, added = organizer_schedule(events)
    frozen = aligned_submission(work / "submission_local_stack.csv", grid)
    conditional = aligned_submission(work / "submission_conditional_stack.csv", grid)
    old_transfer, _ = transferred(history, target, events, conditional)
    new_transfer, _ = transferred(history, target, corrected_events, conditional)
    festival_delta = new_transfer - old_transfer
    event_rows = grid.station_id.isin(STATIONS) & grid.date.between(START, END)
    if np.abs(festival_delta[~event_rows]).max() > 1e-12:
        raise ValueError("The revised festival transfer changed unrelated base forecasts.")
    centers = pd.read_csv(work / "sincon_station_centers.csv").set_index("station_id").reindex(meta.index)
    exposure = read_exposure(work / "sincon_housing_approvals.csv", centers)
    housing, _, _ = forecast_changes(history, grid[KEY], origin, exposure, calendar, meta)
    base_error = np.max(np.abs(frozen - (old_transfer + .5 * housing).clip(min=0)))
    if base_error > 1e-8:
        raise ValueError(f"The frozen conditional and local CSVs do not replay together: {base_error:.3g}")
    print(f"Verified original event transfer and housing on {len(grid):,} rows, max error {base_error:.3g}", flush=True)

    reference = pd.read_csv(team / CHAMPION, float_precision="round_trip")
    local = pd.read_csv(work / "submission_local_stack.csv", float_precision="round_trip")
    bias = pd.read_csv(team / "station_backtest_bias.csv", float_precision="round_trip")
    baseline = calibrate(local, test, bias)
    if (not reference.example_id.equals(test.example_id) or not baseline.example_id.equals(test.example_id)
            or not np.allclose(reference.boardings, baseline.boardings, atol=1e-8, rtol=0)):
        raise ValueError("The current Kaggle champion must replay from the frozen local forecast and calibration.")
    incumbent = reference.set_index("example_id").boardings.reindex(grid.example_id).to_numpy()
    missing = pd.MultiIndex.from_frame(grid[DAY]).isin(added)
    late = grid.hour.ge(21)
    scopes = {
        "full_schedule": event_rows,
        "late_schedule": event_rows & late,
        "late_missing": missing & late,
        "hours_21_22": event_rows & grid.hour.isin([21, 22]),
    }
    forecasts = {}
    summary = {}
    for name, scope in scopes.items():
        active = scope & (np.abs(festival_delta) > 1e-12)
        forecasts[name] = incumbent.copy()
        forecasts[name][active] = np.expm1(np.maximum(np.log1p(incumbent[active]) + festival_delta[active], 0))
        summary[name] = {"changed_rows": int(active.sum()), "late_changed_rows": int((active & late).sum())}

    details = grid[[*KEY, "example_id"]].assign(
        original=incumbent, festival_delta=festival_delta, **forecasts,
    ).loc[event_rows]
    details.to_csv(work / "sincon_songdo_schedule_deltas.csv", index=False)
    daily = details.loc[details.hour.ge(21)].groupby(DAY)[["original", *scopes]].sum().reset_index()
    report = {
        "schedule_source": SOURCE,
        "added_station_days": len(added),
        "unchanged_validation_folds": ["A", "B", "C"],
        "reference": str(team / CHAMPION),
        "prediction_rows": len(grid),
        "base_replay_max_log_error": float(base_error),
        "variants": summary,
        "late_sums": json.loads(daily.to_json(orient="records", date_format="iso")),
    }
    if args.submission:
        save_submission(args.submission, reference, grid, forecasts[args.variant])
        report["submission"] = {"path": str(args.submission.resolve()), "variant": args.variant}
    (work / "sincon_songdo_schedule_results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "late_sums"}), flush=True)


if __name__ == "__main__":
    main()
