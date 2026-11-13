"""Compare late profile and Geomdan opening adjustments on the XGBoost blend."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_fuel_shock import read_archive
from sincon_geomdan_opening_control import OPENED, OLD, fit_control, panel as corridor_panel, predict as corridor_predict
from sincon_late_composition_kernel import apply, build_history, panel, proposals, weights
from sincon_songdo_schedule import CHAMPION, aligned_submission, save_submission
from sincon_station_state import KEY
from sincon_weather_sensitivity import score
from sincon_xgb_conditioned import COMPONENTS, load_episodes, prepare_episode
from sincon_zoo_transfer import SEASON_WEIGHT


BASE_CSV = "submission_local_stack_level_station_events_xgb_weather.csv"
EVENT_CSV = "submission_local_stack_level_station_event_schedule.csv"
LATEST_CSV = "submission_local_stack_level_station_events_xgb_weather_geomdan_late.csv"
VARIANTS = ("late_profile", "opening_control", "combined")


def trained_inputs(raw, work, earlier, holidays, events, local_validation=None):
    values = raw.assign(boardings=np.rint(np.expm1(raw.observed)))
    corridor = corridor_panel(values, "boardings", holidays, events)
    model = fit_control(corridor.loc[corridor.index.get_level_values("date") < OPENED])
    late, daily_logs = panel(raw, ["observed"], holidays, events)
    _, archive = read_archive(local_validation or work / "sincon_local_validation.csv.gz", earlier)
    _, residuals = build_history(late, daily_logs["observed"], archive, holidays, events)
    return corridor, model, late, daily_logs["observed"], residuals


def deltas(future, base_calibrated, origin, protected, fitted):
    holidays, events = protected
    corridor, model, past, logs, archived_errors = fitted
    raw = future.local_stack.to_numpy()
    target, values = proposals(past, logs, future, origin, holidays, events)
    coefficients, days = weights(archived_errors, target, origin)
    late = apply(future, raw, target, values, coefficients) - raw
    before = corridor.index.get_level_values("date") <= origin
    baseline = np.maximum(base_calibrated, 0.)
    candidates, support = corridor_predict(corridor.loc[before], future[KEY].assign(base=baseline), origin, holidays, events, model)
    opening = candidates["incumbents"] - baseline
    opening = np.where(np.abs(opening) > 1e-12, opening, 0.)
    protected_rows = future.date.isin(holidays) | pd.MultiIndex.from_frame(future[KEY[:2]]).isin(events)
    if (not np.isfinite(np.column_stack([opening, late])).all() or np.any(opening[~future.station_id.isin(OLD)])
            or np.any(late[future.hour.lt(21)]) or np.any(opening[protected_rows]) or np.any(late[protected_rows])):
        raise ValueError("The overlays must be finite and only affect their eligible keys.")
    return {"late_profile": late, "opening_control": opening, "combined": late + opening}, {
        "preorigin_late_fit_dates": days, "opening_supported_cells": support["recent_effect_cells"],
    }


def replay(rows, origin, raw, inputs, protected, calibrated, fallback, fitted):
    frame, days = prepare_episode(rows, origin, raw, inputs, protected, calibrated, fallback)
    baseline = frame.reference.to_numpy() + SEASON_WEIGHT * frame[COMPONENTS].sum(axis=1).to_numpy()
    corrected = frame.local_stack.to_numpy() + frame.station_offset.to_numpy()
    changes, support = deltas(rows[[*KEY, "local_stack"]], corrected, origin, protected, fitted)
    scored = rows[[*KEY, "y_log"]].assign(baseline=baseline, **{key: baseline + value for key, value in changes.items()})
    return scored, {"calibration_days": days, **support}


def measurements(frame):
    masks = {"all": np.ones(len(frame), dtype=bool), "late_21_23": frame.hour.ge(21).to_numpy(),
             "gyeyang_gyulhyeon": frame.station_id.isin(OLD).to_numpy()}
    actual, base = frame.y_log.to_numpy(), frame.baseline.to_numpy()
    return {name: {variant: score(actual[keep], base[keep], frame[variant].to_numpy()[keep], frame.date[keep])
                   for variant in VARIANTS} for name, keep in masks.items()}


def export(args, protected, fitted):
    work, team = args.work_dir, args.team
    origin = pd.Timestamp("2026-06-30")
    grid = pd.read_csv(team / "data/test.csv", parse_dates=["date"], usecols=[*KEY, "example_id"])
    baseline = pd.read_csv(work / BASE_CSV, float_precision="round_trip")
    event_release = pd.read_csv(work / EVENT_CSV, float_precision="round_trip")
    champion = pd.read_csv(team / CHAMPION, float_precision="round_trip")
    if (len(grid) != 157_194 or grid.duplicated(KEY).any() or not grid.example_id.is_unique
            or not all(grid.example_id.equals(frame.example_id) for frame in (baseline, event_release, champion))):
        raise ValueError("The overlay needs all published forecast rows in exact test order.")
    old = aligned_submission(team / CHAMPION, grid)
    raw = aligned_submission(work / "submission_local_stack.csv", grid)
    changes, support = deltas(grid[KEY].assign(local_stack=raw), old, origin, protected, fitted)
    event_rows = event_release.boardings.ne(champion.boardings).to_numpy()
    if event_rows.sum() != 96 or not np.array_equal(baseline.boardings[event_rows], event_release.boardings[event_rows]):
        raise ValueError("The existing event forecasts must match the verified 96 event updates.")
    updates = np.where(event_rows, 0., changes[args.variant])
    changed = np.abs(updates) > 1e-12
    values = baseline.boardings.to_numpy(copy=True)
    values[changed] = np.expm1(np.maximum(np.log1p(values[changed]) + updates[changed], 0.))
    if args.submission:
        save_submission(args.submission, baseline, grid, values)
    return {"rows": len(values), "changed": int(np.count_nonzero(values != baseline.boardings)),
            "variant": args.variant, "event_rows_preserved": int(event_rows.sum()),
            "maximum_log_change": float(np.abs(updates).max()), "log_change_rms": float(np.sqrt(np.mean(updates**2))),
            "submission": str(args.submission.resolve()) if args.submission else None, **support}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "earlier", "recent", "extra_xgb", "june_xgb", "april_xgb"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--variant", choices=VARIANTS, default="combined")
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    episodes, raw, inputs, protected, calibrated, fallback_archive = load_episodes(args)
    fitted = trained_inputs(raw, args.work_dir, args.earlier, *protected)
    truth = raw.set_index(KEY).observed
    reports, scored = {}, {}
    for name, (rows, origin, _) in episodes.items():
        rows = rows.reset_index(drop=True)
        index = pd.MultiIndex.from_frame(rows[KEY])
        if not index.is_unique or rows.date.min() <= origin or not np.allclose(truth.reindex(index), rows.y_log, atol=1e-8, rtol=0):
            raise ValueError(f"Archive validation keys do not match official data for {name}.")
        scored[name], support = replay(rows, origin, raw, inputs, protected, calibrated, fallback_archive, fitted)
        reports[name] = {"origin": str(origin.date()), **support, **measurements(scored[name])}

    warm = pd.concat([scored[name].loc[scored[name].date.dt.month.eq(month)] for name, month in
                      (("2025-07-31", 8), ("2025-08-31", 9), ("2025-09-30", 10))], ignore_index=True)
    ac = pd.concat([scored[name] for name in ("A", "C")], ignore_index=True)
    for name, block in (("disjoint_warm", warm), ("disjoint_A_C", ac)):
        if block.duplicated(KEY).any():
            raise ValueError(f"The pooled validation dates overlap in {name}.")
        reports[name] = {"days": block.date.nunique(), **measurements(block)}
    reports["final"] = export(args, protected, fitted)
    path = args.work_dir / "sincon_overlay_candidate_results.json"
    path.write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: part.get("all", part) for name, part in reports.items()}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
