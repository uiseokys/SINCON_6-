"""Validate and export ordinary-day adjustments at Munhak and Geomdan Lake Park."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_3107_mature_hour_profile import HOURS, STATION, adjustment as geomdan_adjustment
from sincon_munhak_ordinary import STADIUM, WARM, episode_context, historical_adjustment, home_dates
from sincon_munhak_spectator import FINAL, HISTORY, earlier_profiles, proposals as spectator_proposals
from sincon_overlay_candidate import EVENT_CSV, LATEST_CSV, replay, trained_inputs
from sincon_songdo_schedule import CHAMPION, save_submission
from sincon_station_state import KEY
from sincon_weather_sensitivity import score
from sincon_xgb_conditioned import load_episodes


def station_adjustments(history, future, origin, protected, games):
    if history.date.max() > origin or future.date.min() <= origin or {"y_log", "boardings", "observed"} & set(future):
        raise ValueError("Station adjustments must fit past counts and use unlabeled forecasts.")
    target = episode_context(future, history, origin, *protected, games)
    geometry, support = geomdan_adjustment(history, future, origin, protected)
    key = pd.MultiIndex.from_frame(future[KEY])
    stadium = pd.Series(historical_adjustment(target), index=pd.MultiIndex.from_frame(target[KEY]))
    stadium = stadium.reindex(key, fill_value=0.).to_numpy()
    lake = geometry["recent_relative_to_neighbors"]
    if (not np.isfinite(np.column_stack([stadium, lake])).all() or np.any(stadium[future.station_id.ne(STADIUM)])
            or np.any(lake[future.station_id.ne(STATION) | ~future.hour.isin(HOURS)])):
        raise ValueError("Only ordinary Munhak and Geomdan Lake Park hours can change.")
    return {"munhak": stadium, "geomdan": lake, "joint": stadium + lake}, support


def add_spectator_events(changes, future, fixtures, origin, historical):
    added = np.zeros(len(future))
    support = {}
    for game in fixtures:
        selected = future.date.eq(game.date) & future.station_id.eq(STADIUM)
        if not selected.any():
            continue
        positions = np.flatnonzero(selected)
        ordered = positions[np.argsort(future.hour.to_numpy()[positions])]
        if (not np.array_equal(future.hour.to_numpy()[ordered], np.arange(6, 24))
                or np.any(changes["joint"][ordered])):
            raise ValueError("Spectator forecasts require eighteen distinct hours and no ordinary-day adjustment.")
        baseline = future.combined.to_numpy()[ordered].clip(min=0.)
        candidates, precedents = spectator_proposals(game, origin, baseline, historical)
        if candidates:
            added[ordered] = candidates["fixed"] - baseline
        support[game.day] = {"precedents": precedents, "rows": int(np.count_nonzero(added[ordered]))}
    if not np.isfinite(added).all() or (added < -1e-12).any():
        raise ValueError("Spectator demand increments must be finite and nonnegative.")
    return {**changes, "spectator": added, "joint_with_spectator": changes["joint"] + added}, support


def measures(rows):
    actual, base = rows.y_log.to_numpy(), rows.combined.to_numpy()
    return {name: {**score(actual, base, base + rows[name].to_numpy(), rows.date),
                   "changed_rows": int(rows[name].ne(0.).sum())} for name in
            ("munhak", "geomdan", "joint", "spectator", "joint_with_spectator") if name in rows}


def export(args, history, protected, games, historical_events):
    origin = pd.Timestamp("2026-06-30")
    work = args.work_dir
    grid = pd.read_csv(args.team / "data/test.csv", usecols=[*KEY, "example_id"], parse_dates=["date"])
    baseline = pd.read_csv(work / LATEST_CSV, float_precision="round_trip")
    champion = pd.read_csv(args.team / CHAMPION, float_precision="round_trip")
    event = pd.read_csv(work / EVENT_CSV, float_precision="round_trip")
    if (len(grid) != 157_194 or grid.duplicated(KEY).any() or not grid.example_id.is_unique
            or any(not table.example_id.equals(grid.example_id) for table in (baseline, champion, event))):
        raise ValueError("All inputs must match the official test grid and ID order.")
    if not np.isfinite(baseline.boardings).all() or baseline.boardings.lt(0.).any():
        raise ValueError("The reference CSV must have finite nonnegative counts.")
    future = grid[KEY].assign(combined=np.log1p(baseline.boardings))
    values, support = station_adjustments(history, future, origin, protected, games)
    method = "joint"
    if historical_events is not None:
        if any(game.date in games or game.date in protected[0] or (game.date, STADIUM) in protected[1] for game in FINAL):
            raise ValueError("Spectator events must be distinct from known baseball, holidays and prior protected events.")
        values, event_support = add_spectator_events(values, future, FINAL, origin, historical_events)
        support["spectator"] = event_support
        method = "joint_with_spectator"
    event_rows = event.boardings.ne(champion.boardings).to_numpy()
    if event_rows.sum() != 96 or not baseline.boardings[event_rows].equals(event.boardings[event_rows]):
        raise ValueError("Expected 96 previously verified event updates.")
    if np.any(values[method][event_rows]):
        raise ValueError("Ordinary-day adjustments overlap verified event forecasts.")
    modified = baseline.boardings.to_numpy(copy=True)
    use = values[method] != 0.
    modified[use] = np.expm1((future.combined.to_numpy()[use] + values[method][use]).clip(min=0.))
    if args.submission:
        save_submission(args.submission, baseline, grid, modified)
    return {"rows": len(grid), "changed_rows": int(np.count_nonzero(modified != baseline.boardings)),
            "changed_by_station": {str(int(station)): int(count) for station, count in grid.loc[use, "station_id"].value_counts().items()},
            "event_rows_preserved": int(event_rows.sum()), "support": support,
            "submission": str(args.submission.resolve()) if args.submission else None,
            "sha256": hashlib.sha256(args.submission.read_bytes()).hexdigest() if args.submission else None}


def run(args):
    episodes, raw, inputs, protected, calibrated, fallback = load_episodes(args)
    fitted = trained_inputs(raw, args.work_dir, args.earlier, *protected)
    games, _ = home_dates(args.work_dir)
    historical_events = earlier_profiles(raw, protected, games) if args.spectator_events else None
    truth = raw.set_index(KEY).observed
    official = json.loads((args.work_dir / "sincon_overlay_candidate_results.json").read_text())
    cases, report = {}, {}
    for name, (rows, origin, _) in episodes.items():
        rows = rows.reset_index(drop=True)
        keys = pd.MultiIndex.from_frame(rows[KEY])
        if not keys.is_unique or not np.allclose(truth.reindex(keys), rows.y_log, atol=1e-8, rtol=0):
            raise ValueError(f"Validation keys and labels differ from official data in {name}.")
        base, _ = replay(rows, origin, raw, inputs, protected, calibrated, fallback, fitted)
        incumbent = np.sqrt(np.square(base.y_log - base.combined.clip(lower=0.)).mean())
        if abs(incumbent - official[name]["all"]["combined"]["candidate"]) > 1e-10:
            raise ValueError(f"Reference overlay replay differs in {name}.")
        future = base[[*KEY, "combined"]]
        changes, support = station_adjustments(raw.loc[raw.date.le(origin)], future, origin, protected, games)
        if historical_events is not None:
            changes, event_support = add_spectator_events(changes, future, HISTORY, origin, historical_events)
            support["spectator"] = event_support
        cases[name] = base.assign(**changes)
        report[name] = {"origin": str(origin.date()), "support": support, **measures(cases[name])}
    warm = pd.concat([cases[name].loc[cases[name].date.dt.month.eq(month)] for name, month in WARM], ignore_index=True)
    ac = pd.concat([cases["A"], cases["C"]], ignore_index=True)
    for name, block in (("disjoint_A_C", ac), ("disjoint_warm", warm)):
        if block.duplicated(KEY).any():
            raise ValueError(f"Repeated forecast keys in {name}.")
        report[name] = {"days": block.date.nunique(), **measures(block)}
    report["final"] = export(args, raw, protected, games, historical_events)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "earlier", "recent", "extra_xgb", "june_xgb", "april_xgb"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--spectator-events", action="store_true")
    args = parser.parse_args()
    report = run(args)
    stem = "sincon_station_profile_spectator" if args.spectator_events else "sincon_station_profile_candidate"
    report_path = args.report or args.work_dir / f"{stem}_results.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    primary = "joint_with_spectator" if args.spectator_events else "joint"
    print(json.dumps({name: item.get(primary, item) for name, item in report.items()}), flush=True)


if __name__ == "__main__":
    main()
