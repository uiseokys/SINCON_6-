"""Measure and stage date-keyed non-KBO spectator-game demand at Munhak."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_munhak_ordinary import STADIUM, STADIUM_EVENTS, category, home_dates, robust_profile
from sincon_overlay_candidate import replay, trained_inputs
from sincon_songdo_schedule import aligned_submission
from sincon_station_state import KEY
from sincon_xgb_conditioned import load_episodes


@dataclass(frozen=True)
class Game:
    day: str
    start: float
    series: str
    finish: float | None = None

    @property
    def date(self):
        return pd.Timestamp(self.day)


HISTORY = (
    Game("2025-07-27", 18., "studio"),
    Game("2025-08-10", 18., "studio"),
    Game("2025-09-14", 17., "studio"),
    Game("2025-10-24", 18., "landers", 20 + 43 / 60),
    Game("2026-05-30", 17., "studio"),
    Game("2026-06-13", 17., "studio"),
)
FINAL = (
    Game("2026-10-17", 17.5, "landers", 20 + 23 / 60),
    Game("2026-10-31", 14., "studio"),
)
HISTORICAL_FOLDS = dict(zip(
    (game.day for game in HISTORY),
    ("summer", "2025-07-31", "2025-08-31", "2025-09-30", "A", "A"),
))
WEIGHT = .72
HOURS = np.arange(6, 24)
CUTOFF = pd.Timestamp("2026-06-30")
SOURCES = {
    **STADIUM_EVENTS,
    "2025_fan_game_timing_secondary": "https://www.namu.moe/w/SSG%20%EB%9E%9C%EB%8D%94%EC%8A%A4/2025%EB%85%84/%EC%84%AC%EA%B3%A4%EC%A0%84",
    "2025_fan_game_attendance_report": "https://sports.news.nate.com/view/20251024n34420?mid=e1400",
    "2026_fan_game_timing_secondary": "https://www.namu.moe/w/SSG%20%EB%9E%9C%EB%8D%94%EC%8A%A4/2026%EB%85%84/%EC%B5%9C%EA%B4%91%EC%95%BC%EA%B5%AC%201429",
    "2026_fan_game_organizer_attendance": "https://www.ssglanders.com/media/news/detail?idx=19944",
    "2026_fan_game_organizer_broadcast": "https://www.youtube.com/live/TzunCSawrp4",
}


def earlier_profiles(raw, protected, games):
    if raw.date.max() != CUTOFF:
        raise ValueError("Only the official historical train data is permitted.")
    station = raw.loc[raw.station_id.eq(STADIUM)]
    ordinary = station.loc[category(station, *protected, games) == "ordinary"]
    profiles = {}
    for game in HISTORY:
        if game.date in games:
            raise ValueError(f"Non-KBO historical event overlaps KBO coverage: {game.day}")
        keys = station.loc[station.date.eq(game.date), KEY].sort_values("hour")
        labels = station.loc[station.date.eq(game.date)].sort_values("hour").observed.to_numpy()
        if not np.array_equal(keys.hour, HOURS):
            raise ValueError(f"Unexpected hour grid for {game.day}")
        end = game.date - pd.Timedelta(days=1)
        typical = robust_profile(ordinary.loc[ordinary.date.le(end)], keys, end)
        residual = np.maximum(np.expm1(labels) - np.expm1(typical), 0.)
        residual[(HOURS + .5 < game.start - 1.5) | (HOURS + .5 > game.start + 6.)] = 0.
        profiles[game] = residual
    return profiles


def proposals(game, origin, baseline, profiles):
    if not origin < game.date or np.shape(baseline) != HOURS.shape:
        raise ValueError("Expected a future game with exactly eighteen incumbent hourly forecasts.")
    prior = [source for source in profiles if source.date <= origin]
    same = [source for source in prior if source.series == game.series]
    sources = same or prior
    if not sources:
        return {}, []
    finish_aligned = game.finish is not None and all(source.finish is not None for source in sources)
    relative_hour = HOURS + .5 - game.start
    active = (relative_hour >= -1.5) & (relative_hour <= 6.)
    results = {}
    for variant, offset, weight in (("fixed", 0., WEIGHT), ("earlier_half_hour", -.5, WEIGHT),
                                    ("later_half_hour", .5, WEIGHT), ("half_strength", 0., .5)):
        transferred = []
        for source in sources:
            gap = game.start - source.start
            if finish_aligned:
                finish_gap = game.finish - source.finish
                gap = np.where(relative_hour >= 1.5, finish_gap, gap)
            transferred.append(np.interp(HOURS - gap - offset, HOURS, profiles[source], left=0., right=0.))
        added = np.where(active, np.median(transferred, axis=0), 0.)
        predicted = np.log1p(np.expm1(baseline) + added)
        results[variant] = baseline + weight * (predicted - baseline)
    return results, [source.day for source in sources]


def evaluate(args):
    episodes, raw, inputs, protected, calibrated, fallback = load_episodes(args)
    games, _ = home_dates(args.work_dir)
    profiles = earlier_profiles(raw, protected, games)
    model = trained_inputs(raw, args.work_dir, args.earlier, *protected)
    historical = {}
    exact_archive = json.loads((args.work_dir / "sincon_overlay_candidate_results.json").read_text())
    replayed = {}
    for game in HISTORY:
        fold = HISTORICAL_FOLDS[game.day]
        rows, origin, _ = episodes[fold]
        if fold not in replayed:
            exact, _ = replay(rows.reset_index(drop=True), origin, raw, inputs, protected, calibrated, fallback, model)
            mse = np.mean((exact.y_log - exact.combined.clip(lower=0.))**2)
            if not np.isclose(np.sqrt(mse), exact_archive[fold]["all"]["combined"]["candidate"], atol=1e-10, rtol=0):
                raise ValueError(f"Archived exact overlay scores do not match {fold}.")
            replayed[fold] = exact
        exact = replayed[fold]
        event = exact.loc[exact.station_id.eq(STADIUM) & exact.date.eq(game.date)].sort_values("hour")
        if not np.array_equal(event.hour, HOURS):
            raise ValueError(f"Missing validation station hours on {game.day}")
        target = raw.loc[raw.station_id.eq(STADIUM) & raw.date.eq(game.date)].sort_values("hour").observed
        if not np.allclose(target, event.y_log, atol=1e-8, rtol=0):
            raise ValueError(f"Archived event labels differ from official observations on {game.day}")
        baseline, truth = event.combined.clip(lower=0.).to_numpy(), event.y_log.to_numpy()
        candidates, eligible = proposals(game, origin, baseline, profiles)
        method = {"incumbent": baseline, **candidates}
        historical[game.day] = {
            "origin": str(origin.date()), "start": game.start, "precedents": eligible,
            "sse": {name: float(np.square(truth - values).sum()) for name, values in method.items()},
            "hourly": [{"hour": int(hour), "observed": int(np.rint(np.expm1(observed))),
                        **{name: round(float(np.expm1(values[i])), 2) for name, values in method.items()}}
                       for i, (hour, observed) in enumerate(zip(HOURS, truth))],
        }
    final = {}
    grid = pd.read_csv(args.team / "data/test.csv", parse_dates=["date"], usecols=[*KEY, "example_id"])
    if len(grid) != 157_194 or grid.duplicated(KEY).any() or not grid.example_id.is_unique:
        raise ValueError("Invalid published submission grid.")
    grid["incumbent"] = aligned_submission(args.incumbent, grid)
    for game in FINAL:
        date = game.date
        if date in games or (date, STADIUM) in protected[1] or date in protected[0]:
            raise ValueError(f"Unexpected overlapping event on {game.day}")
        event = grid.loc[grid.station_id.eq(STADIUM) & grid.date.eq(date)].sort_values("hour")
        baseline = event.incumbent.to_numpy()
        candidates, eligible = proposals(game, CUTOFF, baseline, profiles)
        if not candidates or np.any(candidates["fixed"] < baseline - 1e-12):
            raise ValueError(f"Expected a nonnegative and available forecast increment on {game.day}.")
        final[game.day] = {
            "start": game.start, "finish_secondary_source": game.finish, "precedents": eligible,
            "rows_changed": int(np.count_nonzero(np.abs(candidates["fixed"] - baseline) > 1e-12)),
            "hourly": [{"hour": int(hour), "example_id": example, "station_id": STADIUM,
                        "incumbent": float(np.expm1(baseline[i])),
                        "delta_log": float(candidates["fixed"][i] - baseline[i]),
                        **{name: float(np.expm1(values[i])) for name, values in candidates.items()}}
                       for i, (hour, example) in enumerate(zip(HOURS, event.example_id))],
        }
    supported = [part for part in historical.values() if part["precedents"]]
    totals = {method: sum(part["sse"][method] for part in supported) for method in supported[0]["sse"]}
    warm = [historical[day] for day in ("2025-08-10", "2025-09-14", "2025-10-24")]
    return {"weight_log_space": WEIGHT, "sources": SOURCES, "historical_events": historical,
            "supported_event_rows": len(supported) * len(HOURS),
            "supported_event_rmsle": {name: float(np.sqrt(sse / (len(supported) * len(HOURS)))) for name, sse in totals.items()},
            "disjoint_warm_event_sse": {name: sum(part["sse"][name] for part in warm) for name in totals},
            "incumbent_file": str(args.incumbent.resolve()), "final": final}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "earlier", "recent", "extra_xgb", "june_xgb", "april_xgb"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--incumbent", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args)
    path = args.work_dir / "sincon_munhak_spectator_results.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"supported_event_rmsle": result["supported_event_rmsle"],
                      "disjoint_warm_event_sse": result["disjoint_warm_event_sse"]}, indent=2), flush=True)
