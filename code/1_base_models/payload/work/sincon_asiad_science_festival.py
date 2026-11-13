"""Audit one historical science festival and project a conservative daytime increment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_asiad_concert_spillover import EVENTS as CONCERTS, proposals as concert_proposals
from sincon_munhak_ordinary import home_dates
from sincon_munhak_spectator import HISTORY, earlier_profiles
from sincon_overlay_candidate import replay, trained_inputs
from sincon_station_profile_candidate import add_spectator_events, station_adjustments
from sincon_station_state import KEY
from sincon_weekday_contrast import ordinary_dates_and_events
from sincon_xgb_conditioned import load_episodes


STATIONS = (3209, 3210)
HOURS = list(range(10, 18))
DEPARTURE = [15, 16, 17]
DATES = tuple(zip(pd.to_datetime(["2025-10-24", "2025-10-25"]), pd.to_datetime(["2026-10-23", "2026-10-24"])))
BLOCKED = pd.DatetimeIndex([date for pair in DATES for date in pair] + list(CONCERTS) + [pd.Timestamp("2025-09-20")])
SOURCES = {
    "venue_access": "https://www.insiseol.or.kr/sport/asiad/introduce/location.jsp",
    "2025_education_office": "https://www.ice.go.kr/ice/na/ntt/selectNttInfo.do?mi=11620&nttSn=3343366",
    "2025_program": "https://isw.icehs.kr/boardCnts/updateCnt.do?action=view&boardID=36901&boardSeq=33225728&lev=0&m=1001&viewBoardID=36901",
    "2026_education_office": "https://www.ice.go.kr/ice/na/ntt/selectNttInfo.do?bbsAllView=Y&bbsId=1499&mi=&nttSn=3387242",
    "2026_program": "https://www.icemf.kr/",
    "2026_reported_aggregate_participation": "https://www.ice.go.kr/ice/na/ntt/selectNttInfo.do?mi=&nttSn=3387304",
}


def controls(dates, ordinary, day):
    dates = pd.DatetimeIndex(dates.unique()).intersection(ordinary).sort_values()
    return dates[(dates < day) & (dates.day_of_week == day.day_of_week)][-6:]


def profile(frame, date, column):
    values = frame.loc[frame.date.eq(date)].set_index("hour")[column].reindex(HOURS)
    if values.isna().any():
        raise ValueError(f"Missing hour for {date.date()} and {column}.")
    return np.expm1(values.to_numpy())


def comparison(frame, dates):
    selected = frame.loc[frame.date.isin(dates) & frame.hour.isin(HOURS)]
    result = {}
    for name, rows in (("daytime", selected), ("departure", selected.loc[selected.hour.isin(DEPARTURE)])):
        if len(rows):
            result[name] = {
                "rows": len(rows),
                "actual": float(np.expm1(rows.y_log).sum()),
                "forecast": float(np.expm1(rows.forecast).sum()),
                "rmsle": float(np.sqrt(np.square(rows.y_log - rows.forecast).mean())),
                "mean_log_residual": float((rows.y_log - rows.forecast).mean()),
            }
    return result


def replay_case(name, episode, raw, inputs, protected, calibrated, fallback, fitted, games, spectator, expected):
    rows, origin, _ = episode
    rows = rows.reset_index(drop=True)
    keys = pd.MultiIndex.from_frame(rows[KEY])
    if not keys.is_unique or rows.date.min() <= origin or not np.allclose(
        raw.set_index(KEY).observed.reindex(keys), rows.y_log, atol=1e-8, rtol=0
    ):
        raise ValueError(f"Validation observations or keys do not match in {name}.")
    base, _ = replay(rows, origin, raw, inputs, protected, calibrated, fallback, fitted)
    future = base[[*KEY, "combined"]]
    history = raw.loc[raw.date.le(origin)]
    layer, _ = station_adjustments(history, future, origin, protected, games)
    layer, _ = add_spectator_events(layer, future, HISTORY, origin, spectator)
    future = base[KEY].assign(incumbent=base.combined + layer["joint_with_spectator"])
    adjusted, _ = concert_proposals(future, history, origin, protected, include_placebos=False)
    result = base[KEY].assign(y_log=base.y_log, forecast=adjusted.three_quarters_excess)
    score = np.sqrt(np.square(result.y_log - result.forecast).mean())
    if not np.isclose(score, expected, atol=1e-10, rtol=0):
        raise ValueError(f"Exact spectator + concert baseline does not reproduce in {name}.")
    return origin, result.loc[result.station_id.isin(STATIONS)].copy(), float(score)


def run(args):
    work = args.work_dir
    episodes, raw, inputs, protected, calibrated, fallback = load_episodes(args)
    fitted = trained_inputs(raw, work, args.earlier, *protected)
    games, _ = home_dates(work)
    spectator = earlier_profiles(raw, protected, games)
    ordinary, events = ordinary_dates_and_events(work)
    known_event_dates = events.get_level_values("date")[events.get_level_values("station_id").isin(STATIONS)]
    ordinary = ordinary.difference(known_event_dates.union(BLOCKED))
    expected = json.loads((work / "sincon_asiad_concert_spillover_results.json").read_text())
    cases, historical, projection = {}, {}, {}
    for name in ("summer", "2025-08-31", "2025-09-30"):
        origin, forecast, score = replay_case(
            name, episodes[name], raw, inputs, protected, calibrated, fallback, fitted, games, spectator,
            expected[name]["all"]["rmsle"]["three_quarters_excess"],
        )
        cases[name] = {"origin": str(origin.date()), "baseline_rmsle": score, "station_days": {}}
        for station in STATIONS:
            station_days = forecast.loc[forecast.station_id.eq(station)]
            for source, _ in DATES:
                placebo_dates = controls(station_days.date, ordinary, source)
                key = f"{station}:{source.date()}"
                cases[name]["station_days"][key] = {
                    "event": comparison(station_days, [source]),
                    "placebo_dates": [str(date.date()) for date in placebo_dates],
                    "placebos": comparison(station_days, placebo_dates),
                    "event_forecast": profile(station_days, source, "forecast").tolist(),
                }
    grid = pd.read_csv(args.team / "data/test.csv", usecols=[*KEY, "example_id"], parse_dates=["date"])
    final = pd.read_csv(work / args.incumbent)
    if not grid.example_id.equals(final.example_id):
        raise ValueError("Incumbent submission IDs must match the official unlabeled grid.")
    future = grid[KEY].assign(forecast=np.log1p(final.boardings))
    for station in STATIONS:
        known = raw.loc[raw.station_id.eq(station)]
        if known.date.max() > pd.Timestamp("2026-06-30"):
            raise ValueError("Competition-period ridership cannot be used.")
        unlabelled = future.loc[future.station_id.eq(station)]
        for source, target in DATES:
            past_controls = controls(known.date, ordinary, source)
            observed = profile(known, source, "observed")
            ordinary_counts = np.median([profile(known, day, "observed") for day in past_controls], axis=0)
            excess = np.maximum(observed - ordinary_counts, 0.)
            historical[f"{station}:{source.date()}"] = {
                "control_dates": [str(day.date()) for day in past_controls],
                "observed": observed.tolist(), "ordinary_median": ordinary_counts.tolist(),
                "positive_excess": excess.tolist(),
            }
            future_controls = controls(unlabelled.date, ordinary, target)
            reference = np.median([profile(unlabelled, day, "forecast") for day in future_controls], axis=0)
            base = profile(unlabelled, target, "forecast")
            apply = np.isin(HOURS, DEPARTURE)
            adjusted = np.where(apply, np.maximum(base, reference + .5 * excess), base)
            projection[f"{station}:{target.date()}"] = {
                "source": str(source.date()), "unlabelled_control_dates": [str(day.date()) for day in future_controls],
                "base": base.tolist(), "unlabelled_ordinary_median": reference.tolist(), "half_excess_floor": adjusted.tolist(),
            }
    return {"hours": HOURS, "departure_hours": DEPARTURE, "historical": historical,
            "backtest_diagnostic_only": cases, "unvalidated_projection_only": projection, "sources": SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("team", "earlier", "recent", "extra_xgb", "june_xgb", "april_xgb"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--incumbent", default="submission_station_profiles_event_timing.csv")
    args = parser.parse_args()
    result = run(args)
    report = args.work_dir / "sincon_asiad_science_festival_results.json"
    report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, case in result["backtest_diagnostic_only"].items():
        print(name, json.dumps(case["station_days"]))
    print("projection", json.dumps(result["unvalidated_projection_only"]))


if __name__ == "__main__":
    main()
