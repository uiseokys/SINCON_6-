"""Replay a concert departure correction at Seo-gu Office station."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_munhak_ordinary import WARM, home_dates
from sincon_munhak_spectator import HISTORY, earlier_profiles
from sincon_overlay_candidate import replay, trained_inputs
from sincon_station_profile_candidate import add_spectator_events, station_adjustments
from sincon_station_state import KEY
from sincon_xgb_conditioned import load_episodes


STATION = 3210
HOURS = [21, 22, 23]
EVENTS = {
    "2025-05-03": "trend_wave",
    "2025-05-31": "sbs_mega",
    "2025-06-28": "psy_2025",
    "2025-06-29": "psy_2025",
    "2025-09-13": "seventeen_2025",
    "2025-09-14": "seventeen_2025",
    "2025-10-18": "stray_kids",
    "2025-10-19": "stray_kids",
    "2026-04-04": "seventeen_encore",
    "2026-04-05": "seventeen_encore",
    "2026-06-20": "carat_land",
    "2026-06-21": "carat_land",
    "2026-07-11": "psy_2026",
}
EVENTS = {pd.Timestamp(date): production for date, production in EVENTS.items()}
WEIGHTS = {"half_excess": .5, "three_quarters_excess": .75}
BASE_FILE = "submission_station_profiles_spectator.csv"
VENUE = "https://www.insiseol.or.kr/sport/asiad/introduce/location.jsp"
CALENDAR = "https://www.insiseol.or.kr/sport_asiad/scheduleCalendar.do?schDiv=main&siteDiv=sport_asiad&year_month="
EXCLUDE = [
    ("2025-05-26", "2025-05-30"),
    ("2025-06-27", "2025-06-27"),
    ("2025-09-05", "2025-09-22"),
    ("2025-10-09", "2025-10-27"),
    ("2026-07-08", "2026-07-14"),
]


def ordinary_profile(history, date, ordinary, origin):
    earlier = ordinary[(ordinary <= origin) & (ordinary < date) & (ordinary.day_of_week == date.day_of_week)]
    if len(earlier) < 6:
        raise ValueError(f"Insufficient prior ordinary dates for {date.date()}.")
    return history.loc[earlier[-6:], HOURS].median()


def analog(history, ordinary, date, origin, production):
    if history.index.max() > origin or date <= origin:
        raise ValueError("Only observations at or before the forecast origin may be used.")
    samples = []
    for day, name in EVENTS.items():
        if day > origin or name == production:
            continue
        level = ordinary_profile(history, day, ordinary, origin)
        samples.append((day, name, (history.loc[day, HOURS] - level).clip(lower=0)))
    if production.startswith("psy") and any(name.startswith("psy") for _, name, _ in samples):
        samples = [sample for sample in samples if sample[1].startswith("psy")]
    if not samples:
        return None, {}
    by_production = pd.DataFrame([counts for _, _, counts in samples])
    by_production.index = [name for _, name, _ in samples]
    excess = by_production.groupby(level=0).mean().median()
    level = ordinary_profile(history, date, ordinary, origin)
    return level, {
        "sources": [str(day.date()) for day, _, _ in samples],
        "prior_ordinary": level.to_list(),
        "prior_excess": excess.to_list(),
    }


def proposals(future, history, origin, protected, include_placebos=True):
    if {"y_log", "boardings", "observed"} & set(future):
        raise ValueError("Concert forecasts must not receive future observations.")
    blocked = pd.DatetimeIndex(list(EVENTS)).union(protected[0])
    event_dates = protected[1].get_level_values("date")
    event_stations = protected[1].get_level_values("station_id")
    blocked = blocked.union(event_dates[event_stations.isin([STATION, 3209])])
    for start, stop in EXCLUDE:
        blocked = blocked.union(pd.date_range(start, stop))
    history = history.loc[history.date.le(origin) & history.station_id.eq(STATION)]
    history = np.expm1(history.pivot(index="date", columns="hour", values="observed"))
    ordinary = history.index.difference(blocked)
    forecasts = future[KEY].assign(incumbent=future.incumbent.clip(lower=0))
    forecasts = forecasts.assign(**{name: forecasts.incumbent for name in WEIGHTS})
    forecasts = forecasts.assign(event=False, placebo=False, placebo_forecast=forecasts.incumbent)
    details = {}
    target_dates = pd.DatetimeIndex(forecasts.date.unique()).sort_values()
    for date, production in EVENTS.items():
        if date not in target_dates:
            continue
        level, support = analog(history, ordinary, date, origin, production)
        if level is None:
            details[str(date.date())] = {"production": production, "sources": []}
            continue
        rows = forecasts.date.eq(date) & forecasts.station_id.eq(STATION) & forecasts.hour.isin(HOURS)
        positions = forecasts.loc[rows].sort_values("hour").index
        if forecasts.loc[positions, "hour"].tolist() != HOURS:
            raise ValueError("Each concert needs one row for each departure hour.")
        excess = np.asarray(support["prior_excess"])
        base = forecasts.loc[positions, "incumbent"].to_numpy()
        forecasts.loc[positions, "event"] = True
        for method, weight in WEIGHTS.items():
            floors = np.log1p(level.to_numpy() + weight * excess)
            forecasts.loc[positions, method] = np.maximum(base, floors)
        details[str(date.date())] = {
            "production": production,
            **support,
            "base": np.expm1(base).tolist(),
            **{method: np.expm1(forecasts.loc[positions, method]).tolist() for method in WEIGHTS},
        }
        if include_placebos:
            controls = target_dates[(target_dates < date) & (target_dates.day_of_week == date.day_of_week)]
            controls = controls.difference(blocked)[-2:]
            for control in controls:
                mask = forecasts.date.eq(control) & forecasts.station_id.eq(STATION) & forecasts.hour.isin(HOURS)
                place = forecasts.loc[mask].sort_values("hour").index
                control_level = ordinary_profile(history, control, ordinary, origin)
                floors = np.log1p(control_level.to_numpy() + WEIGHTS["three_quarters_excess"] * excess)
                forecasts.loc[place, "placebo_forecast"] = np.maximum(forecasts.loc[place, "placebo_forecast"], floors)
                forecasts.loc[place, "placebo"] = True
    for method in WEIGHTS:
        changed = forecasts[method].ne(forecasts.incumbent)
        if (changed & ~forecasts.event).any() or (forecasts[method] < forecasts.incumbent).any():
            raise ValueError("The proposed correction may only raise verified event departure forecasts.")
    return forecasts, details


def metrics(rows):
    output = {}
    subsets = {
        "all": (np.ones(len(rows), dtype=bool), list(WEIGHTS)),
        "concert_3210_21_23": (rows.event, list(WEIGHTS)),
        "ordinary_placebo_3210_21_23": (rows.placebo, ["placebo_forecast"]),
    }
    for name, (scope, methods) in subsets.items():
        part = rows.loc[scope]
        if part.empty:
            continue
        errors = part[["incumbent", *methods]].rsub(part.y_log, axis=0).pow(2)
        output[name] = {
            "rows": len(part),
            "dates": part.date.nunique(),
            "rmsle": np.sqrt(errors.mean()).to_dict(),
            "sse_delta": {method: float((errors[method] - errors.incumbent).sum()) for method in methods},
        }
    return output


def run(args):
    work = args.work_dir
    episodes, raw, inputs, protected, calibrated, fallback = load_episodes(args)
    fitted = trained_inputs(raw, work, args.earlier, *protected)
    games, _ = home_dates(work)
    spectator = earlier_profiles(raw, protected, games)
    truth = raw.set_index(KEY).observed
    official = json.loads((work / "sincon_station_profile_spectator_results.json").read_text())
    cases, report = {}, {}
    for label, (rows, origin, _) in episodes.items():
        rows = rows.reset_index(drop=True)
        key = pd.MultiIndex.from_frame(rows[KEY])
        if not key.is_unique or rows.date.min() <= origin or not np.allclose(truth.reindex(key), rows.y_log, atol=1e-8, rtol=0):
            raise ValueError(f"Validation keys, labels or dates are invalid in {label}.")
        base, _ = replay(rows, origin, raw, inputs, protected, calibrated, fallback, fitted)
        future = base[[*KEY, "combined"]]
        history = raw.loc[raw.date.le(origin)]
        layer, _ = station_adjustments(history, future, origin, protected, games)
        layer, _ = add_spectator_events(layer, future, HISTORY, origin, spectator)
        future = base[KEY].assign(incumbent=base.combined + layer["joint_with_spectator"])
        prediction, details = proposals(future, history, origin, protected)
        cases[label] = prediction.assign(y_log=base.y_log)
        report[label] = {"origin": str(origin.date()), "events": details, **metrics(cases[label])}
        reproduced = report[label]["all"]["rmsle"]["incumbent"]
        if abs(reproduced - official[label]["joint_with_spectator"]["candidate"]) > 1e-10:
            raise ValueError(f"Spectator-augmented baseline does not reproduce in {label}.")
    cases["disjoint_A_C"] = pd.concat([cases[label] for label in ("A", "C")], ignore_index=True)
    cases["disjoint_warm"] = pd.concat([cases[label].loc[cases[label].date.dt.month.eq(month)] for label, month in WARM],
                                      ignore_index=True)
    for label in ("disjoint_A_C", "disjoint_warm"):
        if cases[label].duplicated(KEY).any():
            raise ValueError(f"Validation keys must be disjoint in {label}.")
        report[label] = metrics(cases[label])
    origin = pd.Timestamp("2026-06-30")
    grid = pd.read_csv(args.team / "data/test.csv", usecols=[*KEY, "example_id"], parse_dates=["date"])
    final = pd.read_csv(work / BASE_FILE, float_precision="round_trip")
    if not grid.example_id.equals(final.example_id):
        raise ValueError("The test grid and spectator submission must have the same IDs.")
    future = grid[KEY].assign(incumbent=np.log1p(final.boardings))
    _, projection = proposals(future, raw, origin, protected, include_placebos=False)
    report["final_projection_only"] = projection
    report["sources"] = {"directions": VENUE, "venue_calendar": [CALENDAR + date for date in
        ("2025-05", "2025-06", "2025-09", "2025-10", "2026-04", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10")],
        "trend_wave_city_announcement": "https://www.incheon.go.kr/IC010205/view?repSeq=DOM_0000000012337767",
        "seogu_day_program_excluded": "https://www.youtube.com/watch?v=nOcRANQcGTU",
        "psy_2026_ticket": "https://ticket.interpark.com/OtherSite/Ticket/Ticket_GoodsInfo.asp?Ca=01003&GroupCode=26007871"}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("team", "earlier", "recent", "extra_xgb", "june_xgb", "april_xgb"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = run(args)
    (args.work_dir / "sincon_asiad_concert_spillover_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for label in ("A", "B", "C", "summer", "disjoint_A_C", "disjoint_warm", "final_projection_only"):
        print(label, json.dumps({key: value for key, value in result[label].items() if key != "events"}))


if __name__ == "__main__":
    main()
