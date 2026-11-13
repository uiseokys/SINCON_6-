"""Backtest a count-scale departure forecast from earlier stadium concert days."""

import argparse
import html
import json
import re
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sincon_station_state import KEY, WINDOWS


STADIUMS = {3127: "FC002149", 3209: "FC002083"}
HOURS = np.arange(6, 24)
DEPARTURE = HOURS >= 19
WEEKDAYS = "월화수목금토일"
CLOCK = re.compile(r"(\d{1,2}):([0-5]\d)")


def start_hour(text):
    match = CLOCK.search(str(text))
    return int(match[1]) + int(match[2]) / 60 if match else np.nan


def runtime_hours(text):
    hours, minutes = (re.search(rf"(\d+)\s*{unit}", str(text)) for unit in ("시간", "분"))
    duration = (int(hours[1]) if hours else 0) + (int(minutes[1]) / 60 if minutes else 0)
    return duration if 0 < duration <= 6 else np.nan


def scheduled_start(schedule, weekday):
    text = html.unescape(str(schedule))
    for fragment in re.split(r"<br\s*/?>", text):
        if WEEKDAYS[weekday] in re.findall(r"([월화수목금토일])요일", fragment):
            return start_hour(fragment)
    starts = {match[0] for match in CLOCK.finditer(text)}
    return start_hour(text) if len(starts) == 1 else np.nan


def read_inputs(work_dir, team_dir):
    with ZipFile(work_dir / "sincon_local_repro.zip") as archive:
        events = pd.read_csv(archive.open("external_inputs/external/events_daily.csv"), parse_dates=["date"])
        calendar = pd.read_csv(
            archive.open("external_inputs/calendar_features.csv"), parse_dates=["date"],
            usecols=["date", "is_holiday"],
        ).set_index("date").is_holiday
    shows = pd.read_csv(work_dir / "sincon_kopis_scheduled.csv", parse_dates=["start_date", "end_date"])
    shows = shows.loc[shows.venue_id.isin(STADIUMS.values()) & shows.status_detail.eq("공연완료")]
    dates = pd.read_csv(work_dir / "sincon_kbo_game_data.csv", usecols=["date"], parse_dates=["date"]).date
    train = pd.read_csv(team_dir / "data/train.csv", usecols=[*KEY, "boardings"], parse_dates=["date"])
    train = train.loc[train.station_id.isin(STADIUMS)].pivot(index=["station_id", "date"], columns="hour", values="boardings")
    concerts = events.loc[events.category.eq("concert") & events.station_id.isin(STADIUMS)]
    records = []
    for event in concerts.itertuples(index=False):
        match = shows.loc[
            shows.venue_id.eq(STADIUMS[event.station_id]) & shows.start_date.le(event.date) & shows.end_date.ge(event.date)
        ]
        if len(match) != 1:
            continue
        show = next(match.itertuples(index=False))
        start = scheduled_start(show.schedule, event.date.dayofweek)
        start = start if np.isfinite(start) else start_hour(event.start_time_kst)
        if not np.isfinite(start) or not np.isfinite(show.seats) or show.seats <= 0:
            continue
        records.append({
            "date": event.date, "station_id": event.station_id, "performance": show.performance_id,
            "title": event.event_name, "start": start, "duration": runtime_hours(show.runtime), "seats": show.seats,
            "source_url": show.source_url,
        })
    matched = pd.DataFrame(records)
    if matched.duplicated(["station_id", "date"]).any():
        raise ValueError("Multiple concerts at the same stadium on one day.")
    return train, events, matched, calendar, dates


def usual_days(station, history, events, baseball):
    dates = events.loc[events.station_id.eq(station), "date"]
    if station == 3127:
        dates = pd.concat([dates, baseball], ignore_index=True)
    return history.loc[~history.index.isin(dates)].reindex(columns=HOURS).dropna()


def ordinary_profile(ordinary, date, calendar):
    day_type = np.where(calendar.reindex(ordinary.index).eq(1), 7, ordinary.index.dayofweek)
    target_type = 7 if calendar.loc[date] == 1 else date.dayofweek
    distance = np.abs((ordinary.index - date).days)
    selected = np.lexsort((distance, day_type != target_type))[:12]
    if len(selected) < 8:
        raise ValueError("Insufficient ordinary days for a station profile.")
    return ordinary.iloc[selected].median(axis=0).to_numpy()


def analog_forecast(target, ordinary, history, source, calendar, timing):
    series = {}
    for old in source.itertuples(index=False):
        reference = ordinary_profile(ordinary, old.date, calendar)
        counts = history.loc[old.date, HOURS].to_numpy(dtype=float)
        excess = np.maximum(counts - reference, 0)
        # Discard arrival traffic and focus the analog on the departure wave.
        excess[HOURS + 1 <= old.start + 1] = 0
        delta = target.start - old.start
        if timing and np.isfinite(target.duration) and np.isfinite(old.duration):
            delta += target.duration - old.duration
        profile = np.interp(HOURS - delta, HOURS, excess, left=0, right=0)
        if timing and (not np.isfinite(target.duration) or not np.isfinite(old.duration)):
            profile = np.convolve(np.pad(profile, 1), [0.15, 0.7, 0.15], mode="valid")
        scale = np.clip(np.sqrt(target.seats / old.seats), 0.5, 2)
        weight = 1 / (1 + abs(delta))
        peak = scale * np.max(excess[DEPARTURE])
        series.setdefault(old.performance, []).append((weight, scale * profile, peak))
    # One analog vote per KOPIS performance, including multi-night runs.
    analogs, weights, peaks = [], [], []
    for profiles in series.values():
        weight, counts, peak = zip(*profiles)
        analogs.append(np.average(counts, axis=0, weights=weight))
        weights.append(np.mean(weight))
        peaks.append(np.median(peak))
    return np.average(analogs, axis=0, weights=weights), np.median(peaks)


def rmsle(truth, estimate):
    return float(np.sqrt(np.square(truth - estimate).mean())) if len(truth) else None


def evaluate(archive, train, events, matches, calendar, baseball, cutoff):
    prediction = archive.copy()
    methods = ("start_aligned", "finish_aligned", "incumbent_peak_height")
    for method in methods:
        prediction[method] = prediction.local_stack
    target_dates = set(prediction.date)
    all_targets = events.loc[events.category.eq("concert") & events.station_id.isin(STADIUMS) & events.date.isin(target_dates)]
    keys = pd.MultiIndex.from_frame(all_targets[["date", "station_id"]])
    event = pd.MultiIndex.from_frame(prediction[["date", "station_id"]]).isin(keys)
    event_late = event & prediction.hour.ge(19)
    changed = np.zeros(len(prediction), dtype=bool)
    coverage = []
    for station in STADIUMS:
        history = train.loc[station].loc[:cutoff]
        if history.index.max() > cutoff:
            raise ValueError("Analog source crosses forecast origin.")
        ordinary = usual_days(station, history, events, baseball)
        station_matches = matches.loc[matches.station_id.eq(station)]
        past = station_matches.loc[station_matches.date.le(cutoff) & station_matches.date.isin(history.index)]
        future = station_matches.loc[station_matches.date.gt(cutoff) & station_matches.date.isin(target_dates)]
        for target in future.itertuples(index=False):
            # Excluding the entire performance also tests transfers to unfamiliar artists.
            sources = past.loc[past.performance.ne(target.performance)]
            rows = prediction.station_id.eq(station) & prediction.date.eq(target.date) & prediction.hour.ge(19)
            if sources.performance.nunique() < 1 or not rows.any():
                continue
            positions = np.searchsorted(HOURS, prediction.loc[rows, "hour"])
            background = ordinary_profile(ordinary, target.date, calendar)[positions]
            previous = prediction.loc[rows, "local_stack"].to_numpy()
            for timing, method in ((False, "start_aligned"), (True, "finish_aligned")):
                excess, peak_height = analog_forecast(target, ordinary, history, sources, calendar, timing)
                estimate = np.log1p(background + excess[positions])
                prediction.loc[rows, method] = (previous + estimate) / 2
            # Exploratory follow-up. Trust the incumbent's clock time and use
            # the analog only to raise the busiest predicted departure hour.
            peak = np.argmax(np.expm1(previous) - background)
            lifted = max(previous[peak], np.log1p(background[peak] + peak_height))
            prediction.loc[prediction.index[rows][peak], "incumbent_peak_height"] = (previous[peak] + lifted) / 2
            changed |= rows.to_numpy()
            coverage.append({
                "date": str(target.date.date()), "station_id": station, "title": target.title,
                "source_days": [str(day.date()) for day in sources.date], "source_url": target.source_url,
                "scheduled_runtime_hours": float(target.duration) if np.isfinite(target.duration) else None,
                "incumbent_peak_hour": int(prediction.loc[rows, "hour"].iloc[peak]),
            })
    if not prediction.loc[~changed, list(methods)].eq(prediction.loc[~changed, "local_stack"], axis=0).all().all():
        raise ValueError("Concert forecast changed an unrelated row.")
    metrics = {}
    for name in ("local_stack", *methods):
        metrics[name] = {
            "overall": rmsle(prediction.y_log, prediction[name]),
            "concert_hours_19_23": rmsle(prediction.y_log[event_late], prediction.loc[event_late, name]),
            "adjusted_hours": rmsle(prediction.y_log[changed], prediction.loc[changed, name]),
        }
    return {
        "scores": metrics,
        "event_rows_19_23": int(event_late.sum()),
        "adjusted_rows": int(changed.sum()),
        "peak_height_adjusted_rows": int(prediction.incumbent_peak_height.ne(prediction.local_stack).sum()),
        "oracle_overall_perfect_concert_hours": rmsle(prediction.y_log, prediction.local_stack.mask(event_late, prediction.y_log)),
        "coverage": coverage,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team_dir", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    train, events, matches, calendar, baseball = read_inputs(args.work_dir, args.team_dir)
    backtests = pd.read_csv(args.work_dir / "sincon_local_validation.csv.gz", parse_dates=["date"])
    summer = pd.read_csv(args.work_dir / "sincon_summer_replay_validation.csv.gz", parse_dates=["date"])
    results = {}
    for fold, (origin, end) in {**WINDOWS, "summer_2025": ("2025-06-30", "2025-10-31")}.items():
        start = perf_counter()
        reference = summer if fold == "summer_2025" else backtests.loc[backtests.fold.eq(fold)]
        if reference.duplicated(KEY).any() or not reference.date.gt(origin).all() or not reference.date.le(end).all():
            raise ValueError(f"Unexpected evaluation rows in {fold}.")
        results[fold] = evaluate(reference, train, events, matches, calendar, baseball, pd.Timestamp(origin))
        results[fold]["seconds"] = round(perf_counter() - start, 3)
        print(json.dumps({fold: {k: v for k, v in results[fold].items() if k != "coverage"}}, allow_nan=False))
    results["test_concert_days"] = int(matches.date.between("2026-07-01", "2026-10-31").sum())
    (args.work_dir / "sincon_stadium_concert_analogs_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
