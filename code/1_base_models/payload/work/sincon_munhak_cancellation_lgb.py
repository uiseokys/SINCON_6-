"""Compute paired LightGBM deltas for four cancelled Munhak music events."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd


STATION = 3127
CANCELLED = {
    "2026-09-26": "OBJET K-POP FESTA 2026",
    "2026-09-27": "OBJET K-POP FESTA 2026",
    "2026-10-24": "GHOSTIVAL [인천]",
    "2026-10-25": "GHOSTIVAL [인천]",
}
EVENT_COLUMNS = ["ev_today", "ev_is_concert", "ev_day_index", "ev_n_days", "ev_hours_from_start", "ev_city"]
KEY = ["date", "station_id", "hour", "example_id"]
CUTOFF = "2026-06-30"


def build_test_copies(target):
    expected_dates = list(CANCELLED)
    main = target.station_id.eq(STATION)
    station = target.copy()
    station.loc[main, ["ev_today", "ev_is_concert"]] = 0
    station.loc[main, ["ev_day_index", "ev_n_days", "ev_hours_from_start"]] = np.nan
    station_and_city = station.assign(ev_city=station.ev_city - 1)
    if (len(target) != 4 * 71 * 18 or target.duplicated(KEY).any() or not target.example_id.is_unique
            or target.date.unique().tolist() != expected_dates or main.sum() != 72
            or not target.loc[main, "ev_today"].eq(1).all()
            or not target.loc[main, "mh_game"].eq(0).all()
            or not target.ev_city.ge(1).all()):
        raise ValueError("Expected one falsely active event per target date and no KBO home game.")
    if not target.drop(columns=EVENT_COLUMNS).equals(station_and_city.drop(columns=EVENT_COLUMNS)):
        raise ValueError("A feature unrelated to the cancelled event changed.")
    return station, station_and_city


def evaluate(args):
    sys.path.insert(0, str(args.team.resolve()))
    from ensemble import FEATURES, LGB_SEEDS
    from improved import PARAMS, enrich, fit_predict
    from make_calendar import build_calendar

    PARAMS["num_threads"] = args.threads
    data = args.team / "data"
    calendar = build_calendar()
    training = pd.read_csv(data / "train.csv", usecols=["date", "station_id", "hour", "boardings"], dtype={"date": str})
    keys = pd.read_csv(data / "test.csv", usecols=KEY, dtype={"date": str})
    if training.date.max() != CUTOFF or len(keys) != 157_194 or "boardings" in keys:
        raise ValueError("Expected official cutoff and unlabeled test keys.")
    meta = pd.read_csv(data / "metaData.csv", usecols=["station_id", "canonical_line"]).rename(columns={"canonical_line": "line"})
    event_file = pd.read_csv(args.team / "external/events_daily.csv")
    listed = event_file.loc[event_file.date.isin(CANCELLED) & event_file.station_id.eq(STATION)]
    if listed.set_index("date").event_name.to_dict() != CANCELLED or len(listed) != 4:
        raise ValueError("The original input no longer matches the cancellation audit.")
    training = enrich(training, calendar, meta, True)
    original = enrich(keys.loc[keys.date.isin(CANCELLED)], calendar, meta, True)
    local, city = build_test_copies(original)
    combined = pd.concat([original, local, city], ignore_index=True)
    started = perf_counter()
    seeds = LGB_SEEDS if args.seeds == 3 else LGB_SEEDS[:1]
    original_logs, local_logs, city_logs = np.split(np.log1p(fit_predict(training, combined, FEATURES, seeds)), 3)
    local_delta = local_logs - original_logs
    city_delta = city_logs - local_logs
    main = original.station_id.eq(STATION).to_numpy()
    if np.any(local_delta[~main]) or not np.isfinite(np.column_stack([local_delta, city_delta])).all():
        raise ValueError("The station event features may only affect the 72 selected station-hours.")
    frame = original[KEY].assign(original_log_lgb=original_logs, local_delta_log=local_delta, city_delta_log=city_delta)
    frame["station_and_city_delta_log"] = frame.local_delta_log + frame.city_delta_log
    archived = pd.read_csv(args.team / "ensemble_parts.csv", usecols=["example_id", "log_lgb"])
    checked = frame.merge(archived, on="example_id", validate="one_to_one")
    if len(checked) != len(frame):
        raise ValueError("Archived original LightGBM forecast keys differ from the test grid.")
    archived_error = np.abs(checked.log_lgb - checked.original_log_lgb)
    return {
        "seeds": seeds,
        "seconds_fit": round(perf_counter() - started, 2),
        "archive_mean_absolute_difference": float(archived_error.mean()),
        "archive_max_absolute_difference": float(archived_error.max()),
        "archive_station_max_absolute_difference": float(archived_error[checked.station_id.eq(STATION)].max()),
        "station_only": frame.loc[main].to_dict(orient="records"),
        "city_delta_other_station_summary": frame.loc[~main].groupby("date").city_delta_log.agg(
            ["size", "mean", "min", "max"]).to_dict(orient="index"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team", type=Path)
    parser.add_argument("--seeds", choices=[1, 3], type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--report", type=Path, default=Path("sincon_munhak_cancellation_lgb_results.json"))
    args = parser.parse_args()
    result = evaluate(args)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print({key: value for key, value in result.items() if key != "station_only"}, flush=True)
