"""Compare LightGBM learning-rate, L2, and stopping changes on fixed forecasts."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import sincon_forecast as forecast

NEW_STATIONS = [3107, 3108, 3109]
HOLIDAY_FLAGS = ["is_substitute", "is_election", "is_lunar_holiday"]
EXTRA_FEATURES = HOLIDAY_FLAGS + [
    "is_lunar_season", "offday_block_len", "offday_block_pos",
    "is_bridge_day", "is_pre_long_offday", "is_post_long_offday",
]
FEATURES = forecast.FEATURES + EXTRA_FEATURES


def holiday_snapshot(calendar, cutoff):
    holiday = pd.Series(calendar.index.isin(forecast.holidays_at(cutoff)), index=calendar.index)
    snapshot = calendar[HOLIDAY_FLAGS].mul(holiday, axis=0)
    offday = holiday | (calendar.index.dayofweek >= 5)
    block = offday.ne(offday.shift(fill_value=False)).cumsum()
    block_len = offday.groupby(block).transform("sum")
    long_block = block_len.ge(3)
    return snapshot.assign(
        is_lunar_season=offday & snapshot["is_lunar_holiday"].groupby(block).transform("max").eq(1),
        offday_block_len=block_len,
        offday_block_pos=offday.groupby(block).cumcount().add(1).where(offday, 0),
        is_bridge_day=~offday & offday.shift(1, fill_value=False) & offday.shift(-1, fill_value=False),
        is_pre_long_offday=~offday & long_block.shift(-1, fill_value=False),
        is_post_long_offday=~offday & long_block.shift(1, fill_value=False),
    )


def attach_calendar(frame, snapshot, cutoff):
    return forecast.add_calendar(frame, cutoff).join(snapshot, on="date", validate="many_to_one")


def select_rounds(history, calendar, cutoff, threads):
    inner_cutoff = cutoff - pd.Timedelta(days=28)
    snapshot = holiday_snapshot(calendar, inner_cutoff)
    inner_train = history.loc[history["date"].le(inner_cutoff)]
    inner_valid = history.loc[
        history["date"].gt(inner_cutoff) & ~history["station_id"].isin(NEW_STATIONS)
    ]
    model = forecast.fit_lightgbm(
        attach_calendar(inner_train, snapshot, inner_cutoff), threads,
        features=FEATURES, rounds=900,
        stopping_validation=attach_calendar(inner_valid, snapshot, inner_cutoff),
    )
    return model.best_iteration


def run(data_zip, calendar_path, threads):
    train, _ = forecast.read_inputs(data_zip)
    calendar = pd.read_csv(
        calendar_path, usecols=["date", *HOLIDAY_FLAGS], parse_dates=["date"],
    ).set_index("date").sort_index()
    expected_dates = pd.date_range(train["date"].min(), forecast.CUTOFF + pd.Timedelta(days=123))
    if not calendar.index.equals(expected_dates) or calendar.isna().any().any():
        raise ValueError("Calendar must have a complete, unique date grid and holiday flags.")

    experiments = {
        "calendar_baseline": {},
        "learning_rate_0.025": {"learning_rate": 0.025},
        "l2_10": {"lambda_l2": 10.0},
        "early_stopping": {},
    }
    scores = {experiment: [] for experiment in experiments}
    counts = []
    for fold, cutoff in forecast.BACKTESTS.items():
        history, validation = forecast.split_backtest(train, cutoff)
        snapshot = holiday_snapshot(calendar, cutoff)
        training = attach_calendar(history, snapshot, cutoff)
        future = attach_calendar(validation.drop(columns="boardings"), snapshot, cutoff)
        new_stations = future["station_id"].isin(NEW_STATIONS).to_numpy()
        seasonal = forecast.seasonal_predict(training, future)
        counts.append(len(validation))

        for experiment, options in experiments.items():
            started = perf_counter()
            rounds = forecast.LIGHTGBM_ROUNDS
            if experiment == "early_stopping":
                rounds = select_rounds(history, calendar, cutoff, threads)
            predictions = forecast.lightgbm_predict(
                training, future, threads, features=FEATURES, rounds=rounds, **options,
            )
            hybrid = np.where(new_stations, seasonal, predictions)
            score = forecast.rmsle(validation["boardings"], hybrid)
            scores[experiment].append(score)
            print(json.dumps({
                "fold": fold,
                "cutoff": str(cutoff.date()),
                "validation_rows": len(validation),
                "experiment": experiment,
                "rounds": rounds,
                "rmsle": score,
                "seconds": round(perf_counter() - started, 2),
            }, allow_nan=False), flush=True)

    pooled = {
        experiment: float(np.sqrt(np.average(np.square(values), weights=counts)))
        for experiment, values in scores.items()
    }
    print(json.dumps({"pooled_rmsle": pooled, "best_experiment": min(pooled, key=pooled.get)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_zip", type=Path)
    parser.add_argument("calendar_csv", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    run(args.data_zip, args.calendar_csv, args.threads)
