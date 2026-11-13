"""Compare hourly and daily-profile forecasts using cutoff-safe SINCON backtests."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import sincon_forecast as forecast
from sincon_lightgbm_experiments import (
    FEATURES, HOLIDAY_FLAGS, NEW_STATIONS, attach_calendar, holiday_snapshot,
)

DAY_KEY = ["date", "station_id"]
MIN_HISTORY_DAYS = 28
FOLDS = {**forecast.BACKTESTS, "autumn_origin": pd.Timestamp("2025-10-31")}


def decompose_day(frame):
    result = frame.assign(log_boardings=np.log1p(frame["boardings"].astype("float64")))
    level = result.groupby(DAY_KEY)["log_boardings"].transform("mean")
    return result.assign(daily_level=level, hourly_shape=result["log_boardings"] - level)


def lookup(series, rows, keys):
    return series.reindex(pd.MultiIndex.from_frame(rows[keys])).to_numpy()


def catboost_log_predict(history, future, features, target, depth, threads):
    from catboost import CatBoostRegressor

    history = history.sort_values("date", kind="stable")
    model = CatBoostRegressor(
        iterations=350, learning_rate=0.07, depth=depth, loss_function="RMSE",
        random_seed=42, thread_count=threads, task_type="CPU", has_time=True,
        one_hot_max_size=12, max_ctr_complexity=2, allow_writing_files=False, verbose=False,
        cat_features=[name for name in forecast.CATEGORICAL if name in features],
    )
    model.fit(history[features], history[target])
    return model.predict(future[features], thread_count=threads)


def estimate_hourly_shape(history, future):
    station_hour = ["station_id", "hour"]
    weekday = station_hour + ["weekday"]
    holiday_kind = station_hour + ["holiday_type"]
    ordinary = history[history["holiday_type"].eq(0)]
    holiday = history[history["holiday_type"].gt(0)]

    profile = lookup(ordinary.groupby(weekday)["hourly_shape"].mean(), future, weekday)
    fallback = lookup(ordinary.groupby(station_hour)["hourly_shape"].mean(), future, station_hour)
    profile = np.where(np.isfinite(profile), profile, fallback)
    overall_holiday = lookup(
        holiday.groupby(station_hour)["hourly_shape"].mean(), future, station_hour,
    )
    overall_holiday = np.where(np.isfinite(overall_holiday), overall_holiday, profile)
    by_type = holiday.groupby(holiday_kind)["hourly_shape"].agg(["sum", "count"])
    total = np.nan_to_num(lookup(by_type["sum"], future, holiday_kind))
    count = np.nan_to_num(lookup(by_type["count"], future, holiday_kind))
    holiday_profile = (total + 4 * overall_holiday) / (count + 4)
    return np.where(future["holiday_type"].gt(0), holiday_profile, profile)


def daily_catboost_log_predict(history, future, threads):
    day_features = [name for name in FEATURES if name != "hour"]
    daily_train = history.drop_duplicates(DAY_KEY)
    daily_future = future.drop_duplicates(DAY_KEY)
    predictions = catboost_log_predict(
        daily_train, daily_future, day_features, "daily_level", depth=6, threads=threads,
    )
    daily_index = pd.MultiIndex.from_frame(daily_future[DAY_KEY])
    daily_level = lookup(pd.Series(predictions, index=daily_index), future, DAY_KEY)
    return daily_level + estimate_hourly_shape(history, future)


def cold_station_log_profile(history, future, cold_stations):
    warm = history.loc[~history["station_id"].isin(cold_stations)].copy()
    station_hour = ["station_id", "hour"]
    peer_keys = ["canonical_line", "hour", "day_group"]
    station_mean = warm.groupby(station_hour)["log_boardings"].transform("mean")
    peer_effect = (warm["log_boardings"] - station_mean).groupby(
        [warm[key] for key in peer_keys],
    ).mean()
    cold = history.loc[history["station_id"].isin(cold_stations)].copy()
    first = cold.groupby("station_id")["date"].transform("min")
    last = cold.groupby("station_id")["date"].transform("max")
    cold = cold.loc[cold["date"].gt(first) | first.eq(last)]
    cold["station_level"] = cold["log_boardings"] - np.nan_to_num(lookup(peer_effect, cold, peer_keys))
    level = lookup(cold.groupby(station_hour)["station_level"].mean(), future, station_hour)
    return level + np.nan_to_num(lookup(peer_effect, future, peer_keys))


def retrospective_variation(train):
    ordinary = train.loc[
        ~train["station_id"].isin(NEW_STATIONS) & train["holiday_type"].eq(0)
    ]
    clean = decompose_day(ordinary).assign(year_month=ordinary["date"].dt.to_period("M"))
    columns = ["log_boardings", "daily_level", "hourly_shape"]
    groups = ["station_id", "hour", "year_month", "weekday"]
    residuals = clean[columns] - clean.groupby(groups)[columns].transform("mean")
    return np.sqrt(residuals.pow(2).mean()).to_dict()


def run(data_zip, calendar_csv, threads):
    train, _ = forecast.read_inputs(data_zip)
    calendar = pd.read_csv(
        calendar_csv, usecols=["date", *HOLIDAY_FLAGS], parse_dates=["date"],
    ).set_index("date").sort_index()
    expected_dates = pd.date_range(train["date"].min(), forecast.CUTOFF + pd.Timedelta(days=forecast.HORIZON))
    if not calendar.index.equals(expected_dates) or calendar.isna().any().any():
        raise ValueError("Calendar must cover the complete date grid with no missing holiday flags.")

    results = {}
    for fold, cutoff in FOLDS.items():
        started = perf_counter()
        history, validation = forecast.split_backtest(train, cutoff)
        assert history["date"].max() == cutoff
        assert len(validation) == 157_194
        snapshot = holiday_snapshot(calendar, cutoff)
        history = decompose_day(attach_calendar(history, snapshot, cutoff))
        future = attach_calendar(validation.drop(columns="boardings"), snapshot, cutoff)
        history_days = history.groupby("station_id")["date"].nunique()
        cold_stations = history_days.index[history_days.lt(MIN_HISTORY_DAYS)]
        cold = future["station_id"].isin(cold_stations).to_numpy()
        fixed_new = future["station_id"].isin(NEW_STATIONS).to_numpy()
        actual = validation["boardings"].to_numpy()

        seasonal_log = np.log1p(forecast.seasonal_predict(history, future))
        lgbm_log = np.log1p(forecast.lightgbm_predict(history, future, threads, features=FEATURES))
        hourly = catboost_log_predict(history, future, FEATURES, "log_boardings", depth=8, threads=threads)
        daily = daily_catboost_log_predict(history, future, threads)
        log_blend = 0.5 * lgbm_log + 0.25 * hourly + 0.25 * daily
        cold_peer = seasonal_log.copy()
        if cold.any():
            cold_peer[cold] = cold_station_log_profile(history, future.loc[cold], cold_stations)

        models = {
            "seasonal_average": seasonal_log,
            "existing_hybrid": np.where(fixed_new, seasonal_log, lgbm_log),
            "lightgbm_with_short_history_fallback": np.where(cold, seasonal_log, lgbm_log),
            "hourly_catboost": np.where(cold, seasonal_log, hourly),
            "daily_catboost_and_hourly_profile": np.where(cold, seasonal_log, daily),
            "catboost_daily_hourly_blend": np.where(cold, seasonal_log, 0.5 * (hourly + daily)),
            "lightgbm_catboost_blend": np.where(cold, seasonal_log, log_blend),
            "lightgbm_catboost_blend_peer_profile": np.where(cold, cold_peer, log_blend),
        }
        if not all(np.isfinite(value).all() for value in models.values()):
            raise ValueError(f"Nonfinite forecasts in {fold}.")
        predictions = {name: np.expm1(np.maximum(log_values, 0)) for name, log_values in models.items()}
        scores = {name: forecast.rmsle(actual, value) for name, value in predictions.items()}
        scores_warm = {
            name: forecast.rmsle(actual[~cold], value[~cold]) for name, value in predictions.items()
        }
        scores_cold = {
            name: forecast.rmsle(actual[cold], value[cold]) for name, value in predictions.items()
        } if cold.any() else {}
        results[fold] = {
            "cutoff": str(cutoff.date()), "rows": len(validation), "cold_stations": cold_stations.tolist(),
            "rmsle": scores, "rmsle_warm": scores_warm, "rmsle_cold": scores_cold,
            "seconds": round(perf_counter() - started, 2),
        }
        print(json.dumps({fold: results[fold]}, allow_nan=False), flush=True)

    common_folds = [results[name] for name in forecast.BACKTESTS]
    pooled = {
        name: float(np.sqrt(np.average(
            [fold["rmsle"][name] ** 2 for fold in common_folds],
            weights=[fold["rows"] for fold in common_folds],
        ))) for name in models
    }
    return {
        "folds": results,
        "pooled_existing_folds": pooled,
        "historical_within_group_variation": retrospective_variation(train),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_zip", type=Path)
    parser.add_argument("calendar_csv", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    results = run(args.data_zip, args.calendar_csv, args.threads)
    if args.results:
        args.results.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"pooled_existing_folds": results["pooled_existing_folds"]}, allow_nan=False), flush=True)
