"""Forecast whole station-day profiles with calendar-based kernel ridge regression."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sincon_forecast as forecast
import sincon_intraday_ensemble as intraday
from sincon_hierarchical_experiments import FOLDS, MIN_HISTORY_DAYS
from sincon_lightgbm_experiments import HOLIDAY_FLAGS, attach_calendar, holiday_snapshot


def calendar_kernel(left, right):
    dates_left = left["date"].to_numpy(dtype="datetime64[D]").astype("int64")
    dates_right = right["date"].to_numpy(dtype="datetime64[D]").astype("int64")
    separation = np.abs(dates_left[:, None] - dates_right[None, :])
    weekday = left["weekday"].to_numpy()[:, None] == right["weekday"].to_numpy()[None, :]
    regime = left["regime"].to_numpy()[:, None] == right["regime"].to_numpy()[None, :]
    annual = np.exp(-2 * np.sin(np.pi * separation / 365.2425) ** 2 / 0.55**2)
    nearby = np.exp(-separation / 85)
    return 1.0 + 2.0 * weekday + 2.5 * regime + (0.25 + 0.75 * regime) * (1.8 * annual + 0.8 * nearby)


def dates_and_regimes(frame):
    dates = frame.drop_duplicates("date").sort_values("date")
    regime = np.select(
        [dates["is_lunar_holiday"].eq(1), dates["holiday_type"].gt(0), dates["is_lunar_season"],
         dates["is_bridge_day"], dates["is_pre_long_offday"], dates["is_post_long_offday"],
         dates["weekday"].ge(5)],
        [2, 3, 4, 5, 6, 7, 1], default=0,
    )
    return dates[["date", "weekday"]].assign(regime=regime).reset_index(drop=True)


def clip_unusual_shapes(values, days):
    result = values.copy()
    for weekday in range(7):
        ordinary = days["weekday"].eq(weekday) & days["regime"].le(1)
        if ordinary.sum() < 8:
            continue
        group = values[ordinary]
        center = np.median(group, axis=0)
        spread = 1.4826 * np.median(np.abs(group - center), axis=0)
        limit = np.maximum(0.25, 3 * spread)
        result[ordinary] = np.clip(group, center - limit, center + limit)
    return result - result.mean(axis=2, keepdims=True)


def kernel_predict(gram, cross, targets, penalty):
    from scipy.linalg import cho_factor, cho_solve

    shape = targets.shape
    values = targets.reshape(shape[0], -1)
    center = values.mean(axis=0)
    factor = cho_factor(gram + penalty * np.eye(len(gram)))
    weights = cho_solve(factor, values - center)
    return (center + cross @ weights).reshape((len(cross), *shape[1:]))


def kernel_forecasts(history, future, calendar, cutoff, fallback):
    if history["date"].max() > cutoff or future["date"].min() <= cutoff or "boardings" in future:
        raise ValueError("Kernel forecasts require pre-origin history and unlabeled future rows.")
    snapshot = holiday_snapshot(calendar, cutoff)
    history = attach_calendar(history, snapshot, cutoff)
    future = attach_calendar(future, snapshot, cutoff)
    counts = history.groupby("station_id")["date"].nunique()
    first_dates = history.groupby("station_id")["date"].min()
    warm = counts.index[counts.ge(MIN_HISTORY_DAYS)]
    target_days = dates_and_regimes(future)
    direct = fallback.copy()
    separated = fallback.copy()
    hours = list(range(6, 24))

    for first_date, stations in first_dates.loc[warm].groupby(first_dates.loc[warm]).groups.items():
        observed = history.loc[history["station_id"].isin(stations) & history["date"].ge(first_date)]
        days = dates_and_regimes(observed)
        source_keys = pd.MultiIndex.from_product([days["date"], stations, hours], names=forecast.KEY)
        actual = observed.set_index(forecast.KEY)["boardings"].reindex(source_keys).to_numpy(dtype="float64")
        if not np.isfinite(actual).all():
            raise ValueError("Station groups must have all 18 observed hours on each training day.")
        logs = np.log1p(actual).reshape(len(days), len(stations), len(hours))
        gram = calendar_kernel(days, days)
        cross = calendar_kernel(target_days, days)
        simple = kernel_predict(gram, cross, logs, penalty=1.5)
        levels = logs.mean(axis=2, keepdims=True)
        shape = clip_unusual_shapes(logs - levels, days)
        split = kernel_predict(gram, cross, levels, penalty=1.2) + kernel_predict(gram, cross, shape, penalty=2.5)

        target_keys = pd.MultiIndex.from_product([target_days["date"], stations, hours], names=forecast.KEY)
        mask = future["station_id"].isin(stations)
        index = pd.MultiIndex.from_frame(future.loc[mask, forecast.KEY])
        direct[mask] = pd.Series(simple.ravel(), index=target_keys).reindex(index).to_numpy()
        separated[mask] = pd.Series(split.ravel(), index=target_keys).reindex(index).to_numpy()
    return direct, separated


def predict_at(history, future, calendar, cutoff, threads):
    candidates, calibration_end = intraday.predict_at(history, future, calendar, cutoff, threads)
    baseline = np.log1p(candidates["fixed_day_learned_hour"])
    direct, separated = kernel_forecasts(history, future, calendar, cutoff, baseline)
    levels, shapes = intraday.split_levels_and_shapes(
        future.assign(baseline=baseline, separated=separated), ["baseline", "separated"],
    )
    logs = {
        "previous_best": baseline,
        "calendar_kernel": direct,
        "kernel_daily_and_robust_hourly": separated,
        "kernel_blend": 0.65 * baseline + 0.35 * separated,
        "kernel_shape_blend": levels["baseline"].to_numpy() + (0.65 * shapes["baseline"] + 0.35 * shapes["separated"]).to_numpy(),
    }
    predictions = {name: np.expm1(np.maximum(values, 0)) for name, values in logs.items()}
    if not all(np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("Nonfinite calendar kernel predictions.")
    return predictions, calibration_end


def run(data_zip, calendar_csv, threads):
    train, test = forecast.read_inputs(data_zip)
    calendar = pd.read_csv(calendar_csv, usecols=["date", *HOLIDAY_FLAGS], parse_dates=["date"]).set_index("date").sort_index()
    if not calendar.index.equals(pd.date_range(train["date"].min(), test["date"].max())) or calendar.isna().any().any():
        raise ValueError("Calendar must cover every train and test date once.")

    results = {}
    for fold, cutoff in {**FOLDS, **intraday.EXTRA_FOLD}.items():
        history, validation = forecast.split_backtest(train, cutoff)
        predictions, calibration_end = predict_at(history, validation.drop(columns="boardings"), calendar, cutoff, threads)
        assert len(validation) == 157_194
        results[fold] = {"cutoff": str(cutoff.date()), "calibration_end": calibration_end,
                         "validation_rows": len(validation), "rmsle": {
                             name: forecast.rmsle(validation["boardings"], values) for name, values in predictions.items()
                         }}
        print(json.dumps({fold: results[fold]}, allow_nan=False), flush=True)
    pooled = {name: float(np.sqrt(np.mean([results[fold]["rmsle"][name]**2 for fold in forecast.BACKTESTS])))
              for name in predictions}
    return {"folds": results, "pooled_existing_folds": pooled, "best": min(pooled, key=pooled.get)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_zip", type=Path)
    parser.add_argument("calendar_csv", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    output = run(args.data_zip, args.calendar_csv, args.threads)
    if args.results:
        args.results.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"pooled_existing_folds": output["pooled_existing_folds"], "best": output["best"]}, allow_nan=False), flush=True)
