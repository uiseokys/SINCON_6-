"""Backtest a forecast with a calendar-matched, growth-adjusted annual profile."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sincon_forecast as forecast
from sincon_hierarchical_experiments import FOLDS, MIN_HISTORY_DAYS
from sincon_lightgbm_experiments import FEATURES, HOLIDAY_FLAGS, attach_calendar, holiday_snapshot


def annual_analog(history, future, cutoff):
    if history["date"].max() > cutoff:
        raise ValueError("Annual analog history includes observations after the forecast origin.")
    ordinary = history["holiday_type"].eq(0) & ~history["is_lunar_season"] & ~history["is_bridge_day"]
    first_date = history.groupby("station_id")["date"].transform("min")
    old_enough = (history["date"] - first_date).dt.days.ge(MIN_HISTORY_DAYS)
    reference = history.loc[ordinary & old_enough, forecast.KEY + ["boardings"]].copy()
    reference["log_boardings"] = np.log1p(reference.pop("boardings").astype("float64"))
    past = reference.set_index(forecast.KEY)["log_boardings"]

    matches = []
    for offset in range(-28, 29, 7):
        reference_dates = future["date"] - pd.Timedelta(days=364 - offset)
        if reference_dates.max() > cutoff:
            raise ValueError("Annual reference dates must fall on or before the forecast origin.")
        keys = pd.MultiIndex.from_frame(future[forecast.KEY].assign(date=reference_dates))
        matches.append(past.reindex(keys).to_numpy())
    candidates = np.column_stack(matches)
    eligible = (
        (np.isfinite(candidates).sum(axis=1) >= 3)
        & future["holiday_type"].eq(0) & ~future["is_bridge_day"] & ~future["is_lunar_season"]
    ).to_numpy()

    recent = reference.loc[reference["date"].gt(cutoff - pd.Timedelta(days=56))].copy()
    paired_keys = pd.MultiIndex.from_frame(recent[forecast.KEY].assign(date=recent["date"] - pd.Timedelta(days=364)))
    recent["difference"] = recent["log_boardings"] - past.reindex(paired_keys).to_numpy()
    paired = recent.dropna(subset=["difference"])
    hour_key = ["station_id", "hour"]
    hourly = paired.groupby(hour_key)["difference"].agg(["median", "size"])
    station_prior = paired.groupby("station_id")["difference"].median()
    weight = hourly["size"] / (hourly["size"] + 12)
    growth = weight * hourly["median"] + (1 - weight) * hourly.index.get_level_values("station_id").map(station_prior)
    adjustment = growth.reindex(pd.MultiIndex.from_frame(future[hour_key])).fillna(0).to_numpy()

    analog = np.full(len(future), np.nan)
    analog[eligible] = np.nanmedian(candidates[eligible], axis=1) + adjustment[eligible]
    return analog, len(paired)


def forecast_components(history, future, calendar, cutoff, threads):
    if history["date"].max() > cutoff or future["date"].min() <= cutoff or "boardings" in future:
        raise ValueError("Prediction inputs must contain pre-origin history and unlabeled future rows.")
    snapshot = holiday_snapshot(calendar, cutoff)
    history = attach_calendar(history, snapshot, cutoff)
    future = attach_calendar(future, snapshot, cutoff)
    observations = history.groupby("station_id")["date"].nunique()
    cold = future["station_id"].map(observations).fillna(0).lt(MIN_HISTORY_DAYS).to_numpy()

    seasonal = np.log1p(forecast.seasonal_predict(history, future))
    lightgbm = np.log1p(forecast.lightgbm_predict(history, future, threads, features=FEATURES))
    base = np.where(cold, seasonal, lightgbm)
    analog, pairs = annual_analog(history, future, cutoff)
    special = (future["holiday_type"].gt(0) | future["offday_block_len"].ge(3)
               | future["is_pre_long_offday"] | future["is_post_long_offday"] | future["is_bridge_day"])
    return future[forecast.KEY].assign(
        base_log=base, seasonal_log=seasonal, annual_log=np.where(cold, np.nan, analog),
        cold=cold, special=special,
    ), pairs


def predict_at(history, future, calendar, cutoff, threads):
    components, pairs = forecast_components(history, future, calendar, cutoff, threads)
    base = components["base_log"].to_numpy()
    seasonal = components["seasonal_log"].to_numpy()
    analog = components["annual_log"].to_numpy()
    eligible = np.isfinite(analog)
    adjusted = np.where(eligible, 0.6 * base + 0.4 * analog, base)
    adjusted_seasonal = np.where(eligible, 0.6 * seasonal + 0.4 * analog, seasonal)
    log_predictions = {"baseline": base, "annual_analog_blend": adjusted, "seasonal_annual_blend": adjusted_seasonal}
    predictions = {name: np.expm1(np.maximum(values, 0)) for name, values in log_predictions.items()}
    if not all(np.isfinite(value).all() for value in predictions.values()):
        raise ValueError("Nonfinite forecasts in the annual analog model.")
    return predictions, {"eligible_rows": int(eligible.sum()), "paired_growth_rows": pairs}


def run(data_zip, calendar_csv, threads, submission_path):
    train, test = forecast.read_inputs(data_zip)
    calendar = pd.read_csv(calendar_csv, usecols=["date", *HOLIDAY_FLAGS], parse_dates=["date"]).set_index("date").sort_index()
    expected = pd.date_range(train["date"].min(), test["date"].max())
    if not calendar.index.equals(expected) or calendar.isna().any().any():
        raise ValueError("Calendar must cover every train and test date once.")

    results = {}
    for fold, cutoff in FOLDS.items():
        history, validation = forecast.split_backtest(train, cutoff)
        predictions, diagnostics = predict_at(history, validation.drop(columns="boardings"), calendar, cutoff, threads)
        assert len(validation) == 157_194
        scores = {name: forecast.rmsle(validation["boardings"], values) for name, values in predictions.items()}
        results[fold] = {"cutoff": str(cutoff.date()), "validation_rows": len(validation), "rmsle": scores, **diagnostics}
        print(json.dumps({fold: results[fold]}, allow_nan=False), flush=True)

    pooled = {name: float(np.sqrt(np.mean([results[fold]["rmsle"][name]**2 for fold in forecast.BACKTESTS])))
              for name in predictions}
    output = {"folds": results, "pooled_existing_folds": pooled}
    if submission_path is not None:
        predictions, diagnostics = predict_at(train, test.drop(columns="example_id"), calendar, forecast.CUTOFF, threads)
        submission = forecast.make_submission(test, predictions["annual_analog_blend"])
        forecast.write_submission(submission, submission_path)
        output["submission"] = {"rows": len(submission), "path": str(submission_path.resolve()), **diagnostics}
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_zip", type=Path)
    parser.add_argument("calendar_csv", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    output = run(args.data_zip, args.calendar_csv, args.threads, args.submission)
    if args.results:
        args.results.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(output, allow_nan=False), flush=True)
