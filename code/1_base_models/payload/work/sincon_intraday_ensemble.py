"""Compare separate calibration of station-day levels and within-day demand patterns."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sincon_forecast as forecast
from sincon_annual_analog import forecast_components
from sincon_calibrated_ensemble import (
    EXPERTS, apply_weights, calibration_rows, error_breakdown, estimate_weights, fit_weights,
    last_weekday_profile,
)
from sincon_hierarchical_experiments import FOLDS, decompose_day, estimate_hourly_shape, lookup
from sincon_lightgbm_experiments import HOLIDAY_FLAGS, attach_calendar, holiday_snapshot

DAY_KEYS = ["date", "station_id"]
EXTRA_FOLD = {"year_end_origin": pd.Timestamp("2025-12-31")}


def split_levels_and_shapes(frame, columns):
    keys = DAY_KEYS + (["calibration_origin"] if "calibration_origin" in frame else [])
    levels = frame.groupby(keys)[columns].transform("mean")
    return levels, frame[columns] - levels


def with_profile(history, future, calendar, cutoff, threads):
    components, pairs = forecast_components(history, future, calendar, cutoff, threads)
    snapshot = holiday_snapshot(calendar, cutoff)
    dated_history = decompose_day(attach_calendar(history, snapshot, cutoff))
    dated_future = attach_calendar(future, snapshot, cutoff)
    profile = estimate_hourly_shape(dated_history, dated_future)
    fallback = split_levels_and_shapes(components, ["seasonal_log"])[1]["seasonal_log"].to_numpy()
    return components.assign(profile_shape=np.where(np.isfinite(profile), profile, fallback)), pairs


def fixed_reference(components, previous_weights):
    eligible = components["annual_log"].notna()
    annual = 0.6 * components["base_log"] + 0.4 * components["annual_log"]
    return np.where(eligible, annual, apply_weights(components, previous_weights))


def calibrate_components(calibration, future, previous_weights):
    shape_experts = EXPERTS + ["profile_shape"]
    levels, shapes = split_levels_and_shapes(calibration, shape_experts + ["actual_log"])
    target_levels, target_shapes = split_levels_and_shapes(future, shape_experts)
    future_level = future["base_log"].to_numpy(copy=True)
    future_shape = target_shapes["base_log"].to_numpy(copy=True)
    cal_available = calibration["annual_log"].notna()
    future_available = future["annual_log"].notna()
    for (has_annual, special), old_weight in previous_weights.items():
        columns = EXPERTS if has_annual else EXPERTS[:2]
        shape_columns = columns + ["profile_shape"]
        valid = cal_available.eq(has_annual) & calibration["special"].eq(special)
        target = future_available.eq(has_annual) & future["special"].eq(special) & ~future["cold"]
        weight_level = np.array([0.6, 0., 0.4]) if has_annual else old_weight
        weight_shape = np.r_[weight_level, 0.]
        if calibration.loc[valid, "date"].nunique() >= 12:
            row_weights = calibration.loc[valid, "row_weight"].to_numpy()
            weight_level = fit_weights(levels.loc[valid, columns].to_numpy(), levels.loc[valid, "actual_log"].to_numpy(),
                                       row_weights, weight_level, penalty=0.015)
            weight_shape = fit_weights(shapes.loc[valid, shape_columns].to_numpy(),
                                       shapes.loc[valid, "actual_log"].to_numpy(), row_weights, weight_shape, penalty=0.015)
        future_level[target] = target_levels.loc[target, columns].to_numpy() @ weight_level
        future_shape[target] = target_shapes.loc[target, shape_columns].to_numpy() @ weight_shape
    # Center each station-day again before adding it to a level forecast.
    centered = split_levels_and_shapes(future.assign(shape=future_shape), ["shape"])[1]["shape"].to_numpy()
    return future_level, centered


def learn_hourly_bias(calibration, future, previous_weights):
    predictions = fixed_reference(calibration, previous_weights)
    errors = calibration.assign(error=calibration["actual_log"] - predictions)
    shape_errors = split_levels_and_shapes(errors, ["error"])[1]["error"]
    errors = errors.assign(
        weekday=errors["date"].dt.dayofweek, weighted_error=shape_errors * errors["row_weight"],
    )
    target = future.assign(weekday=future["date"].dt.dayofweek)
    broad = ["station_id", "hour", "special"]
    fine = broad + ["weekday"]
    pooled = errors.groupby(broad)[["weighted_error", "row_weight"]].sum()
    prior = lookup(pooled["weighted_error"] / (pooled["row_weight"] + 32), target, broad)
    local = errors.groupby(fine)[["weighted_error", "row_weight"]].sum()
    local_sum = np.nan_to_num(lookup(local["weighted_error"], target, fine))
    count = np.nan_to_num(lookup(local["row_weight"], target, fine))
    correction = 0.65 * (local_sum + 20 * np.nan_to_num(prior)) / (count + 20)
    return split_levels_and_shapes(target.assign(correction=correction), ["correction"])[1]["correction"].to_numpy()


def predict_at(history, future, calendar, cutoff, threads):
    components, _ = with_profile(history, future, calendar, cutoff, threads)
    calibration = calibration_rows(history, calendar, cutoff, threads, component_builder=with_profile)
    weights = estimate_weights(calibration, by_regime=True)
    reference = fixed_reference(components, weights)
    reference = last_weekday_profile(history, future, calendar, cutoff, components, reference)
    reference_level = split_levels_and_shapes(components.assign(reference=reference), ["reference"])[0]["reference"].to_numpy()
    new_level, new_shape = calibrate_components(calibration, components, weights)
    hourly_bias = learn_hourly_bias(calibration, components, weights)
    cold = components["cold"].to_numpy()
    log_predictions = {
        "previous_best": reference,
        "learned_day_and_hour": np.where(cold, reference, new_level + new_shape),
        "fixed_day_learned_hour": np.where(cold, reference, reference_level + new_shape),
        "fixed_day_hour_bias": np.where(cold, reference, reference + hourly_bias),
    }
    predictions = {name: np.expm1(np.maximum(values, 0)) for name, values in log_predictions.items()}
    if not all(np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("Nonfinite forecast in the intraday comparison.")
    return predictions, str(calibration["date"].max().date())


def run(data_zip, calendar_csv, threads):
    train, test = forecast.read_inputs(data_zip)
    calendar = pd.read_csv(calendar_csv, usecols=["date", *HOLIDAY_FLAGS], parse_dates=["date"]).set_index("date").sort_index()
    expected = pd.date_range(train["date"].min(), test["date"].max())
    if not calendar.index.equals(expected) or calendar.isna().any().any():
        raise ValueError("Calendar must cover every train and test date once.")

    results = {}
    for fold, cutoff in {**FOLDS, **EXTRA_FOLD}.items():
        history, validation = forecast.split_backtest(train, cutoff)
        predictions, calibration_end = predict_at(history, validation.drop(columns="boardings"), calendar, cutoff, threads)
        assert len(validation) == 157_194
        scores = {name: forecast.rmsle(validation["boardings"], values) for name, values in predictions.items()}
        diagnostics = {name: error_breakdown(validation, values) for name, values in predictions.items()}
        results[fold] = {"cutoff": str(cutoff.date()), "calibration_end": calibration_end,
                         "validation_rows": len(validation), "rmsle": scores, "error_breakdown": diagnostics}
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
