"""Calibrate forecast weights on historical predictions before each outer cutoff."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sincon_forecast as forecast
from sincon_annual_analog import forecast_components
from sincon_hierarchical_experiments import FOLDS, cold_station_log_profile, decompose_day
from sincon_lightgbm_experiments import HOLIDAY_FLAGS, attach_calendar, holiday_snapshot

EXPERTS = ["base_log", "seasonal_log", "annual_log"]
DEFAULT_ANNUAL_WEIGHT = 0.4


def fit_weights(predictions, actual, row_weights, prior, penalty):
    from scipy.optimize import minimize

    X = predictions - predictions[:, :1]
    y = actual - predictions[:, 0]
    weights = row_weights / row_weights.sum()
    gram = X.T @ (weights[:, None] * X) + penalty * np.eye(X.shape[1])
    linear = X.T @ (weights * y) + penalty * prior
    result = minimize(
        lambda w: w @ gram @ w - 2 * linear @ w,
        prior,
        jac=lambda w: 2 * (gram @ w - linear),
        bounds=[(0.0, 1.0)] * len(prior),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
        method="SLSQP", options={"ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Forecast-weight optimization failed: {result.message}")
    return result.x


def calibration_rows(history, calendar, cutoff, threads, component_builder=forecast_components):
    parts = []
    for days in (forecast.HORIZON, 56):
        origin = cutoff - pd.Timedelta(days=days)
        fit = history.loc[history["date"].le(origin)]
        holdout = history.loc[history["date"].gt(origin) & history["date"].le(cutoff)]
        if fit.empty or holdout.empty:
            continue
        components, _ = component_builder(fit, holdout.drop(columns="boardings"), calendar, origin, threads)
        parts.append(components.assign(
            actual_log=np.log1p(holdout["boardings"].astype("float64")), calibration_origin=origin,
        ).loc[lambda frame: ~frame["cold"]])
    data = pd.concat(parts, ignore_index=True)
    if data.empty or data["date"].max() > cutoff:
        raise ValueError("Calibration must use earlier forecasts scored by the outer cutoff.")
    return data.assign(row_weight=1 / data.groupby("date")["calibration_origin"].transform("nunique"))


def estimate_weights(data, by_regime):
    weights = {}
    available = data["annual_log"].notna()
    for has_annual in (False, True):
        columns = EXPERTS if has_annual else EXPERTS[:2]
        prior = np.array([0.6, 0., 0.4] if has_annual else [1., 0.])
        group = data.loc[available.eq(has_annual)]
        learned = prior
        if group["date"].nunique() >= 8:
            learned = fit_weights(group[columns].to_numpy(), group["actual_log"].to_numpy(),
                                  group["row_weight"].to_numpy(), prior, penalty=0.005)
        for special in (False, True):
            segment = group.loc[group["special"].eq(special)]
            weight = learned
            if by_regime and segment["date"].nunique() >= 8:
                weight = fit_weights(segment[columns].to_numpy(), segment["actual_log"].to_numpy(),
                                     segment["row_weight"].to_numpy(), learned, penalty=0.010)
            weights[(has_annual, special)] = weight
    return weights


def apply_weights(components, weights):
    log_values = components["base_log"].to_numpy(copy=True)
    available = components["annual_log"].notna()
    for (has_annual, special), weight in weights.items():
        mask = available.eq(has_annual) & components["special"].eq(special) & ~components["cold"]
        columns = EXPERTS if has_annual else EXPERTS[:2]
        log_values[mask] = components.loc[mask, columns].to_numpy() @ weight
    return log_values


def last_weekday_profile(history, future, calendar, cutoff, components, initial):
    cold = components["cold"]
    adjusted = initial.copy()
    if cold.any():
        ids = components.loc[cold, "station_id"].unique()
        snapshot = holiday_snapshot(calendar, cutoff)
        dated_history = attach_calendar(history, snapshot, cutoff)
        cold_history = dated_history["station_id"].isin(ids)
        ordinary = dated_history["holiday_type"].eq(0) & dated_history["weekday"].lt(5)
        last_weekday = dated_history.loc[cold_history & ordinary].groupby("station_id")["date"].max()
        usable = dated_history["station_id"].isin(last_weekday.index)
        latest = dated_history["date"].eq(dated_history["station_id"].map(last_weekday))
        trimmed = decompose_day(dated_history.loc[~usable | latest])
        dated_future = attach_calendar(future.loc[cold], snapshot, cutoff)
        transferred = cold_station_log_profile(trimmed, dated_future, ids)
        adjusted[cold] = np.where(np.isfinite(transferred), transferred, initial[cold])
    return adjusted


def predict_at(history, future, calendar, cutoff, threads):
    components, diagnostics = forecast_components(history, future, calendar, cutoff, threads)
    available = components["annual_log"].notna()
    base = components["base_log"].to_numpy()
    annual = components["annual_log"].to_numpy()
    fixed = np.where(available, (1 - DEFAULT_ANNUAL_WEIGHT) * base + DEFAULT_ANNUAL_WEIGHT * annual, base)
    calibration = calibration_rows(history, calendar, cutoff, threads)
    global_weights = estimate_weights(calibration, by_regime=False)
    regime_weights = estimate_weights(calibration, by_regime=True)
    global_blend = apply_weights(components, global_weights)
    regime_blend = apply_weights(components, regime_weights)
    cold = components["cold"].to_numpy()
    weekday_profile = last_weekday_profile(history, future, calendar, cutoff, components, base)
    alternate_cold = np.where(cold, weekday_profile, regime_blend)
    fixed_with_cold_profile = np.where(cold, weekday_profile, fixed)
    predictions = {
        name: np.expm1(np.maximum(log_values, 0)) for name, log_values in {
            "fixed_annual_blend": fixed,
            "calibrated_global": global_blend,
            "calibrated_by_regime": regime_blend,
            "calibrated_regime_last_weekday": alternate_cold,
            "fixed_annual_last_weekday": fixed_with_cold_profile,
            "fixed_when_annual_available": np.where(available, fixed, alternate_cold),
        }.items()
    }
    if not all(np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("Nonfinite calibrated forecast.")
    summary = {
        "calibration_end": str(calibration["date"].max().date()),
        "calibration_origins": [str(value.date()) for value in sorted(calibration["calibration_origin"].unique())],
        "paired_growth_rows": diagnostics,
        "regime_weights": {str(key): value.tolist() for key, value in regime_weights.items()},
    }
    return predictions, summary


def error_breakdown(validation, predictions):
    log_error = np.log1p(predictions) - np.log1p(validation["boardings"].to_numpy(dtype="float64"))
    rows = validation[["date", "station_id"]].assign(log_error=log_error)
    daily_level = rows.groupby(["date", "station_id"])["log_error"].transform("mean").to_numpy()
    return {"daily_level_rms": float(np.sqrt(np.mean(daily_level ** 2))),
            "hourly_shape_rms": float(np.sqrt(np.mean((log_error - daily_level) ** 2)))}


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
        decomposition = error_breakdown(validation, predictions["fixed_annual_blend"])
        results[fold] = {"cutoff": str(cutoff.date()), "validation_rows": len(validation), "rmsle": scores,
                         "fixed_blend_error_breakdown": decomposition, **diagnostics}
        print(json.dumps({fold: results[fold]}, allow_nan=False), flush=True)

    pooled = {name: float(np.sqrt(np.mean([results[fold]["rmsle"][name]**2 for fold in forecast.BACKTESTS])))
              for name in predictions}
    best = min(pooled, key=pooled.get)
    output = {"folds": results, "pooled_existing_folds": pooled, "best": best}
    if submission_path is not None:
        predictions, diagnostics = predict_at(train, test.drop(columns="example_id"), calendar, forecast.CUTOFF, threads)
        submission = forecast.make_submission(test, predictions[best])
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
    print(json.dumps({"pooled_existing_folds": output["pooled_existing_folds"], "best": output["best"]}, allow_nan=False), flush=True)
