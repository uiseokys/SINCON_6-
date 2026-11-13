"""Predict station-level growth after the Geomdan extension using local bus demand."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from sincon_station_state import KEY, WINDOWS, read_team


CORRIDOR = (3107, 3108, 3109, 3110, 3111)
OPEN = pd.Timestamp("2025-06-28")
PLAIN = ["is_holiday", "is_bridge_day", "is_quasi_holiday", "is_pre_long_offday", "is_post_long_offday"]


def observed_bus(path):
    raw = pd.read_csv(path)
    raw["month"] = pd.PeriodIndex(raw["month"], freq="M")
    share = 0.5 * (np.log1p(raw["bus_boardings"]) + np.log1p(raw["bus_alightings"]))
    regional = 0.5 * (np.log1p(raw["bus_city_boardings"]) + np.log1p(raw["bus_city_alightings"]))
    values = raw.assign(bus=share - regional).pivot(index="month", columns="station_id", values="bus")
    months = pd.period_range("2025-01", "2026-10", freq="M")
    return values.reindex(months).ffill(limit=1).loc[:, list(CORRIDOR)]


def relative(frame, value, controls):
    daily = frame.groupby(["date", "station_id"], observed=True)[value].mean().unstack("station_id")
    average = daily.loc[:, controls].mean(axis=1)
    return daily.loc[:, list(CORRIDOR)].sub(average, axis=0)


def predictors(dates, station, bus):
    month = dates.to_period("M")
    x = bus[station].reindex(month).to_numpy(dtype=float)
    elapsed = np.maximum((dates - OPEN).days.to_numpy(dtype=float), 0)
    # The adoption curve progressively flattens as the station matures.
    age = 1 - np.exp(-elapsed / 180)
    return x, age


def fit_levels(past, bus, day_type):
    subset = past.loc[day_type]
    months = subset.index.to_period("M")
    filtered = subset.groupby(months).mean()
    dates = pd.DatetimeIndex(subset.index.to_series(index=months).groupby(level=0).mean()).normalize()
    features = []
    for station in CORRIDOR:
        x, age = predictors(dates, station, bus)
        features.append((x, age, filtered[station].to_numpy(dtype=float)))
    station_count = len(CORRIDOR)
    rows, labels, centers = [], [], {}
    for index, (station, (x, age, y)) in enumerate(zip(CORRIDOR, features)):
        usable = np.isfinite(x) & np.isfinite(y)
        if usable.sum() < 3:
            raise ValueError(f"Too few pre-origin months for station {station}.")
        mean_x, mean_age = float(np.mean(x[usable])), float(np.mean(age[usable]))
        centers[station] = mean_x, mean_age
        matrix = np.zeros((int(usable.sum()), station_count * 3))
        matrix[:, index] = 1
        matrix[:, station_count + index] = x[usable] - mean_x
        matrix[:, 2 * station_count + index] = age[usable] - mean_age
        rows.append(matrix)
        labels.append(y[usable])
    # Shrink local bus elasticities toward a common corridor response and
    # permit slower growth with age only for the three new stations.
    p = station_count * 3
    prior = np.zeros(p)
    prior[station_count:2 * station_count] = [0.55, 0.55, 0.55, 0.25, 0.25]
    prior[2 * station_count:] = [0.14, 0.14, 0.14, 0, 0]
    precision = np.sqrt(np.r_[np.zeros(station_count), np.full(station_count, 0.65), np.full(station_count, 0.42)])
    design = np.vstack([*rows, np.diag(precision)])
    target = np.concatenate([*labels, prior * precision])
    low = np.r_[np.full(station_count, -np.inf), np.zeros(station_count), np.zeros(3), -0.7, -0.7]
    high = np.r_[np.full(station_count, np.inf), np.full(station_count, 1.3), np.full(3, 1.2), 0.7, 0.7]
    fit = lsq_linear(design, target, bounds=(low, high)).x.reshape(3, station_count)
    return centers, fit


def shifts(history, future, controls, bus, calendar, origin):
    if history["date"].max() > origin or future["date"].min() <= origin or "y_log" in future:
        raise ValueError("Growth fit requires pre-origin observations and unlabeled forecasts.")
    # The opening weekend is a partial service month and is not comparable
    # with the full-month bus statistics used in this regression.
    first_full_month = OPEN + pd.offsets.MonthBegin(1)
    observed_months = bus.loc[first_full_month.to_period("M"):origin.to_period("M")]
    if observed_months.notna().sum().min() < 3:
        return np.zeros(len(future)), {"reason": "Fewer than three post-opening bus months"}
    history = history.loc[history["date"].ge(first_full_month)].assign(log=lambda frame: np.log1p(frame["boardings"]))
    observed = relative(history, "log", controls)
    reference = relative(future, "joint", controls)
    reg = calendar[PLAIN].eq(0).all(axis=1)
    adjustment = pd.DataFrame(0., index=reference.index, columns=reference.columns)
    details = {}
    for weekend in (False, True):
        train_days = reg.reindex(observed.index).fillna(False) & ((observed.index.dayofweek >= 5) == weekend)
        forecast_days = reg.reindex(reference.index).fillna(False) & ((reference.index.dayofweek >= 5) == weekend)
        centers, coefficients = fit_levels(observed, bus, train_days)
        details["weekend" if weekend else "weekday"] = {str(station): coefficients[1:, i].tolist() for i, station in enumerate(CORRIDOR)}
        for i, station in enumerate(CORRIDOR):
            x, age = predictors(reference.index, station, bus)
            bx, ax = centers[station]
            predicted = coefficients[0, i] + coefficients[1, i] * (x - bx) + coefficients[2, i] * (age - ax)
            plain = reference.loc[forecast_days, station].groupby(reference.index[forecast_days].to_period("M")).mean()
            implied = plain.reindex(reference.index.to_period("M")).to_numpy()
            delta = np.clip(np.nan_to_num(predicted - implied), -0.50, 0.50)
            adjustment.loc[forecast_days, station] = delta[forecast_days.to_numpy()]
    station_day = adjustment.stack().rename("shift").rename_axis(["date", "station_id"])
    rows = future.join(station_day, on=KEY[:2], validate="many_to_one")
    return rows["shift"].fillna(0).to_numpy(), details


def run(team_dir, external, validation, bus_path, prediction_path):
    train, meta, calendar, _ = read_team(team_dir, external)
    controls = list(meta.index[meta["canonical_line"].eq(1) & ~meta.index.isin([*CORRIDOR, 3125, 3127, 3139])])
    bus = observed_bus(bus_path)
    joint = pd.read_csv(validation, parse_dates=["date"])
    results, saved = {}, []
    for fold, (cutoff, _) in WINDOWS.items():
        origin = pd.Timestamp(cutoff)
        past = train.loc[train["date"].le(origin)]
        fold_rows = joint.loc[joint["fold"].eq(fold)]
        target = fold_rows[[*KEY, "joint"]]
        truth = fold_rows["y_log"].to_numpy()
        base = target["joint"].to_numpy()
        move, diagnostics = shifts(past, target, controls, bus, calendar, origin)
        versions = {"joint": base, "bus_growth": base + 0.7 * move, "bus_growth_full": base + move}
        affected = target["station_id"].isin(CORRIDOR).to_numpy()
        def score(actual, estimate):
            return float(np.sqrt(np.mean(np.square(actual - np.maximum(estimate, 0)))))

        results[fold] = {"rmsle": {name: score(truth, values) for name, values in versions.items()},
                         "corridor_rmsle": {name: score(truth[affected], values[affected]) for name, values in versions.items()},
                         "coefficients": diagnostics}
        saved.append(target[KEY].assign(fold=fold, **versions))
    if prediction_path:
        pd.concat(saved, ignore_index=True).to_csv(prediction_path, index=False, float_format="%.7f")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team_dir", "external_zip", "validation", "bus"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    report = run(args.team_dir, args.external_zip, args.validation, args.bus, args.predictions)
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.results.write_text(output, encoding="utf-8")
    print(output, end="")
