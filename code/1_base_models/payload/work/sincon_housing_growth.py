"""Estimate added subway demand near newly completed apartment complexes."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from sincon_bus_stop_data import locations
from sincon_joint_candidate import rmsle_logs
from sincon_station_state import KEY, WINDOWS, read_team


OFF = ["is_holiday", "is_bridge_day", "is_quasi_holiday", "is_pre_long_offday", "is_post_long_offday"]
NEW_STATIONS = [3107, 3108, 3109]


def exposure(housing, centers, days, ramp_days):
    coordinates = ["latitude", "longitude"]
    meter = np.array([111_195, 111_195 * np.cos(np.deg2rad(37.5))])
    distance = cdist(centers[coordinates].to_numpy() * meter, housing[coordinates].to_numpy() * meter)
    weight = np.exp(-0.5 * (distance / 550) ** 2) * (distance <= 1_600)
    # Nearby stations share the same local pool of new residents.
    weight /= np.maximum(1, weight.sum(axis=0))
    # Use the midpoint of the recorded first occupancy month as the center
    # of a gradual move-in window, never starting before legal approval.
    occupancy = housing["first_month"].fillna(housing["approval_date"] + pd.Timedelta(days=21))
    earliest = occupancy - pd.Timedelta(days=ramp_days / 2)
    start = pd.concat([earliest, housing["approval_date"]], axis=1).max(axis=1)
    elapsed = (days.to_numpy()[:, None] - start.to_numpy()[None, :]) / np.timedelta64(1, "D")
    inhabited = np.clip(elapsed / ramp_days, 0, 1) * housing["units"].to_numpy()
    return pd.DataFrame(inhabited @ weight.T, index=days, columns=centers.index)


def day_array(frame, value, dates, stations):
    hours = list(range(6, 24))
    index = pd.MultiIndex.from_product([dates, stations, hours], names=KEY)
    values = frame.set_index(KEY)[value].reindex(index).to_numpy(dtype=float)
    return values.reshape(len(dates), len(stations), len(hours))


def station_levels(history, cutoff, regular, dates, stations):
    counts = day_array(history, "boardings", dates, stations)
    with np.errstate(invalid="ignore"):
        daily = counts.sum(axis=-1)
        median = np.nanmedian(daily, axis=0)
    recent = (dates > cutoff - pd.Timedelta(days=56)) & regular
    weekend = dates.dayofweek >= 5
    norm = np.stack([np.nanmedian(daily[recent & (weekend == value)], axis=0) for value in (False, True)])
    norm = np.where(np.isfinite(norm), norm, median[None, :])
    return np.maximum(norm, 500), np.log1p(counts)


def learn_trip_response(history, cutoff, housing_level, regular, meta):
    dates = pd.date_range(history["date"].min(), cutoff)
    stations = housing_level.columns
    regular = regular.reindex(dates).fillna(False).to_numpy()
    norm, logs = station_levels(history, cutoff, regular, dates, stations)
    weekend = (dates.dayofweek >= 5).astype(int)
    typical = ~stations.isin(NEW_STATIONS + [3125, 3127, 3139, 3209, 3754])
    eligible = dates.to_numpy()[:, None] >= (
        meta["first_date"].reindex(stations).to_numpy()[None, :] + np.timedelta64(35, "D")
    )
    residential = housing_level.reindex(dates).to_numpy()
    past_units = residential / norm[weekend]
    changes, outcomes, types = [], [], []
    for lag in (28, 56):
        both = regular[lag:] & regular[:-lag]
        delta = logs[lag:] - logs[:-lag]
        # Remove time effects shared across the 71 stations on the same days.
        relative = delta - np.nanmedian(delta[:, typical], axis=1, keepdims=True)
        move = past_units[lag:] - past_units[:-lag]
        usable = both[:, None] & eligible[:-lag] & (move > .007) & np.isfinite(relative).all(axis=2)
        changes.append(move[usable])
        outcomes.append(relative[usable])
        types.append(np.broadcast_to(weekend[lag:, None], move.shape)[usable])
    x, y, w = np.concatenate(changes), np.concatenate(outcomes), np.concatenate(types)
    response = []
    for weekend_type, prior in enumerate((0.50, 0.35)):
        choose = w == weekend_type
        # Partial pooling over all stations and hours, equivalent to roughly
        # 20 independent station-day exposures of 0.1 units per daily boarding.
        x_type = x[choose]
        y_type = y[choose].clip(-0.55, 0.55)
        gram = x_type @ x_type
        moment = x_type @ y_type
        common = (np.mean(moment) + 0.20 * prior) / (gram + 0.20)
        hourly = (moment + (0.20 + gram) * common) / (2 * gram + 0.20)
        response.append(np.clip(hourly, 0, 2))
    return norm, np.array(response), [int((w == value).sum()) for value in (0, 1)]


def forecast_changes(history, target, cutoff, housing_level, calendar, meta):
    if history["date"].max() > cutoff or target["date"].min() <= cutoff or "y_log" in target or "boardings" in target:
        raise ValueError("Housing effect estimation requires strictly earlier boardings and unlabeled keys.")
    ordinary = calendar[OFF].eq(0).all(axis=1)
    norm, response, samples = learn_trip_response(history, cutoff, housing_level, ordinary, meta)
    days = pd.date_range(target["date"].min(), target["date"].max())
    x = housing_level.reindex(days).to_numpy() - housing_level.loc[cutoff].to_numpy()
    weekend = ((days.dayofweek >= 5) | calendar["is_holiday"].reindex(days).eq(1).to_numpy()).astype(int)
    delta = x / norm[weekend]
    adjustment = np.log1p(delta[:, :, None] * response[weekend][:, None, :]).clip(0, 0.60)
    day_level = np.broadcast_to(adjustment.mean(axis=2, keepdims=True), adjustment.shape)
    index = pd.MultiIndex.from_product([days, housing_level.columns, list(range(6, 24))], names=KEY)
    future_index = pd.MultiIndex.from_frame(target[KEY])
    hourly = pd.Series(adjustment.ravel(), index=index).reindex(future_index).to_numpy()
    level = pd.Series(day_level.ravel(), index=index).reindex(future_index).to_numpy()
    return hourly, level, {"observations": samples, "weekday_rate": response[0].tolist(), "weekend_rate": response[1].tolist()}


def read_exposure(path, centers):
    housing = pd.read_csv(path, parse_dates=["approval_date"], dtype={"first_month": str})
    housing["first_month"] = pd.to_datetime(housing["first_month"], format="%Y%m", errors="coerce") + pd.Timedelta(days=14)
    if housing["complex_id"].duplicated().any() or not housing["units"].gt(0).all():
        raise ValueError("Each apartment complex needs one record with a positive unit count.")
    return exposure(housing, centers, pd.date_range("2025-01-01", "2026-10-31"), ramp_days=60)


def run(args):
    train, meta, calendar, _ = read_team(args.team_dir, args.external_zip)
    centers, _ = locations(args.entrances, args.bus_coordinates, pd.read_csv(args.team_dir / "data/metaData.csv"))
    exposed = read_exposure(args.housing, centers)
    validation = pd.read_csv(args.validation, parse_dates=["date"])
    results, outputs = {}, []
    for fold, (origin, _) in WINDOWS.items():
        started = perf_counter()
        cutoff = pd.Timestamp(origin)
        past = train.loc[train["date"].le(cutoff)]
        actual = validation.loc[validation["fold"].eq(fold)]
        target = actual[KEY + ["conditional_stack"]]
        base, true = (actual[name].to_numpy() for name in ("conditional_stack", "y_log"))
        hourly, level, diagnostics = forecast_changes(past, target, cutoff, exposed, calendar, meta)
        scores = {"reference": base, "housing_level": base + level, "housing_hourly": base + hourly,
                  "housing_shrunk": base + 0.5 * hourly}
        affected = hourly > 0.01
        results[fold] = {"rmsle": {key: rmsle_logs(true, values) for key, values in scores.items()},
                         "affected_rmsle": {key: rmsle_logs(true[affected], values[affected]) for key, values in scores.items()},
                         "affected_rows": int(affected.sum()), "learned": diagnostics, "seconds": round(perf_counter() - started, 2)}
        outputs.append(target[KEY].assign(fold=fold, **scores))
        print(json.dumps({fold: results[fold]}, allow_nan=False), flush=True)
    args.results.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.predictions:
        pd.concat(outputs, ignore_index=True).to_csv(args.predictions, index=False, float_format="%.9f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team_dir", "external_zip", "validation", "housing", "entrances", "bus_coordinates"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    run(parser.parse_args())
