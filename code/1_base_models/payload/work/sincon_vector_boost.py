"""Use XGBoost vector leaves to predict all 18 hourly outputs jointly."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xgboost as xgb

from sincon_daily_network import FIRST_HOUR, HOURS, tensor_data
from sincon_joint_candidate import rmsle_logs, setup
from sincon_station_state import KEY, WINDOWS


HOUR_BANDS = ((0, 3), (3, 7), (7, 11), (11, 14), (14, 18))


def design(frame, daily_names, station_ids, hour_names):
    station, weekday, daily, hour_weather = tensor_data(frame, daily_names, station_ids)
    extra = frame[hour_names].to_numpy(dtype=np.float32, copy=True).reshape(-1, HOURS, len(hour_names))
    continuous = np.concatenate([hour_weather[:, :, :7], extra], axis=2)
    bands = np.column_stack([np.mean(continuous[:, start:end], axis=1) for start, end in HOUR_BANDS])
    time_from_event = hour_weather[:, :, 7:]
    arrival_peak = np.exp(-0.5 * ((time_from_event + 0.5) / 1.25) ** 2)
    departure_peak = np.exp(-0.5 * ((time_from_event - 0.5) / 1.25) ** 2)
    sports = np.nan_to_num(np.concatenate([arrival_peak[:, :, :2], departure_peak[:, :, 2:]], axis=2))
    geo = frame.iloc[::HOURS][["station_id", "line"]].to_numpy(dtype=np.float32)
    return np.column_stack([np.eye(len(station_ids))[station], np.eye(7)[weekday], geo, daily, bands, sports.reshape(len(geo), -1)]).astype(np.float32)


def forecast(past, future, daily_names, station_ids, hour_names, threads, rounds, mode):
    if past["date"].max() >= future["date"].min() or future["boardings"].notna().any():
        raise ValueError("Vector boosting requires earlier labels and an unlabeled forecast window.")
    inputs = design(past, daily_names, station_ids, hour_names)
    target = design(future, daily_names, station_ids, hour_names)
    labels = np.log1p(past["boardings"].to_numpy(dtype=np.float32)).reshape(-1, HOURS)
    if mode == "shape":
        labels = labels - labels.mean(axis=1, keepdims=True)
    dataset = xgb.QuantileDMatrix(inputs, label=labels, max_bin=64, nthread=threads)
    params = {"objective": "reg:squarederror", "tree_method": "hist", "multi_strategy": "multi_output_tree",
              "max_depth": 6, "max_bin": 64, "learning_rate": 0.06, "min_child_weight": 30,
              "reg_lambda": 4.0, "subsample": 0.9, "colsample_bytree": 0.85, "nthread": threads,
              "verbosity": 0, "seed": 19}
    model = xgb.train(params, dataset, num_boost_round=rounds)
    estimate = model.inplace_predict(target).astype(float)
    if estimate.shape != (len(future) // HOURS, HOURS):
        raise ValueError("Expected a full intraday prediction from every tree leaf.")
    return estimate


def run(args):
    data, _, daily, features, metadata, _, _ = setup(args.team_dir, args.external_zip, args.scoreboard,
                                                    args.mobility_directory, args.station_addresses, args.threads)
    station_ids = {station: index for index, station in enumerate(metadata.index)}
    daily = daily + [name for name in features if name.startswith("daily_arrivals_")]
    hourly = [name for name in features if name.startswith(("arrivals_", "hourly_share_"))]
    archive = pd.read_csv(args.validation, parse_dates=["date"])
    result, predictions = {}, []
    for fold, (cutoff, end) in WINDOWS.items():
        started = perf_counter()
        past = data.loc[data["date"].le(cutoff)]
        future = data.loc[data["date"].gt(cutoff) & data["date"].le(end)].assign(boardings=np.nan)
        reference = archive.loc[archive["fold"].eq(fold)].set_index(KEY).reindex(pd.MultiIndex.from_frame(future[KEY]))
        actual, base = (reference[name].to_numpy() for name in ("y_log", "conditional_stack"))
        center = base.reshape(-1, HOURS).mean(axis=1, keepdims=True)
        variants = {"reference": base}
        for mode in args.modes:
            estimate = forecast(past, future, daily, station_ids, hourly, args.threads, args.rounds, mode)
            if mode == "shape":
                estimate = center + estimate - estimate.mean(axis=1, keepdims=True)
            delta = estimate - base.reshape(-1, HOURS)
            variants.update({mode: estimate.ravel(), f"{mode}_blend": (base + 0.25 * delta.ravel()),
                             f"{mode}_shape_blend": (base + 0.25 * (delta - delta.mean(axis=1, keepdims=True)).ravel())})
        if not np.isfinite(np.column_stack([actual, *variants.values()])).all():
            raise ValueError(f"Nonfinite vector predictions in fold {fold}.")
        late = future["hour"].ge(FIRST_HOUR + HOURS - 3).to_numpy()
        result[fold] = {"rmsle": {name: rmsle_logs(actual, prediction) for name, prediction in variants.items()},
                        "rmsle_21_23": {name: rmsle_logs(actual[late], prediction[late]) for name, prediction in variants.items()},
                        "seconds": round(perf_counter() - started, 2)}
        predictions.append(future[KEY].assign(fold=fold, y_log=actual, **variants))
        args.results.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({fold: result[fold]}, allow_nan=False), flush=True)
    if args.predictions:
        pd.concat(predictions, ignore_index=True).to_csv(args.predictions, index=False, float_format="%.8f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team_dir", "external_zip", "scoreboard", "mobility_directory", "station_addresses", "validation"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=650)
    parser.add_argument("--modes", nargs="+", choices=["raw", "shape"], default=["raw", "shape"])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    run(parser.parse_args())
