"""Evaluate local bus volume as an independent station growth signal."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sincon_bus_stop_data import COUNTS
from sincon_joint_candidate import rmsle_logs, setup
from sincon_station_state import KEY, WINDOWS


def attach_bus(frame, path):
    bus = pd.read_csv(path)
    bus["bus_month"] = pd.PeriodIndex(bus.pop("month"), freq="M")
    if bus.duplicated(["bus_month", "station_id"]).any() or bus.groupby("bus_month")["station_id"].size().ne(71).any():
        raise ValueError("Bus aggregates need one value for all 71 stations in each published month.")
    board, alight = (np.log1p(bus.pop(name)) for name in COUNTS)
    city_board = np.log1p(bus.pop("bus_city_boardings"))
    city_alight = np.log1p(bus.pop("bus_city_alightings"))
    bus = bus.assign(bus_board=board, bus_alight=alight, bus_board_share=board - city_board, bus_alight_share=alight - city_alight)
    joined = frame.assign(bus_month=frame["date"].dt.to_period("M")).merge(
        bus, on=["bus_month", "station_id"], how="left", validate="many_to_one",
    )
    return joined, [name for name in bus.columns if name.startswith("bus_") and name != "bus_month"]


def bus_movement(past, future):
    features = ["bus_board", "bus_alight", "bus_board_share", "bus_alight_share"]
    # Normalize around each station's own published range before the cutoff.
    # A row with absent or effectively zero local coverage uses the prior model.
    observed = (past.loc[past["bus_stop_count"].ge(2), ["station_id", "bus_month", *features]]
                .drop_duplicates(["station_id", "bus_month"]))
    base = observed.groupby("station_id")[features].median()
    for frame in (past, future):
        station_center = base.reindex(frame["station_id"]).to_numpy()
        change = frame[features].to_numpy() - station_center
        for col, values in zip(features, change.T):
            frame[f"{col}_relative"] = np.where(frame["bus_stop_count"].ge(2), np.clip(values, -1, 1), np.nan)
    return [f"{name}_relative" for name in features]


def run(args):
    frame, _, _, features, _, _, fit_predict = setup(args.team_dir, args.external_zip, args.scoreboard,
                                                    args.mobility_directory, args.station_addresses, args.threads)
    frame, bus_features = attach_bus(frame, args.bus)
    validation = pd.read_csv(args.validation, parse_dates=["date"])
    output, parts = {}, []
    for fold, (cutoff, end) in WINDOWS.items():
        started = perf_counter()
        fit = frame.loc[frame["date"].le(cutoff)].copy()
        target = frame.loc[frame["date"].gt(cutoff) & frame["date"].le(end)].assign(boardings=np.nan)
        additions = bus_movement(fit, target)
        archived = validation.loc[validation["fold"].eq(fold)].set_index(KEY).reindex(pd.MultiIndex.from_frame(target[KEY]))
        true, joint, old_tree = (archived[name].to_numpy() for name in ("y_log", "joint", "mobility_lgb"))
        forecast = np.log1p(fit_predict(fit, target, features + bus_features + additions, list(range(args.seeds))))
        if not np.isfinite(np.column_stack([true, joint, old_tree, forecast])).all():
            raise ValueError(f"Incomplete forecasts in {fold}.")
        coverage = target["bus_stop_count"].ge(2).to_numpy()
        difference = np.where(coverage, forecast - old_tree, 0)
        level = target[KEY[:2]].assign(delta=difference).groupby(KEY[:2])["delta"].transform("mean").to_numpy()
        withheld = target["bus_month"].eq(pd.Period(end, freq="M")).to_numpy()
        changed = joint + 0.375 * difference + 0.125 * level
        predictions = {"joint": joint, "bus_replaces_tree": changed, "bus_half": (joint + changed) / 2,
                       "bus_level_only": joint + 0.5 * level, "bus_without_final_month": np.where(withheld, joint, changed)}
        new = target["station_id"].isin([3107, 3108, 3109, 3110, 3111]).to_numpy()
        output[fold] = {"rmsle": {key: rmsle_logs(true, values) for key, values in predictions.items()},
                        "geomdan_rmsle": {key: rmsle_logs(true[new], values[new]) for key, values in predictions.items()},
                        "covered_rows": int(coverage.sum()), "seconds": round(perf_counter() - started, 2)}
        parts.append(target[KEY].assign(fold=fold, **predictions))
        args.results.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({fold: output[fold]}, allow_nan=False), flush=True)
    if args.predictions:
        pd.concat(parts, ignore_index=True).to_csv(args.predictions, index=False, float_format="%.7f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team_dir", "external_zip", "scoreboard", "mobility_directory", "station_addresses", "validation", "bus"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    run(parser.parse_args())
