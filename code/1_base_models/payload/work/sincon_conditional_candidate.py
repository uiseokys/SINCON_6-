"""Train and validate the candidate with separate corridor and sporting-event models."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

from sincon_additive_model import predict as predict_additive
from sincon_bus_model import attach_bus, bus_movement
from sincon_corridor_bus_growth import CORRIDOR, observed_bus, shifts
from sincon_daily_network import HOURLY
from sincon_flow_mixture import predict as predict_mixture
from sincon_graph_network import predict as predict_graph
from sincon_joint_candidate import assemble, rmsle_logs, setup
from sincon_sports_stack import sports_forecasts, sports_inputs
from sincon_station_state import KEY, WINDOWS


FINAL_CUTOFF = "2026-06-30"


def by_station_day(frame, values):
    return frame[KEY[:2]].assign(value=values).groupby(KEY[:2])["value"].transform("mean").to_numpy()


def choose_stack(joint, shape, bus_level, corridor, mixture, sports, is_corridor, is_sports):
    general = shape + np.where(is_corridor, corridor, bus_level) + mixture - 2 * joint
    return np.where(is_sports, sports, general)


class Candidate:
    def __init__(self, args):
        data, self.archive, self.daily, self.features, stations, self.adjacency, self.fit_tree = setup(
            args.team_dir, args.external_zip, args.scoreboard, args.mobility_directory, args.station_addresses, args.threads,
        )
        self.data, self.bus_features = attach_bus(data, args.bus)
        self.bus = observed_bus(args.bus)
        self.stations = stations.index
        self.controls = list(stations.index[stations["canonical_line"].eq(1) & ~stations.index.isin([*CORRIDOR, 3125, 3127, 3139])])
        self.mixture_daily = self.daily + [name for name in self.features if name.startswith("daily_arrivals_")]
        self.mixture_hourly = HOURLY + [name for name in self.features if name.startswith(("arrivals_", "hourly_share_"))]
        with ZipFile(args.external_zip) as source:
            self.calendar = pd.read_csv(source.open("calendar_features.csv"), parse_dates=["date"]).set_index("date")
        self.sports = sports_inputs(args.external_zip, args.scoreboard, args.football, self.calendar)
        self.threads = args.threads

    def predict(self, history, future, neural, reference, cutoff, with_missing_test=False):
        if history["date"].max() > pd.Timestamp(cutoff) or future["date"].min() <= pd.Timestamp(cutoff) or future["boardings"].notna().any():
            raise ValueError("A fixed-origin forecast requires earlier labels and an unlabeled prediction window.")
        tree = np.log1p(self.fit_tree(history, future, self.features, [0, 1, 2]))
        graph = predict_graph(history, future, self.stations, self.adjacency, self.daily, 120, [0, 1])
        joint = assemble(reference, neural, tree, graph, future)["joint"]
        additive = predict_additive(history, future, cutoff)
        shape = joint + 0.25 * (additive - joint - by_station_day(future, additive - joint))
        bus_train, bus_future = history.copy(), future.copy()
        change_features = bus_movement(bus_train, bus_future)
        bus_tree = np.log1p(self.fit_tree(bus_train, bus_future, self.features + self.bus_features + change_features, [0]))
        has_bus = future["bus_stop_count"].ge(2).to_numpy()
        bus_level = joint + 0.5 * by_station_day(future, np.where(has_bus, bus_tree - tree, 0))
        is_corridor = future["station_id"].isin(CORRIDOR).to_numpy()
        origin = pd.Timestamp(cutoff)
        keys = future[KEY].assign(joint=joint)
        growth, _ = shifts(history, keys, self.controls, self.bus, self.calendar, origin)
        corridor = joint + 0.7 * growth
        flow = predict_mixture(history, future, self.stations, self.adjacency, self.mixture_daily, self.mixture_hourly, 6, 125, [0, 1])
        mixture = joint + 0.25 * (flow - joint)
        sports, is_sports = sports_forecasts(history, keys, self.calendar, self.sports, origin, self.threads)
        sporting = sports["football_and_baseball"]
        stack = choose_stack(joint, shape, bus_level, corridor, mixture, sporting, is_corridor, is_sports)
        predictions = {"team": reference, "joint": joint, "conditional_stack": stack}
        if with_missing_test:
            last_month = future["date"].max().to_period("M")
            withheld = future["date"].dt.to_period("M").eq(last_month).to_numpy()
            missing_bus = self.bus.copy()
            missing_bus.loc[last_month] = missing_bus.loc[last_month - 1].to_numpy()
            new_growth, _ = shifts(history, keys, self.controls, missing_bus, self.calendar, origin)
            predictions["withheld_bus_month"] = choose_stack(
                joint, shape, np.where(withheld, joint, bus_level), joint + 0.7 * new_growth, mixture, sporting, is_corridor, is_sports,
            )
        if not all(np.isfinite(values).all() for values in predictions.values()):
            raise ValueError(f"Nonfinite forecasts at cutoff {cutoff}.")
        return predictions


def station_bias(frame, column):
    error = np.maximum(frame[column], 0) - frame["y_log"]
    cells = frame[["fold", "station_id"]].assign(error=error)
    bias = cells.groupby(["fold", "station_id"])["error"].mean().groupby("station_id").mean()
    return bias - bias.mean()


def validate(model, results_path, predictions_path, candidate_key="conditional_stack", reference_key="joint"):
    results, predictions = {"folds": {}}, []
    for name, (cutoff, end) in WINDOWS.items():
        started = perf_counter()
        past = model.data.loc[model.data["date"].le(cutoff)]
        future = model.data.loc[model.data["date"].gt(cutoff) & model.data["date"].le(end)].assign(boardings=np.nan)
        expected = model.archive.loc[model.archive["fold"].eq(name)].set_index(KEY).reindex(pd.MultiIndex.from_frame(future[KEY]))
        actual, neural, original = (expected[key].to_numpy() for key in ("y_log", "nn_log", "ens_log"))
        output = model.predict(past, future, neural, original, cutoff, with_missing_test=True)
        scores = {key: rmsle_logs(actual, values) for key, values in output.items()}
        results["folds"][name] = {"rmsle": scores, "seconds": round(perf_counter() - started, 2)}
        predictions.append(future[KEY].assign(fold=name, y_log=actual, **output))
        results_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({name: results["folds"][name]}, allow_nan=False), flush=True)
    archive = pd.concat(predictions, ignore_index=True)
    for name, (cutoff, _) in WINDOWS.items():
        prior = archive.loc[archive["date"].le(cutoff) & archive["fold"].eq("C")]
        if prior.empty:
            continue
        target = archive.loc[archive["fold"].eq(name)]
        calibrated = {}
        for column in ("team", "joint", "conditional_stack", "withheld_bus_month"):
            adjustment = target["station_id"].map(station_bias(prior, column)).to_numpy()
            calibrated[column] = rmsle_logs(target["y_log"].to_numpy(), target[column].to_numpy() - adjustment)
        results["folds"][name]["station_calibrated_with_earlier_oof"] = calibrated
    results["acceptance"] = {column: sum(fold["rmsle"][reference_key] - fold["rmsle"][column] >= 0.003
                                         for fold in results["folds"].values())
                             for column in (candidate_key, "withheld_bus_month") if column != reference_key}
    archive.to_csv(predictions_path, index=False, float_format="%.9f")
    results_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"acceptance": results["acceptance"]}), flush=True)
    return results


def write_submission(args, model, results, candidate_key="conditional_stack"):
    if results["acceptance"][candidate_key] < 2:
        raise ValueError("Final candidate needs a 0.003 gain over the prior candidate in at least two windows.")
    history = model.data.loc[model.data["date"].le(FINAL_CUTOFF)]
    future = model.data.loc[model.data["date"].gt(FINAL_CUTOFF)]
    if len(future) != 157_194 or not future["example_id"].is_unique or future["boardings"].notna().any():
        raise ValueError("Final inputs must be the 157,194 unlabeled competition rows.")
    old = pd.read_csv(args.team_dir / "ensemble_parts.csv").set_index("example_id").reindex(future["example_id"])
    neural = old["log_nn"].to_numpy()
    reference = (old["log_lgb"].to_numpy() + neural) / 2
    estimate = model.predict(history, future, neural, reference, FINAL_CUTOFF)[candidate_key]
    ids = pd.read_csv(args.team_dir / "data/test.csv", usecols=["example_id"])["example_id"]
    submission = pd.Series(np.expm1(np.maximum(estimate, 0)), index=future["example_id"], name="boardings").reindex(ids).reset_index()
    if not submission["example_id"].equals(ids) or not np.isfinite(submission["boardings"]).all() or submission["boardings"].lt(0).any():
        raise ValueError("Submission must have finite nonnegative forecasts in test.csv order.")
    temporary = args.submission.with_suffix(".tmp")
    submission.to_csv(temporary, index=False)
    temporary.replace(args.submission)
    print(json.dumps({"submission": str(args.submission), "rows": len(submission)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team_dir", "external_zip", "scoreboard", "mobility_directory", "station_addresses", "bus", "football"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model = Candidate(args)
    results = validate(model, args.results, args.predictions)
    if args.submission:
        write_submission(args, model, results)
