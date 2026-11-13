"""Measure the interaction between local events, station growth, and hourly shape."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_joint_candidate import rmsle_logs
from sincon_station_state import KEY, WINDOWS
from sincon_corridor_bus_growth import CORRIDOR


JOIN = ["fold", *KEY]


def attach(archive, path, columns):
    candidate = pd.read_csv(path, usecols=[*JOIN, "joint", *columns], parse_dates=["date"])
    candidate = candidate.rename(columns={"joint": "reference"})
    joined = archive.merge(candidate, on=JOIN, how="left", validate="one_to_one")
    if not np.allclose(joined.pop("reference"), joined["joint"], atol=1.01e-7, rtol=0):
        raise ValueError(f"The reference forecast or validation rows differ in {path.name}.")
    return joined


def interval(frame, actual, reference, estimate):
    errors = frame[["date"]].assign(reference=(np.maximum(reference, 0) - actual) ** 2,
                                    estimate=(np.maximum(estimate, 0) - actual) ** 2)
    # Resample seven-day blocks of paired squared errors. A date's 1,278
    # station-hour errors are kept together.
    days = errors.groupby("date")[["reference", "estimate"]].mean().to_numpy()
    rng = np.random.default_rng(20261112)
    block = 7
    offsets = np.arange(block)
    origins = rng.integers(0, len(days), size=(2000, (len(days) + block - 1) // block, 1))
    samples = ((origins + offsets) % len(days)).reshape(len(origins), -1)[:, :len(days)]
    paired = np.sqrt(days[samples].mean(axis=1))
    improvement = paired[:, 0] - paired[:, 1]
    return np.quantile(improvement, [0.025, 0.975]).tolist()


def run(directory):
    archive = pd.read_csv(directory / "sincon_joint_valid_predictions.csv.gz", usecols=[*JOIN, "y_log", "ensemble", "joint"], parse_dates=["date"])
    for name, columns in (
        ("additive", ["blend_shape"]), ("bus_model", ["bus_level_only", "bus_without_final_month"]),
        ("flow_mixture", ["blend"]), ("sports", ["football_and_baseball", "sports_day"]),
        ("corridor_bus_growth", ["bus_growth"]),
    ):
        archive = attach(archive, directory / f"sincon_{name}_validation.csv.gz", columns)
    results = {}
    for fold in WINDOWS:
        selected = archive.loc[archive["fold"].eq(fold)]
        actual, original, joint, sports, shape, bus, no_month, mixture, corridor = (
            selected[name].to_numpy() for name in ("y_log", "ensemble", "joint", "football_and_baseball", "blend_shape",
                                                  "bus_level_only", "bus_without_final_month", "blend", "bus_growth")
        )
        local = selected["sports_day"].to_numpy(dtype=bool)
        in_corridor = selected["station_id"].isin(CORRIDOR).to_numpy()
        general = shape + bus + mixture - 2 * joint
        conditional = shape + np.where(in_corridor, corridor, bus) + mixture - 2 * joint
        versions = {"original": original, "joint": joint, "sports": sports, "general": general,
                    "general_and_sports": np.where(local, sports, general),
                    "general_and_sports_bus_last_month_hidden": np.where(local, sports, shape + no_month + mixture - 2 * joint),
                    "conditional_and_sports": np.where(local, sports, conditional)}
        results[fold] = {"rmsle": {name: rmsle_logs(actual, prediction) for name, prediction in versions.items()},
                         "improvement_95pct_week_bootstrap": {name: interval(selected, actual, joint, values)
                                                              for name, values in versions.items() if name != "joint"}}
    gates = {name: sum(fold["rmsle"]["joint"] - fold["rmsle"][name] >= 0.003 for fold in results.values())
             for name in versions if name not in {"joint", "original"}}
    return {"folds": results, "folds_at_or_above_0.003_improvement_from_joint": gates,
            "status": "exploratory combinations selected using the existing A/B/C results"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    output = json.dumps(run(args.directory), indent=2, allow_nan=False) + "\n"
    args.results.write_text(output, encoding="utf-8")
    print(output, end="")
