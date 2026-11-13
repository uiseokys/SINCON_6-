"""Train the team's LightGBM and MLP and save only uncalibrated forecasts."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from sincon_station_state import KEY, WINDOWS


def run(team, output, threads):
    sys.path.insert(0, str(team.resolve()))
    from ensemble import build_calendar, enrich, predict_both
    from improved import PARAMS

    torch.set_num_threads(threads)
    PARAMS["num_threads"] = threads
    train = pd.read_csv(team / "data/train.csv", usecols=[*KEY, "boardings"])
    test = pd.read_csv(team / "data/test.csv", usecols=[*KEY, "example_id"])
    if (len(train) != 688_176 or train.date.max() != "2026-06-30" or len(test) != 157_194
            or train.date.max() >= test.date.min() or not test.example_id.is_unique):
        raise ValueError("Use the official historical and prediction grids.")
    metadata = pd.read_csv(team / "data/metaData.csv", usecols=["station_id", "canonical_line"])
    metadata = metadata.rename(columns={"canonical_line": "line"})
    calendar = build_calendar()
    train, test = (enrich(rows, calendar, metadata, True) for rows in (train, test))
    output.mkdir(parents=True, exist_ok=True)
    paths = output / "valid_predictions_ABC.csv.gz", output / "ensemble_parts.csv"
    if any(path.exists() for path in paths):
        raise FileExistsError("Choose a fresh output directory.")
    validation = []
    for label, (cutoff, end) in WINDOWS.items():
        fit = train.loc[train.date.le(cutoff)]
        held = train.loc[train.date.gt(cutoff) & train.date.le(end)]
        lgb, nn = predict_both(fit, held.drop(columns="boardings"))
        validation.append(held[KEY].assign(fold=label, train_end=cutoff, y_log=np.log1p(held.boardings),
                                           lgb_log=lgb, nn_log=nn, ens_log=(lgb + nn) / 2.))
        print(f"{label} {len(held)} uncalibrated predictions", flush=True)
    pd.concat(validation, ignore_index=True).to_csv(paths[0], index=False)
    lgb, nn = predict_both(train, test)
    test[["example_id"]].assign(log_lgb=lgb, log_nn=nn).to_csv(paths[1], index=False)
    print(f"final {len(test)} uncalibrated predictions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    run(args.team, args.output, args.threads)
