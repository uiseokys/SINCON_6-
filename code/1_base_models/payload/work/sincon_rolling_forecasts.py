"""Generate chronological forecasts for training a forecast correction model."""
import argparse
from datetime import date
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from sincon_joint_candidate import rmsle_logs
from sincon_local_candidate import LocalCandidate
from sincon_station_state import KEY


ORIGINS = {
    "2025-06-30": "2025-10-31",
    "2025-07-31": "2025-11-30",
    "2025-08-31": "2025-12-31",
    "2025-09-30": "2026-01-31",
}


def forecast_window(value):
    try:
        start, end = (date.fromisoformat(item) for item in value.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use YYYY-MM-DD:YYYY-MM-DD for each cutoff and end date.") from exc
    if start >= end:
        raise argparse.ArgumentTypeError("The forecast end must follow its training cutoff.")
    return start.isoformat(), end.isoformat()


def run(args):
    model = LocalCandidate(args)
    from ensemble import FEATURES
    from nn_model import fit_predict_nn

    args.directory.mkdir(parents=True, exist_ok=True)
    for cutoff, end in dict(args.window or ORIGINS.items()).items():
        path = args.directory / f"origin_{cutoff}.csv.gz"
        dates = pd.date_range(pd.Timestamp(cutoff) + pd.Timedelta(days=1), end)
        expected_rows = len(dates) * 71 * 18
        if path.is_file():
            archived = pd.read_csv(path, usecols=["date", "origin"], parse_dates=["date", "origin"])
            if (len(archived) != expected_rows or set(archived.date) != set(dates)
                    or not archived.origin.eq(pd.Timestamp(cutoff)).all()):
                raise ValueError(f"Existing archive does not match the requested window: {path}")
            continue
        started = perf_counter()
        history = model.data.loc[model.data["date"].le(cutoff)]
        held = model.data.loc[model.data["date"].gt(cutoff) & model.data["date"].le(end)]
        if (len(held) != expected_rows or set(held.date) != set(dates)
                or held.duplicated(KEY).any() or held.boardings.isna().any()):
            raise ValueError("Rolling validation needs a complete window of observed training-data dates.")
        target = held.assign(boardings=np.nan)
        actual = np.log1p(held["boardings"].to_numpy(dtype=float))
        neural = fit_predict_nn(history, target, FEATURES, [0], width=384, epochs=30, drop=0.15)
        print(f"Completed single historical MLP at {cutoff} in {perf_counter() - started:.0f}s", flush=True)
        # The reference parameter is carried through for reporting only.
        estimates = model.predict(history, target, neural, neural, cutoff)
        columns = {name: value for name, value in estimates.items() if name != "team"}
        archive = target[KEY].assign(origin=cutoff, y_log=actual, **columns)
        temporary = path.with_name(path.stem + ".tmp.csv.gz")
        archive.to_csv(temporary, index=False, float_format="%.9f")
        temporary.replace(path)
        print(json.dumps({"origin": cutoff, "end": end, "rows": len(archive),
                          "rmsle": {name: rmsle_logs(actual, value) for name, value in columns.items()},
                          "seconds": round(perf_counter() - started, 2)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team_dir", "external_zip", "scoreboard", "mobility_directory", "station_addresses", "bus", "football",
                 "events", "housing", "station_centers", "directory"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--window", action="append", type=forecast_window, help="Use a cutoff and end date, repeatable.")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run(args)
