"""Rebuild the current candidates at the strict July–October 2025 origin."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from sincon_joint_candidate import rmsle_logs
from sincon_local_candidate import LocalCandidate
from sincon_station_state import KEY


ORIGIN = "2025-06-30"
END = "2025-10-31"


def run(args):
    started = perf_counter()
    torch.set_num_threads(args.threads)
    candidate = LocalCandidate(args)
    from ensemble import FEATURES, LGB_SEEDS, NN_KW, build_calendar, enrich
    from nn_model import fit_predict_nn

    raw = pd.read_csv(args.team_dir / "data/train.csv", usecols=[*KEY, "boardings"])
    raw = raw.loc[raw["date"].le(END)].sort_values(KEY)
    metadata = pd.read_csv(args.team_dir / "data/metaData.csv", usecols=["station_id", "canonical_line"]).rename(
        columns={"canonical_line": "line"})
    panel = enrich(raw, build_calendar(), metadata, True)
    history = panel.loc[panel["date"].le(ORIGIN)]
    target = panel.loc[panel["date"].gt(ORIGIN)].drop(columns="boardings")
    if len(target) != 157_194:
        raise ValueError("Strict summer replay requires every target station, date and hour.")
    per_seed = []
    for seed in range(args.seeds):
        values = fit_predict_nn(history, target, FEATURES, [seed], **NN_KW)
        per_seed.append(values)
        print(f"Strict summer team neural seed {seed + 1}/{args.seeds}, {perf_counter() - started:.0f}s", flush=True)
    neural = np.mean(per_seed, axis=0)
    tree = np.log1p(candidate.fit_tree(history, target, FEATURES, LGB_SEEDS))
    original = target[KEY].assign(neural=neural, team=(tree + neural) / 2)
    original["date"] = pd.to_datetime(original["date"])
    preceding = candidate.data.loc[candidate.data["date"].le(ORIGIN)]
    held = candidate.data.loc[candidate.data["date"].gt(ORIGIN) & candidate.data["date"].le(END)]
    known = held[KEY].merge(original, on=KEY, how="left", validate="one_to_one")
    if known[["neural", "team"]].isna().any().any():
        raise ValueError("The team and candidate forecasts need identical summer prediction keys.")
    truth = np.log1p(held["boardings"].to_numpy(dtype=float))
    outputs = candidate.predict(preceding, held.assign(boardings=np.nan), known["neural"].to_numpy(), known["team"].to_numpy(), ORIGIN)
    mature = held["station_id"].map(candidate.meta["first_date"]).eq(candidate.meta["first_date"].min()).to_numpy()
    results = {"origin": ORIGIN, "end": END, "rows": len(held), "mlp_seeds": args.seeds,
               "rmsle_all": {name: rmsle_logs(truth, values) for name, values in outputs.items()},
               "rmsle_mature_68": {name: rmsle_logs(truth[mature], values[mature]) for name, values in outputs.items()},
               "rmsle_new_3": {name: rmsle_logs(truth[~mature], values[~mature]) for name, values in outputs.items()},
               "seconds": round(perf_counter() - started, 2)}
    args.results.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    held[KEY].assign(y_log=truth, **outputs).to_csv(args.predictions, index=False, float_format="%.9f")
    print(json.dumps(results, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team_dir", "external_zip", "scoreboard", "mobility_directory", "station_addresses", "bus", "football",
                 "events", "housing", "station_centers"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    run(parser.parse_args())
