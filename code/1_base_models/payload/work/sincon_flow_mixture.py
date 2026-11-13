"""Decompose boarding counts into nonnegative station-hour demand components."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.functional import softplus

from sincon_daily_network import HOURLY, HOURS, normalize
from sincon_graph_network import panel
from sincon_joint_candidate import rmsle_logs, setup
from sincon_station_state import KEY, WINDOWS


class FlowMixture(nn.Module):
    def __init__(self, adjacency, n_daily, n_hourly, components, start):
        super().__init__()
        stations = len(adjacency)
        self.register_buffer("adjacency", torch.from_numpy(adjacency))
        self.register_buffer("station_ids", torch.arange(stations))
        self.register_buffer("hours", torch.arange(HOURS))
        self.station = nn.Embedding(stations, 24)
        self.weekday = nn.Embedding(7, 8)
        self.hour = nn.Embedding(HOURS, 8)
        self.log_profile = nn.Parameter(torch.empty(stations, HOURS, components).normal_(start - np.log(components), 0.2))
        self.weekly_shape = nn.Parameter(torch.zeros(7, HOURS, components))
        self.station_hour = nn.Parameter(torch.zeros(stations, 7, HOURS))
        self.encode = nn.Sequential(nn.Linear(n_daily + n_hourly + 32, 72), nn.SiLU(), nn.Dropout(0.10))
        self.mix = nn.Sequential(nn.Linear(144, 96), nn.SiLU(), nn.Dropout(0.12), nn.Linear(96, 64), nn.SiLU())
        self.activity = nn.Linear(64, components)
        self.conditional = nn.Sequential(nn.Linear(72 + n_hourly, 48), nn.SiLU(), nn.Linear(48, components))

    def forward(self, weekday, daily, hourly):
        b, n, _, _ = hourly.shape
        station = self.station(self.station_ids)[None].expand(b, -1, -1)
        day = self.weekday(weekday)[:, None].expand(-1, n, -1)
        local = self.encode(torch.cat([station, day, daily, hourly.mean(dim=2)], dim=2))
        context = self.mix(torch.cat([local, self.adjacency @ local], dim=2))
        hour = self.hour(self.hours)[None, None].expand(b, n, -1, -1)
        replicated = context[:, :, None].expand(-1, -1, HOURS, -1)
        weather = self.conditional(torch.cat([replicated, hour, hourly], dim=3))
        terms = (self.log_profile[None] + self.weekly_shape[weekday, None] + self.activity(context)[:, :, None] + weather)
        correction = self.station_hour[self.station_ids[None], weekday[:, None]]
        return softplus(terms.logsumexp(dim=3) + correction)


def predict(history, future, stations, adjacency, daily, hourly, components, epochs, seeds):
    if history["date"].max() >= future["date"].min() or future["boardings"].notna().any():
        raise ValueError("The mixture uses only pre-origin ridership labels.")
    _, fw, fd, fh, observed_counts = panel(history, stations, daily, hourly)
    keys, tw, td, th, _ = panel(future, stations, daily, hourly)
    fd, td = normalize(fd, td, (0, 1))
    fh, th = normalize(fh, th, (0, 1, 2))
    fit = [torch.from_numpy(x) for x in (fw, fd, fh)]
    target = [torch.from_numpy(x) for x in (tw, td, th)]
    observed = torch.from_numpy(np.isfinite(observed_counts))
    values = np.log1p(observed_counts)
    labels = torch.from_numpy(np.nan_to_num(values))
    estimates = []
    for seed in seeds:
        torch.manual_seed(seed)
        network = FlowMixture(adjacency, fd.shape[-1], fh.shape[-1], components, float(np.nanmean(values)))
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.004, weight_decay=0.004)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.004, epochs=epochs,
                                                      steps_per_epoch=(len(labels) + 15) // 16, pct_start=0.12)
        for _ in range(epochs):
            network.train()
            for rows in torch.randperm(len(labels)).split(16):
                optimizer.zero_grad(set_to_none=True)
                keep = observed[rows]
                error = network(*(x[rows] for x in fit))[keep] - labels[rows][keep]
                error.square().mean().backward()
                optimizer.step()
                schedule.step()
        network.eval()
        with torch.no_grad():
            estimate = torch.cat([network(*(x[i:i + 24] for x in target)) for i in range(0, len(tw), 24)]).numpy().ravel()
        estimates.append(estimate)
    return pd.Series(np.mean(estimates, axis=0), index=keys).reindex(pd.MultiIndex.from_frame(future[KEY])).to_numpy()


def run(args):
    data, _, daily, tree_features, stations, adjacency, _ = setup(args.team_dir, args.external_zip, args.scoreboard,
                                                                args.mobility_directory, args.station_addresses, args.threads)
    daily = daily + [name for name in tree_features if name.startswith("daily_arrivals_")]
    hourly = HOURLY + [name for name in tree_features if name.startswith(("arrivals_", "hourly_share_"))]
    archive = pd.read_csv(args.validation, parse_dates=["date"])
    results, saved = {}, []
    for fold, (cutoff, end) in WINDOWS.items():
        started = perf_counter()
        fit = data.loc[data["date"].le(cutoff)]
        future = data.loc[data["date"].gt(cutoff) & data["date"].le(end)].assign(boardings=np.nan)
        base = archive.loc[archive["fold"].eq(fold)].set_index(KEY).reindex(pd.MultiIndex.from_frame(future[KEY]))
        truth, joint = (base[name].to_numpy() for name in ("y_log", "joint"))
        raw = predict(fit, future, stations.index, adjacency, daily, hourly, args.components, args.epochs, list(range(args.seeds)))
        if not np.isfinite(np.column_stack([truth, joint, raw])).all():
            raise ValueError(f"Incomplete mixture forecasts in {fold}.")
        diff = raw - joint
        level = future[KEY[:2]].assign(change=diff).groupby(KEY[:2])["change"].transform("mean").to_numpy()
        versions = {"joint": joint, "mixture": raw, "blend": joint + 0.25 * diff,
                    "blend_shape": joint + 0.25 * (diff - level), "blend_level": joint + 0.25 * level}
        results[fold] = {"rmsle": {name: rmsle_logs(truth, values) for name, values in versions.items()},
                         "seconds": round(perf_counter() - started, 2)}
        saved.append(future[KEY].assign(fold=fold, **versions))
        args.results.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({fold: results[fold]}, allow_nan=False), flush=True)
    if args.predictions:
        pd.concat(saved, ignore_index=True).to_csv(args.predictions, index=False, float_format="%.7f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team_dir", "external_zip", "scoreboard", "mobility_directory", "station_addresses", "validation"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--components", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=125)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run(args)
