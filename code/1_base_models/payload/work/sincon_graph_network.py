"""Forecast every station together with message passing along the rail network."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn

from sincon_daily_network import FIRST_HOUR, HOURLY, HOURS, normalize, prepare_data
from sincon_station_state import KEY, WINDOWS


class GraphNetwork(nn.Module):
    def __init__(self, adjacency, daily_features, hourly_features):
        super().__init__()
        n_stations = len(adjacency)
        width = 80
        self.register_buffer("adjacency", torch.from_numpy(adjacency))
        self.register_buffer("station_ids", torch.arange(n_stations))
        self.station = nn.Embedding(n_stations, 24)
        self.weekday = nn.Embedding(7, 8)
        self.base = nn.Embedding(n_stations, HOURS)
        self.by_weekday = nn.Embedding(n_stations * 7, HOURS)
        nn.init.zeros_(self.base.weight)
        nn.init.zeros_(self.by_weekday.weight)
        self.context = nn.Sequential(nn.Linear(daily_features + 32, width), nn.SiLU(), nn.Dropout(0.08))
        self.mix = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Dropout(0.08))
        self.level = nn.Linear(width, 1)
        self.shape = nn.Sequential(nn.Linear(width + HOURS * hourly_features, width), nn.SiLU(),
                                   nn.Linear(width, HOURS))

    def forward(self, weekday, daily, hourly):
        b, n, _ = daily.shape
        station = self.station(self.station_ids)[None].expand(b, -1, -1)
        day = self.weekday(weekday)[:, None].expand(-1, n, -1)
        local = self.context(torch.cat([station, day, daily], dim=-1))
        nearby = torch.matmul(self.adjacency, local)
        hidden = self.mix(torch.cat([local, nearby], dim=-1))
        weekly = self.by_weekday(weekday[:, None] * n + self.station_ids[None])
        shape = self.shape(torch.cat([hidden, hourly.flatten(2)], dim=-1))
        return self.base(self.station_ids)[None] + weekly + self.level(hidden) + shape - shape.mean(dim=-1, keepdim=True)


def adjacency_matrix(stations):
    ids = stations.index
    weights = pd.DataFrame(0., index=ids, columns=ids)
    for _, group in stations.groupby("canonical_line"):
        line = group.index.sort_values()
        for left, right in zip(line[:-1], line[1:]):
            weights.at[left, right] = weights.at[right, left] = 1
    names = stations["station_name"].str.replace(r"\(.*?\)", "", regex=True).str.strip()
    for _, group in names.groupby(names):
        for key in group.index:
            weights.loc[key, group.index.drop(key)] = 0.5
    return weights.div(weights.sum(axis=1), axis=0).fillna(0).to_numpy(dtype="float32")


def panel(frame, stations, daily_features, hourly_features=HOURLY):
    dates = pd.DatetimeIndex(frame["date"].unique()).sort_values()
    n_days, n_stations = len(dates), len(stations)
    keys = pd.MultiIndex.from_product([dates, stations, range(FIRST_HOUR, FIRST_HOUR + HOURS)], names=KEY)
    ordered = frame.set_index(KEY)
    daily = ordered.xs(FIRST_HOUR, level="hour")[daily_features]
    daily_grid = keys.droplevel("hour").unique()
    daily = daily.reindex(daily_grid).to_numpy(dtype="float32").reshape(n_days, n_stations, -1)
    hourly = ordered[hourly_features].reindex(keys).to_numpy(dtype="float32").reshape(n_days, n_stations, HOURS, -1)
    labels = ordered["boardings"].reindex(keys).to_numpy(dtype="float32").reshape(n_days, n_stations, HOURS)
    return keys, dates.dayofweek.to_numpy(dtype="int64"), daily, hourly, labels


def predict(fit, future, stations, adjacency, daily_features, epochs, seeds):
    if fit["date"].max() >= future["date"].min() or future["boardings"].notna().any():
        raise ValueError("Use earlier ridership for training and unlabeled forecast inputs.")
    _, fw, fd, fh, labels = panel(fit, stations, daily_features)
    keys, tw, td, th, _ = panel(future, stations, daily_features)
    fd, td = normalize(fd, td, (0, 1))
    fh, th = normalize(fh, th, (0, 1, 2))
    inputs = [torch.from_numpy(a) for a in (fw, fd, fh)]
    forecasts = [torch.from_numpy(a) for a in (tw, td, th)]
    labels = np.log1p(labels)
    mean, scale = float(np.nanmean(labels)), float(np.nanstd(labels))
    observed = torch.from_numpy(np.isfinite(labels))
    actual = torch.from_numpy(np.nan_to_num((labels - mean) / scale))
    parts = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = GraphNetwork(adjacency, fd.shape[-1], fh.shape[-1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.008)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.004, epochs=epochs,
                                                      steps_per_epoch=(len(actual) + 15) // 16, pct_start=0.12)
        for _ in range(epochs):
            model.train()
            for rows in torch.randperm(len(actual)).split(16):
                optimizer.zero_grad(set_to_none=True)
                valid = observed[rows]
                difference = model(*(part[rows] for part in inputs))[valid] - actual[rows][valid]
                difference.square().mean().backward()
                optimizer.step()
                schedule.step()
        model.eval()
        with torch.no_grad():
            log_values = torch.cat([model(*(part[i:i + 24] for part in forecasts))
                                    for i in range(0, len(tw), 24)]).numpy().ravel()
        parts.append(np.maximum(mean + scale * log_values, 0))
    return pd.Series(np.mean(parts, axis=0), index=keys).reindex(pd.MultiIndex.from_frame(future[KEY])).to_numpy()


def graph_features(features, daily, stations):
    # Use periodic season terms and keep recent drift separate from the shared day representation.
    daily = [name for name in daily if name not in {"month", "season", "days_since_start"}]
    opened = features["station_id"].map(pd.to_datetime(stations["first_date"]))
    extension = opened.gt(opened.min())
    features = features.assign(
        months_after_extension=np.where(extension, np.log1p((features["date"] - opened).dt.days.clip(lower=0) / 30), 0),
        recent_system_state=np.log1p((features["date"] - features["date"].min()).dt.days / 90),
    )
    daily += ["months_after_extension", "recent_system_state"]
    return features, daily


def run(team_dir, external_zip, scores_path, epochs, seeds, output):
    features, validation, daily = prepare_data(team_dir, external_zip, scores_path)
    stations = pd.read_csv(team_dir / "data/metaData.csv").set_index("station_id").sort_index()
    adjacency = adjacency_matrix(stations)
    features, daily = graph_features(features, daily, stations)
    report = {}
    for name, (cutoff, end) in WINDOWS.items():
        started = perf_counter()
        fit = features.loc[features["date"].le(cutoff)]
        future = features.loc[features["date"].gt(cutoff) & features["date"].le(end)].assign(boardings=np.nan)
        archive = validation.loc[validation["fold"].eq(name)].set_index(KEY).reindex(pd.MultiIndex.from_frame(future[KEY]))
        model = predict(fit, future, stations.index, adjacency, daily, epochs, seeds)
        team = archive["ens_log"].to_numpy()
        actual = archive["y_log"].to_numpy()
        if not np.isfinite(np.column_stack([model, team, actual])).all():
            raise ValueError(f"Incomplete forecasts in {name}.")
        change = model - team
        level_change = pd.DataFrame({"date": future["date"], "station_id": future["station_id"], "change": change})
        level_change = level_change.groupby(["date", "station_id"])["change"].transform("mean").to_numpy()
        candidates = {"ensemble": team, "graph": model, "blend": team + 0.25 * change,
                      "blend_level_only": team + 0.25 * level_change,
                      "blend_shape_only": team + 0.25 * (change - level_change)}
        report[name] = {"rmsle": {key: float(np.sqrt(np.mean(np.square(actual - np.maximum(prediction, 0)))))
                                  for key, prediction in candidates.items()},
                        "seconds": round(perf_counter() - started, 2)}
        if output:
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({name: report[name]}, allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team_dir", type=Path)
    parser.add_argument("external_zip", type=Path)
    parser.add_argument("scores_path", type=Path)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run(args.team_dir, args.external_zip, args.scores_path, args.epochs, range(args.seeds), args.results)
