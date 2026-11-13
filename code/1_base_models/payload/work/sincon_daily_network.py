"""Jointly predict the 18 hourly log boardings at each station on a day."""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from torch import nn

from sincon_game_exit import clock_hour
from sincon_station_state import KEY, WINDOWS, read_team


HOURS = 18
HOURLY = ["temp_feel", "humidity", "precip", "snowfall", "wind", "precip_prev1h", "precip_prev3h",
          "mh_hours_from_start", "ev_hours_from_start", "mh_hours_from_end"]
FIRST_HOUR = 6
STATION_DAY = ["date", "station_id"]


class DailyNetwork(nn.Module):
    def __init__(self, station_count, n_daily, n_hourly, width, architecture):
        super().__init__()
        self.architecture = architecture
        self.station = nn.Embedding(station_count, 32)
        self.weekday = nn.Embedding(7, 8)
        self.hour = nn.Embedding(HOURS, 8)
        self.common = nn.Sequential(nn.Linear(40 + n_daily, width), nn.SiLU(), nn.Dropout(0.10),
                                    nn.Linear(width, width), nn.SiLU(), nn.Dropout(0.10))
        self.level = nn.Linear(width, 1)
        self.shape = nn.Linear(width, HOURS)
        if architecture == "hour_local":
            self.weather = nn.Sequential(nn.Linear(width + 8 + n_hourly, 48), nn.SiLU(), nn.Linear(48, 1))
        else:
            self.weather = nn.Sequential(nn.Linear(width + HOURS * n_hourly, width), nn.SiLU(), nn.Linear(width, HOURS))
        self.register_buffer("hour_ids", torch.arange(HOURS))

    def forward(self, station, weekday, daily, hourly):
        hidden = self.common(torch.cat([self.station(station), self.weekday(weekday), daily], dim=1))
        if self.architecture == "hour_local":
            context = hidden[:, None].expand(-1, HOURS, -1)
            hour = self.hour(self.hour_ids)[None].expand(len(station), -1, -1)
            effects = self.weather(torch.cat([context, hour, hourly], dim=2)).squeeze(2)
        else:
            effects = self.weather(torch.cat([hidden, hourly.flatten(1)], dim=1))
        shape = self.shape(hidden) + effects
        return self.level(hidden) + shape - shape.mean(dim=1, keepdim=True)


def prepare_data(team_dir, external_zip, scores_path):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(team_dir.resolve()))
    try:
        from ensemble import FEATURES, WEATHER_FEATURES, build_calendar, enrich
    finally:
        sys.path.pop(0)

    train, _, _, validation = read_team(team_dir, external_zip)
    metadata = pd.read_csv(team_dir / "data/metaData.csv", usecols=["station_id", "canonical_line"]).rename(columns={"canonical_line": "line"})
    test = pd.read_csv(team_dir / "data/test.csv")
    train = train.assign(date=train["date"].dt.strftime("%Y-%m-%d"))
    combined = enrich(pd.concat([train, test], ignore_index=True), build_calendar(), metadata, True)
    with ZipFile(external_zip) as archive:
        weather = pd.read_csv(archive.open("external/weather_hourly.csv"), usecols=["weather_zone", "date", "hour", *WEATHER_FEATURES])
        zones = pd.read_csv(archive.open("external/weather_zone_map.csv"))
    combined = (combined.drop(columns=WEATHER_FEATURES).merge(zones, on="station_id", validate="many_to_one")
                .merge(weather, on=["weather_zone", "date", "hour"], how="left", validate="many_to_one"))
    if combined[WEATHER_FEATURES].isna().any().any():
        raise ValueError("Regional weather must cover every station-hour.")
    scores = pd.read_csv(scores_path, usecols=["date", "start_kst", "end_kst", "attendance"])
    starts = clock_hour(scores["start_kst"])
    ends = clock_hour(scores["end_kst"])
    scores["end"] = ends.where(ends.ge(starts), ends + 24)
    scoreboard = scores.groupby("date").agg(end=("end", "max"), audience=("attendance", "sum"))
    local = combined["station_id"].eq(3127)
    combined["mh_hours_from_end"] = (combined["hour"] + 0.5 - combined["date"].map(scoreboard["end"])).where(local)
    combined["mh_audience_here"] = np.log1p(combined["date"].map(scoreboard["audience"])).where(local)
    combined["date"] = pd.to_datetime(combined["date"])
    angle = 2 * np.pi * combined["date"].dt.dayofyear.to_numpy() / 365.2425
    combined["annual_sin"] = np.sin(angle)
    combined["annual_cos"] = np.cos(angle)
    daily = [col for col in FEATURES if col not in WEATHER_FEATURES + HOURLY + ["station_id", "hour", "dow", "mh_att_log"]]
    daily += [col for col in WEATHER_FEATURES if col.startswith(("day_", "is_")) and col != "is_raining"]
    daily += ["annual_sin", "annual_cos", "mh_audience_here"]
    return combined.sort_values(KEY).reset_index(drop=True), validation, daily


def tensor_data(frame, daily_features, station_ids):
    hours = frame["hour"].to_numpy().reshape(-1, HOURS)
    days = frame.iloc[::HOURS]
    if not (hours == np.arange(FIRST_HOUR, FIRST_HOUR + HOURS)).all() or frame.duplicated(KEY).any():
        raise ValueError("Expected a complete grid of ordered hourly rows for each station-day.")
    station = days["station_id"].map(station_ids).to_numpy(dtype="int64")
    weekday = days["dow"].to_numpy(dtype="int64")
    daily = days[daily_features].to_numpy(dtype="float32", copy=True)
    hourly = frame[HOURLY].to_numpy(dtype="float32", copy=True).reshape(-1, HOURS, len(HOURLY))
    return station, weekday, daily, hourly


def normalize(fit, future, axes):
    missing = np.isnan(fit)
    mean = np.nanmean(fit, axis=axes)
    scale = np.nanstd(fit, axis=axes)
    mean = np.nan_to_num(mean)
    scale = np.maximum(np.nan_to_num(scale), 0.01)
    train_mask = missing.any(axis=axes)
    train = np.clip((fit - mean) / scale, -8, 8)
    observed_low = np.min(np.where(missing, np.inf, train), axis=axes)
    observed_high = np.max(np.where(missing, -np.inf, train), axis=axes)
    observed_low = np.where(np.isfinite(observed_low), observed_low, 0)
    observed_high = np.where(np.isfinite(observed_high), observed_high, 0)
    test = np.clip((future - mean) / scale, observed_low, observed_high)
    return tuple(np.concatenate([np.nan_to_num(x), np.isnan(x)[..., train_mask].astype(np.float32)], axis=-1)
                 for x in (train, test))


def predict(fit, future, daily_features, architecture, epochs, seeds):
    if fit["date"].max() >= future["date"].min() or future["boardings"].notna().any():
        raise ValueError("Prediction rows must follow training and have no supplied labels.")
    stations = {station: i for i, station in enumerate(sorted(fit["station_id"].unique()))}
    fs, fw, fd, fh = tensor_data(fit, daily_features, stations)
    ts, tw, td, th = tensor_data(future, daily_features, stations)
    fd, td = normalize(fd, td, (0,))
    fh, th = normalize(fh, th, (0, 1))
    fit_inputs = [torch.from_numpy(a) for a in (fs, fw, fd, fh)]
    test_inputs = [torch.from_numpy(a) for a in (ts, tw, td, th)]
    labels = np.log1p(fit["boardings"].to_numpy(dtype="float32")).reshape(-1, HOURS)
    mean, std = float(labels.mean()), float(labels.std())
    labels = torch.from_numpy((labels - mean) / std)
    predictions = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = DailyNetwork(len(stations), fd.shape[-1], fh.shape[-1], 160, architecture)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.005)
        batches = (len(labels) + 255) // 256
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.003, total_steps=epochs * batches, pct_start=0.1)
        for _ in range(epochs):
            model.train()
            for rows in torch.randperm(len(labels)).split(256):
                optimizer.zero_grad(set_to_none=True)
                residual = model(*(part[rows] for part in fit_inputs)) - labels[rows]
                residual.square().mean().backward()
                optimizer.step()
                scheduler.step()
        model.eval()
        with torch.no_grad():
            logs = torch.cat([model(*(part[i:i + 512] for part in test_inputs))
                              for i in range(0, len(ts), 512)]).numpy().ravel() * std + mean
        predictions.append(logs)
    return np.maximum(np.mean(predictions, axis=0), 0)


def run(team_dir, external_zip, scores_path, epochs, seeds, architectures, output):
    combined, validation, daily_features = prepare_data(team_dir, external_zip, scores_path)
    report = {}
    for name, (cutoff, end) in WINDOWS.items():
        fit = combined.loc[combined["date"].le(cutoff)]
        future = combined.loc[combined["date"].gt(cutoff) & combined["date"].le(end)].assign(boardings=np.nan)
        reference = validation.loc[validation["fold"].eq(name)].set_index(KEY).reindex(pd.MultiIndex.from_frame(future[KEY]))
        actual = reference["y_log"].to_numpy()
        ensemble = reference["ens_log"].to_numpy()
        if not np.isfinite(np.column_stack([actual, ensemble])).all():
            raise ValueError(f"Archived ensemble is incomplete for fold {name}.")
        report[name] = {"ensemble": float(np.sqrt(np.mean(np.square(actual - ensemble))))}
        for architecture in architectures:
            started = perf_counter()
            logs = predict(fit, future, daily_features, architecture, epochs, seeds)
            scores = {"alone": logs, "blend_25": 0.75 * ensemble + 0.25 * logs, "blend_50": (ensemble + logs) / 2}
            report[name][architecture] = {key: float(np.sqrt(np.mean(np.square(actual - value)))) for key, value in scores.items()}
            report[name][architecture]["seconds"] = round(perf_counter() - started, 2)
            if output:
                output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(json.dumps({name: {architecture: report[name][architecture]}}, allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team_dir", type=Path)
    parser.add_argument("external_zip", type=Path)
    parser.add_argument("scores_path", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--architecture", nargs="+", choices=["hour_local", "flattened"], default=["hour_local", "flattened"])
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output = run(args.team_dir, args.external_zip, args.scores_path, args.epochs, range(args.seeds), args.architecture, args.results)
    print(json.dumps(output, indent=2, allow_nan=False))
