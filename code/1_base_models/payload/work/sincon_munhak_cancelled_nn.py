"""Measure neural forecast changes when incorrect event flags are removed."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


STATION = 3127
CANCELLED = ("2026-09-26", "2026-09-27", "2026-10-24", "2026-10-25")
BEACH_DATES = tuple(f"2026-08-{day:02d}" for day in range(8, 16))
KEY = ["date", "station_id", "hour"]
LOCAL_EVENT = ["ev_today", "ev_is_concert", "ev_day_index", "ev_n_days", "ev_hours_from_start"]
SCENARIOS = ("original", "remove_station_event", "remove_station_and_city_event")


def target_scenarios(enriched, station=STATION, dates=CANCELLED, remove_city=True):
    rows = enriched.loc[enriched.date.isin(dates) & enriched.station_id.eq(station)].copy()
    if (len(rows) != 18 * len(dates) or not rows.example_id.is_unique or rows.duplicated(KEY).any()
            or not rows.ev_today.eq(1).all() or not rows.ev_city.gt(0).all()):
        raise ValueError("Expected an active event flag on every selected station-hour.")
    if remove_city and rows.mh_game_here.ne(0).any():
        raise ValueError("A cancelled concert overlaps a known home game; inspect the schedule.")
    no_local = rows.assign(ev_today=0, ev_is_concert=0, ev_day_index=np.nan,
                           ev_n_days=np.nan, ev_hours_from_start=np.nan)
    no_city = no_local.assign(ev_city=no_local.ev_city - int(remove_city))
    paired = pd.concat([frame.assign(scenario=label) for frame, label in zip((rows, no_local, no_city), SCENARIOS)],
                       ignore_index=True)
    stable = rows.columns.difference([*LOCAL_EVENT, "ev_city"])
    if not all(frame[stable].equals(rows[stable]) for frame in (no_local, no_city)):
        raise ValueError("Only the cancelled event inputs may change.")
    return rows, paired


def run(args):
    sys.path.insert(0, str(args.team.resolve()))
    from ensemble import FEATURES, NN_KW, build_calendar, enrich
    from nn_model import DEVICE, fit_predict_nn
    import torch

    torch.set_num_threads(args.threads)
    directory = args.team / "data"
    history = pd.read_csv(directory / "train.csv", usecols=[*KEY, "boardings"])
    grid = pd.read_csv(directory / "test.csv", usecols=[*KEY, "example_id"])
    meta = pd.read_csv(directory / "metaData.csv", usecols=["station_id", "canonical_line"]).rename(
        columns={"canonical_line": "line"})
    if history.date.max() != "2026-06-30" or grid.date.min() <= history.date.max() or len(grid) != 157_194:
        raise ValueError("Use official train counts only and preserve all official test keys.")
    calendar = build_calendar()
    history = enrich(history, calendar, meta, True)
    station, dates = (3139, BEACH_DATES) if args.songdo_beach else (STATION, CANCELLED)
    query = grid.loc[grid.date.isin(dates) & grid.station_id.eq(station)]
    rows, paired = target_scenarios(enrich(query, calendar, meta, True), station, dates, remove_city=not args.songdo_beach)
    if "boardings" in paired:
        raise ValueError("The inference scenarios may contain only unlabelled rows.")
    reference = pd.read_csv(args.team / "ensemble_parts.csv", usecols=["example_id", "log_nn"]).set_index("example_id")
    known = reference.log_nn.reindex(rows.example_id).to_numpy()
    if not np.isfinite(known).all():
        raise ValueError("The original 30-seed neural forecast must cover the cancellation keys.")
    forecasts = []
    for seed in args.seeds:
        started = perf_counter()
        log = fit_predict_nn(history, paired, FEATURES, [seed], **NN_KW).reshape(len(SCENARIOS), len(rows))
        if not np.isfinite(log).all():
            raise ValueError("Expected finite neural model output for all scenarios.")
        forecasts.append(log)
        change = log[2] - log[0]
        print(json.dumps({"seed": seed, "device": DEVICE, "seconds": round(perf_counter() - started, 2),
                          "delta_mean": float(change.mean()), "delta_minimum": float(change.min()),
                          "archived_uncancelled_rmse": float(np.sqrt(np.square(log[0] - known).mean()))}), flush=True)
    seed_forecasts = np.stack(forecasts)
    values = seed_forecasts.mean(axis=0)
    output = rows[[*KEY, "example_id", "ev_city"]].assign(archived_nn=known, original_nn=values[0],
                      station_delta=values[1] - values[0], combined_delta=values[2] - values[0])
    for i, seed in enumerate(args.seeds):
        output[f"delta_seed_{seed}"] = seed_forecasts[i, 2] - seed_forecasts[i, 0]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--songdo-beach", action="store_true", help="Keep the city event count and remove its wrong Songdo venue flag.")
    args = parser.parse_args()
    run(args).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
