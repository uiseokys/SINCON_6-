"""Measure game-finish modeling against the team's archived ensemble predictions."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

import sincon_calendar_kernel as kernel
from sincon_event_effects import read_events
from sincon_external_covariates import HOURS, RAIN, STADIUM, summarize_games
from sincon_game_exit import exit_kernel, exit_tree, game_inputs, read_scoreboard
from sincon_lightgbm_experiments import HOLIDAY_FLAGS, attach_calendar, holiday_snapshot
from sincon_station_state import KEY, WINDOWS, read_team


def stadium_inputs(external_zip, scores_csv, calendar):
    with ZipFile(external_zip) as archive:
        games = pd.read_csv(archive.open("external/munhak_games.csv"), parse_dates=["game_date"])
        weather = pd.read_csv(archive.open("external/weather_hourly.csv"), parse_dates=["date"],
                              usecols=["weather_zone", "date", "hour", *RAIN])
        station_zone = pd.read_csv(archive.open("external/weather_zone_map.csv")).set_index("station_id")
    zone = station_zone.at[STADIUM, "weather_zone"]
    weather = weather.loc[weather["weather_zone"].eq(zone) & weather["hour"].isin(HOURS)].drop(columns="weather_zone").set_index(["date", "hour"])
    dates = pd.DatetimeIndex(calendar.index, name="date")
    game_days = summarize_games(games, dates)
    scoreboard = read_scoreboard(scores_csv, game_days)
    concerts = read_events(external_zip).loc[lambda data: data["station_id"].eq(STADIUM), "date"]
    usable_days = scoreboard.index.difference(concerts)
    return weather, game_days, scoreboard, usable_days


def predict_game_rows(history, target, calendar, weather, game_days, scoreboard, origin, threads):
    snapshot = holiday_snapshot(calendar[HOLIDAY_FLAGS], origin)
    dated = []
    for frame in (history, target):
        frame = game_inputs(attach_calendar(frame, snapshot, origin), scoreboard, game_days, weather)
        dated.append(frame.merge(kernel.dates_and_regimes(frame)[["date", "regime"]], on="date", validate="many_to_one"))
    estimates = exit_kernel(*dated)
    estimates["end_crowd_tree"] = exit_tree(*dated, threads)
    return estimates


def run(team_dir, external_zip, scores_csv, threads):
    train, _, calendar, validation = read_team(team_dir, external_zip)
    weather, game_days, scoreboard, usable_days = stadium_inputs(external_zip, scores_csv, calendar)
    history = train.loc[train["station_id"].eq(STADIUM) & train["date"].isin(usable_days)].copy()
    history["canonical_line"] = 1
    report = {}
    for name, (cutoff, _) in WINDOWS.items():
        origin = pd.Timestamp(cutoff)
        fold = validation.loc[validation["fold"].eq(name)]
        train_window = history.loc[history["date"].le(origin)]
        route = fold["station_id"].eq(STADIUM) & fold["date"].isin(usable_days)
        target = fold.loc[route, KEY].assign(canonical_line=1)
        reference = fold["ens_log"].to_numpy()
        outputs = {"ensemble": reference}
        if not target.empty:
            estimates = predict_game_rows(train_window, target, calendar, weather, game_days, scoreboard, origin, threads)
            for model in ("end_kernel", "end_crowd_kernel", "end_crowd_tree"):
                forecast = reference.copy()
                forecast[route] = 0.75 * estimates[model] + 0.25 * reference[route]
                outputs[model] = forecast
        actual = fold["y_log"].to_numpy()
        error = {model: np.square(actual - np.maximum(value, 0)) for model, value in outputs.items()}
        report[name] = {"fit_games": int(train_window["date"].nunique()), "validation_games": int(target["date"].nunique()),
                        "rmsle": {model: float(np.sqrt(values.mean())) for model, values in error.items()},
                        "game_rows_rmsle": {model: float(np.sqrt(values[route].mean())) for model, values in error.items()} if route.any() else {}}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team_dir", type=Path)
    parser.add_argument("external_zip", type=Path)
    parser.add_argument("scores_csv", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    output = json.dumps(run(args.team_dir, args.external_zip, args.scores_csv, args.threads), indent=2, allow_nan=False) + "\n"
    if args.results:
        args.results.write_text(output, encoding="utf-8")
    print(output, end="")
