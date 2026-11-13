"""Measure local football and baseball corrections over the joint team candidate."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_event_effects import read_events
from sincon_external_covariates import STADIUM
from sincon_football_experiment import STATION as FOOTBALL_STATION, match_predictions, read_matches
from sincon_station_state import KEY, WINDOWS, read_team
from sincon_team_game_exit import predict_game_rows, stadium_inputs


def sports_inputs(external_zip, baseball_csv, football_csv, calendar):
    regional_weather, baseball, scoreboard, baseball_dates = stadium_inputs(external_zip, baseball_csv, calendar)
    football = read_matches(football_csv)
    concert_dates = pd.DatetimeIndex(read_events(external_zip).loc[lambda frame: frame["station_id"].eq(FOOTBALL_STATION), "date"])
    return regional_weather, baseball, scoreboard, baseball_dates, football, concert_dates


def sports_forecasts(past, future, calendar, inputs, origin, threads):
    regional_weather, baseball, scoreboard, baseball_dates, football, concert_dates = inputs
    football_dates = football.index.difference(concert_dates)
    soccer = future["station_id"].eq(FOOTBALL_STATION) & future["date"].isin(football_dates)
    kbo = future["station_id"].eq(STADIUM) & future["date"].isin(baseball_dates)
    reference = future["joint"].to_numpy()
    estimates = {"football": reference.copy(), "baseball": reference.copy()}
    if soccer.any():
        football_source = past.loc[past["station_id"].eq(FOOTBALL_STATION), [*KEY, "boardings"]]
        football_target = future.loc[soccer, KEY + ["joint"]].rename(columns={"joint": "ens_log"})
        adjustment, _ = match_predictions(football_source, football_target, football, calendar, concert_dates)
        estimates["football"][soccer] = adjustment["spectator_fraction"]
    if kbo.any():
        eligible = past["station_id"].eq(STADIUM) & past["date"].isin(baseball_dates)
        source = past.loc[eligible, [*KEY, "boardings"]].assign(canonical_line=1)
        target = future.loc[kbo, KEY].assign(canonical_line=1)
        estimate = predict_game_rows(source, target, calendar, regional_weather, baseball, scoreboard, origin, threads)
        estimates["baseball"][kbo] = 0.75 * estimate["end_crowd_tree"] + 0.25 * reference[kbo]
    estimates["football_and_baseball"] = np.where(soccer, estimates["football"], estimates["baseball"])
    return estimates, (soccer | kbo).to_numpy()


def run(team_dir, external_zip, baseball_csv, football_csv, candidate_csv, threads, predictions=None):
    train, _, calendar, archive = read_team(team_dir, external_zip)
    inputs = sports_inputs(external_zip, baseball_csv, football_csv, calendar)
    joint = pd.read_csv(candidate_csv, usecols=["fold", *KEY, "joint"], parse_dates=["date"])
    scored = archive.merge(joint, on=["fold", *KEY], how="left", validate="one_to_one")
    if scored["joint"].isna().any():
        raise ValueError("Every fold needs the archived joint forecast.")
    report, parts = {}, []
    for fold, (cutoff, _) in WINDOWS.items():
        origin = pd.Timestamp(cutoff)
        past = train.loc[train["date"].le(origin)]
        valid = scored.loc[scored["fold"].eq(fold)]
        actual = valid["y_log"].to_numpy()
        reference = valid["joint"].to_numpy()
        sports, local = sports_forecasts(past, valid[[*KEY, "joint"]], calendar, inputs, origin, threads)
        estimates = {"team": valid["ens_log"].to_numpy(), "joint": reference, **sports}
        report[fold] = {name: float(np.sqrt(np.mean(np.square(actual - np.maximum(values, 0))))) for name, values in estimates.items()}
        parts.append(valid[KEY].assign(fold=fold, sports_day=local, **estimates))
    if predictions:
        pd.concat(parts, ignore_index=True).to_csv(predictions, index=False, float_format="%.7f")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team_dir", type=Path)
    parser.add_argument("external_zip", type=Path)
    parser.add_argument("baseball_csv", type=Path)
    parser.add_argument("football_csv", type=Path)
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    output = json.dumps(run(args.team_dir, args.external_zip, args.baseball_csv, args.football_csv,
                            args.candidate_csv, args.threads, args.predictions), indent=2, allow_nan=False) + "\n"
    if args.results:
        args.results.write_text(output, encoding="utf-8")
    print(output, end="")
