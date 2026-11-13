"""Estimate the station's football-day boarding increase from earlier matches."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_event_effects import read_events
from sincon_game_exit import clock_hour
from sincon_station_state import KEY, REGULAR, WINDOWS, read_team


STATION = 3754


def read_matches(path):
    matches = pd.read_csv(path, parse_dates=["date"])
    if matches["date"].duplicated().any() or matches["attendance"].isna().any() or matches["attendance"].le(0).any():
        raise ValueError("Expected one complete football record per match date.")
    return matches.assign(kickoff=clock_hour(matches["kickoff_kst"])).set_index("date")


def fit_examples(history, matches, calendar, event_dates):
    # A Saturday or Sunday can be a normal comparison day for a football match.
    special = ["is_holiday", "is_lunar_season", *REGULAR[1:]]
    regular = calendar.index[calendar[special].eq(0).all(axis=1)]
    normal = history.loc[history["date"].isin(regular) & ~history["date"].isin(matches.index.union(event_dates))]
    observed = normal.pivot(index="date", columns="hour", values="boardings")
    actual = history.loc[history["date"].isin(matches.index.difference(event_dates))].join(matches, on="date")
    actual = actual.assign(relative=actual["hour"] + 0.5 - actual["kickoff"])
    rows = []
    for date, match in actual.groupby("date"):
        gap = np.abs((observed.index - date).days)
        weekend = observed.index.dayofweek >= 5
        weekday = observed.index.dayofweek == date.dayofweek
        typical = observed.loc[(gap <= 75) & weekday]
        if len(typical) < 4:
            typical = observed.loc[(gap <= 75) & (weekend == (date.dayofweek >= 5))]
        if len(typical) < 4:
            continue
        baseline = typical.median().reindex(match["hour"]).to_numpy()
        count = match["boardings"].to_numpy()
        rows.append(match.assign(log_increase=np.log1p(count) - np.log1p(baseline),
                                 extra_share=(count - baseline) / match["attendance"].to_numpy()))
    if not rows:
        raise ValueError("No earlier matches had comparable non-match dates.")
    return pd.concat(rows, ignore_index=True)


def event_weights(target, training):
    rel = target["relative"].to_numpy()[:, None] - training["relative"].to_numpy()[None]
    start = target["kickoff"].to_numpy()[:, None] - training["kickoff"].to_numpy()[None]
    crowd = np.log(target["attendance"].to_numpy()[:, None] / training["attendance"].to_numpy()[None])
    weight = np.exp(-0.5 * (rel / 0.55) ** 2) * (0.6 + 0.4 * np.exp(-0.5 * (start / 1.8) ** 2))
    return weight * (0.65 + 0.35 * np.exp(-0.5 * (crowd / 0.5) ** 2))


def match_predictions(history, future, matches, calendar, event_dates):
    if history["date"].max() >= future["date"].min() or {"boardings", "y_log"} & set(future):
        raise ValueError("Match effects must fit on earlier observations and predict unlabeled rows.")
    effects = fit_examples(history, matches, calendar, event_dates)
    target = future.join(matches, on="date", validate="many_to_one")
    target = target.assign(relative=target["hour"] + 0.5 - target["kickoff"])
    valid = target["relative"].between(-2, 5)
    local = target.loc[valid]
    weight = event_weights(local, effects)
    weight /= weight.sum(axis=1, keepdims=True) + 0.8
    baseline = target["ens_log"].to_numpy()
    adjusted_log = baseline.copy()
    adjusted_share = baseline.copy()
    adjusted_log[valid] += 0.85 * np.maximum(weight @ effects["log_increase"].to_numpy(), 0)
    fraction = np.maximum(weight @ effects["extra_share"].to_numpy(), 0)
    adjusted_share[valid] = np.log1p(np.expm1(baseline[valid]) + 0.85 * local["attendance"].to_numpy() * fraction)
    return {"log_uplift": adjusted_log, "spectator_fraction": adjusted_share}, int(effects["date"].nunique())


def run(team_dir, external_zip, matches_csv):
    train, _, calendar, archive = read_team(team_dir, external_zip)
    matches = read_matches(matches_csv)
    event_days = read_events(external_zip).loc[lambda events: events["station_id"].eq(STATION), "date"]
    event_days = pd.DatetimeIndex(event_days)
    train = train.loc[train["station_id"].eq(STATION)]
    report = {}
    for fold, (cutoff, _) in WINDOWS.items():
        fitting = train.loc[train["date"].le(cutoff)]
        validation = archive.loc[archive["fold"].eq(fold)]
        mask = validation["date"].isin(matches.index.difference(event_days)) & validation["station_id"].eq(STATION)
        future = validation.loc[mask, KEY + ["ens_log"]]
        estimates, count = match_predictions(fitting, future, matches, calendar, event_days)
        reference = validation["ens_log"].to_numpy()
        actual = validation["y_log"].to_numpy()
        versions = {"ensemble": reference}
        for model, values in estimates.items():
            assembled = reference.copy()
            assembled[mask] = values
            versions[model] = assembled
        error = {name: np.square(actual - values) for name, values in versions.items()}
        report[fold] = {"fitted_matches": count, "scored_matches": int(future["date"].nunique()),
                        "rmsle": {name: float(np.sqrt(values.mean())) for name, values in error.items()},
                        "match_rows_rmsle": {name: float(np.sqrt(values[mask].mean())) for name, values in error.items()}}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team_dir", type=Path)
    parser.add_argument("external_zip", type=Path)
    parser.add_argument("matches_csv", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    output = json.dumps(run(args.team_dir, args.external_zip, args.matches_csv), indent=2, allow_nan=False) + "\n"
    if args.results:
        args.results.write_text(output, encoding="utf-8")
    print(output, end="")
