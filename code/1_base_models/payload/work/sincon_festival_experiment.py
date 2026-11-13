"""Measure the effect of official event timing and size on local forecasts."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sincon_joint_candidate import rmsle_logs, setup
from sincon_station_state import KEY, WINDOWS


HOURS = np.arange(6, 24)
DAY = KEY[:2]
WEATHER = ["day_temp_max", "day_precip", "is_holiday"]
FEATURES = ["festival_exposure", "festival_size", "festival_count", "festival_last_day", "festival_timed",
            "festival_end", "festival_park", "festival_race", "festival_reduced"]


def local_days(path):
    events = pd.read_csv(path, parse_dates=["date", "event_start"])
    events["kind"] = events["category"]
    events.loc[events["series"].str.contains("blossom|wonmi_spring|incheon_citizen"), "kind"] = "공원"
    events["attend_day"] = events["previous_visitors"] / events["days"]
    # Co-located programs sharing attendance use a single maximum estimate.
    events = (events.sort_values(["weight", "attend_day"], ascending=False, na_position="last")
              .drop_duplicates([*DAY, "series"]).reset_index(drop=True))
    return events


def feature_table(events):
    size = np.log1p(events["attend_day"].fillna(0) * events["weight"])
    day = events.assign(festival_exposure=events["weight"], festival_size=size, festival_count=1,
                        festival_last_day=events["day_index"].eq(events["days"]) * events["weight"],
                        festival_park=events["kind"].eq("공원") * events["weight"],
                        festival_race=events["kind"].eq("마라톤") * events["weight"],
                        festival_reduced=events["status"].eq("reduced") * events["weight"])
    hourly = day.merge(pd.DataFrame({"hour": HOURS}), how="cross")
    since_start = hourly["hour"] + 0.5 - hourly["start_hour"]
    hourly["festival_timed"] = (np.exp(-0.5 * (since_start / 1.75) ** 2).fillna(0) * hourly["weight"])
    hourly["festival_end"] = (np.exp(-0.5 * ((since_start - hourly["duration_hours"]) / 1.75) ** 2).fillna(0) * hourly["weight"])
    rules = {name: ("sum" if name == "festival_count" else "max") for name in FEATURES}
    return hourly.groupby(KEY)[FEATURES].agg(rules)


def historical_effects(history, events):
    observed = history.pivot(index=DAY, columns="hour", values="boardings").reindex(columns=HOURS)
    conditions = history.drop_duplicates(DAY).set_index(DAY)[WEATHER]
    effects = []
    for station, group in events.groupby("station_id"):
        if station not in observed.index.get_level_values("station_id"):
            continue
        counts = observed.xs(station, level="station_id")
        external = conditions.xs(station, level="station_id")
        ordinary = counts.index.difference(group["date"])
        for event in group.itertuples(index=False):
            if event.date not in counts.index or event.weight < 0.3:
                continue
            condition = external.loc[event.date]
            distance = np.abs((ordinary - event.date).days.to_numpy(dtype=float))
            candidate = external.loc[ordinary]
            holiday = candidate["is_holiday"].to_numpy() > 0
            if condition["is_holiday"]:
                day_difference = (ordinary.dayofweek < 5) & ~holiday
            else:
                day_difference = (ordinary.dayofweek != event.date.dayofweek) | holiday
            temperature = np.abs(candidate["day_temp_max"].to_numpy() - condition["day_temp_max"]) / 8
            rain = np.abs(np.log1p(candidate["day_precip"].to_numpy()) - np.log1p(condition["day_precip"]))
            proximity = distance / 30 + temperature + rain + 4 * day_difference
            peers = ordinary[np.argsort(np.nan_to_num(proximity, nan=np.inf))[:12]]
            if len(peers) < 8:
                continue
            excess = counts.loc[event.date].to_numpy(dtype=float) - counts.loc[peers].median().to_numpy(dtype=float)
            smooth = np.convolve(np.pad(excess, 1, mode="edge"), [0.15, 0.7, 0.15], mode="valid")
            effects.append((event, np.maximum(smooth, 0)))
    return effects


def profile(event, history):
    exact = [(prior, hourly) for prior, hourly in history
             if prior.series == event.series and prior.station_id == event.station_id]
    local = [(prior, hourly) for prior, hourly in history
             if prior.kind == event.kind and prior.station_id == event.station_id]
    related = [(prior, hourly) for prior, hourly in history if prior.kind == event.kind and prior.weight >= 0.6]
    pool = exact or local or related
    if not pool:
        return np.zeros(len(HOURS))
    strength = 0.8 if exact else 0.45 if local else 0.25
    estimates = []
    weights = []
    for prior, hourly in pool:
        size = 1.0
        if np.isfinite(event.attend_day + prior.attend_day) and prior.attend_day > 0:
            size = np.clip(np.sqrt(event.attend_day / prior.attend_day), 0.5, 2)
        shift = event.start_hour - prior.start_hour
        if np.isfinite(shift):
            hourly = np.interp(HOURS - shift, HOURS, hourly, left=0, right=0)
        geographic = min(event.weight / prior.weight, 1.5)
        weights.append(1.0 if event.date.dayofweek == prior.date.dayofweek else 0.6)
        estimates.append(size * geographic * hourly)
    if event.status == "reduced":
        strength *= 0.5
    estimate = strength * np.average(estimates, weights=weights, axis=0)
    if event.kind == "마라톤" and np.isfinite(event.start_hour + event.duration_hours):
        # Road race effects around the finish are separate from evening games.
        estimate[(HOURS < event.start_hour - 1) | (HOURS > event.start_hour + event.duration_hours + 3)] = 0
    return estimate


def transferred(fit, target, events, base):
    before = events.loc[events["date"].le(fit["date"].max())]
    after = events.loc[events["date"].isin(target["date"])]
    source = historical_effects(fit, before)
    forecast = pd.Series(base, index=pd.MultiIndex.from_frame(target[KEY]))
    series = []
    for event in after.itertuples(index=False):
        index = pd.MultiIndex.from_product([[event.date], [event.station_id], HOURS], names=KEY)
        series.append(pd.Series(profile(event, source), index=index))
    adjustment = pd.concat(series).groupby(level=KEY).max().reindex(forecast.index, fill_value=0) if series else forecast * 0
    estimate = np.log1p(np.expm1(np.maximum(forecast, 0)) + adjustment)
    return estimate.to_numpy(), len(source)


def run(args):
    data, _, _, features, _, _, fit_tree = setup(args.team_dir, args.external_zip, args.scoreboard,
                                                args.mobility_directory, args.station_addresses, args.threads)
    events = local_days(args.events)
    data = data.join(feature_table(events), on=KEY, validate="one_to_one")
    data[FEATURES] = data[FEATURES].fillna(0)
    archive = pd.read_csv(args.validation, parse_dates=["date"])
    results, predictions = {}, []
    for fold, (cutoff, end) in WINDOWS.items():
        started = perf_counter()
        fit = data.loc[data["date"].le(cutoff)]
        target = data.loc[data["date"].gt(cutoff) & data["date"].le(end)].assign(boardings=np.nan)
        reference = archive.loc[archive["fold"].eq(fold)].set_index(KEY).reindex(pd.MultiIndex.from_frame(target[KEY]))
        actual, base = (reference[name].to_numpy() for name in ("y_log", "conditional_stack"))
        transferred_estimate, fitted_days = transferred(fit, target, events, base)
        is_event = target["festival_exposure"].gt(0).to_numpy()
        local = target.loc[is_event]
        zero = np.log1p(fit_tree(fit, local, features, [0]))
        added = np.log1p(fit_tree(fit, local, features + FEATURES, [0]))
        tree = base.copy()
        tree[is_event] += 0.75 * np.clip(added - zero, -0.35, 1.9)
        options = {"reference": base, "festival_tree": tree, "transferred_events": transferred_estimate,
                   "festival_combined": (transferred_estimate + tree) / 2}
        if not np.isfinite(np.column_stack([actual, *options.values()])).all():
            raise ValueError(f"Missing event forecasts for {fold}.")
        results[fold] = {"rmsle": {name: rmsle_logs(actual, value) for name, value in options.items()},
                         "event_rmsle": {name: rmsle_logs(actual[is_event], value[is_event]) for name, value in options.items()},
                         "event_rows": int(is_event.sum()), "historic_event_station_days": fitted_days,
                         "seconds": round(perf_counter() - started, 2)}
        predictions.append(target[KEY].assign(fold=fold, y_log=actual, is_event=is_event, **options))
        args.results.write_text(json.dumps(results, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({fold: results[fold]}, ensure_ascii=False), flush=True)
    if args.predictions:
        pd.concat(predictions, ignore_index=True).to_csv(args.predictions, index=False, float_format="%.8f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team_dir", "external_zip", "scoreboard", "mobility_directory", "station_addresses", "validation", "events"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    run(parser.parse_args())
