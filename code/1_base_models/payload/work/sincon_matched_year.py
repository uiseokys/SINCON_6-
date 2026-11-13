"""Transfer weather-adjusted station-hour profiles from the previous year."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sincon_conditional_candidate import by_station_day
from sincon_festival_experiment import local_days
from sincon_joint_candidate import rmsle_logs, setup
from sincon_station_state import EVENT_STATIONS, KEY, WINDOWS


WEATHER = ["temp_feel", "precip", "precip_prev3h", "day_precip", "snowfall", "wind", "is_snow_day", "school_vac_share"]
CONTROL = ["is_holiday", "is_lunar_block", "is_bridge_day", "is_semi_holiday", "is_before_long_off", "is_after_long_off"]
HOURS = range(6, 24)


def grid(panel, days, stations, columns):
    index = pd.MultiIndex.from_product([days, stations, HOURS], names=KEY)
    values = panel.set_index(KEY)[columns].reindex(index).to_numpy(dtype=float)
    return values.reshape(len(days), len(stations), len(HOURS), len(columns)), index


def response_coefficients(weather, logs, regular, mature, station_lines):
    # Compare nearby dates with the same weekday. Most growth and stable
    # station-hour effects cancel before estimating weather sensitivity.
    differences, responses = [], []
    for lag in (7, 14, 21, 28):
        same_type = regular[lag:] & regular[:-lag]
        features = (weather[lag:] - weather[:-lag])[same_type]
        changes = (logs[lag:] - logs[:-lag])[same_type]
        differences.append(features)
        responses.append(changes)
    design = np.concatenate(differences)
    target = np.concatenate(responses).clip(-0.5, 0.5)
    usable = np.isfinite(target) & np.isfinite(design).all(axis=3)
    design = np.where(usable[..., None], design, 0)
    target = np.where(usable, target, 0)
    grams = np.einsum("dshi,dshj->shij", design, design, optimize=True)
    cross = np.einsum("dshi,dsh->shi", design, target, optimize=True)
    width = design.shape[-1]
    coefficients = np.zeros_like(cross)
    for line in np.unique(station_lines):
        group = station_lines == line
        strong = group & mature
        common = np.linalg.solve(grams[strong].sum(axis=0) + np.eye(width) * 350,
                                 cross[strong].sum(axis=0)[..., None]).squeeze(-1)
        ridge = 220
        coefficients[group] = np.linalg.solve(grams[group] + np.eye(width) * ridge,
                                               (cross[group] + ridge * common[None])[..., None]).squeeze(-1)
    return coefficients


def transfer(weather, logs, regular, target_idx, reference_before, beta, span=28):
    weights = np.zeros((len(target_idx), *logs.shape[1:]))
    outcomes = np.zeros_like(weights)
    coverage = np.zeros_like(weights)
    for shift in range(-span, span + 1, 7):
        previous = target_idx - 364 + shift
        permitted = (previous >= 0) & (previous <= reference_before)
        past = previous.clip(0, len(logs) - 1)
        delta = (weather[target_idx] - weather[past]).clip(-4, 4)
        usable = (permitted & regular[past])[:, None, None] & np.isfinite(logs[past])
        adjustment = np.einsum("dshi,shi->dsh", delta, beta, optimize=True).clip(-.30, .30)
        # Keep the seasonal window tight and favor similar demand conditions.
        dissimilar = np.mean(np.square(delta), axis=-1)
        similarity = np.exp(-abs(shift) / 28 - dissimilar / 2)
        weight = np.where(usable, similarity, 0)
        weights += weight
        outcomes += weight * np.nan_to_num(logs[past] + adjustment)
        coverage += usable
    return np.divide(outcomes, weights, out=np.full_like(outcomes, np.nan), where=(weights > .07) & (coverage >= 3))


def predict(history, future, stations, origin, match_winter=False):
    if history["date"].max() > origin or future["date"].min() <= origin or future["boardings"].notna().any():
        raise ValueError("Annual analogs must use pre-origin ridership observations.")
    days = pd.date_range(history["date"].min(), future["date"].max())
    panel = pd.concat([history, future], ignore_index=True)
    window, index = grid(panel, days, stations.index, ["boardings", *WEATHER])
    logs, weather = np.log1p(window[..., 0]), window[..., 1:]
    for i in (1, 2, 3, 4):
        weather[..., i] = np.log1p(np.maximum(weather[..., i], 0))
    prior = int(np.searchsorted(days, origin, side="right"))
    center = np.nanmedian(weather[:prior], axis=(0, 1, 2))
    scale = np.maximum(np.nanstd(weather[:prior], axis=(0, 1, 2)), 0.01)
    weather = np.nan_to_num((weather - center) / scale)
    regular = (panel.drop_duplicates("date").set_index("date").reindex(days)[CONTROL].eq(0).all(axis=1).to_numpy())
    mature = stations["first_date"].eq(stations["first_date"].min()).to_numpy()
    beta = response_coefficients(weather[:prior], logs[:prior], regular[:prior], mature, stations["canonical_line"].to_numpy())
    future_idx = np.arange(prior, len(days))
    analog = transfer(weather, logs, regular, future_idx, prior - 1, beta)
    pairs = np.arange(max(366, prior - 56), prior)
    if len(pairs) >= 7:
        matched = transfer(weather, logs, regular, pairs, prior - 1, beta)
        adjustment = np.where(regular[pairs, None, None], logs[pairs] - matched, np.nan)
        available = np.isfinite(adjustment).sum(axis=0)
        growth = np.ma.median(np.ma.masked_invalid(adjustment), axis=0).filled(0)
        growth = growth * available / (available + 10)
        growth = 0.6 * growth + 0.4 * growth.mean(axis=1, keepdims=True)
        analog += growth[None].clip(-0.35, 0.35)
    elif match_winter:
        # A December origin can have comparable weather in the preceding
        # winter even when a full 364-day labeled pair does not yet exist.
        pairs = np.arange(max(290, prior - 56), prior)
        matched = transfer(weather, logs, regular, pairs, prior - 1, beta, span=77)
        difference = np.where(regular[pairs, None, None], logs[pairs] - matched, np.nan)
        available = np.isfinite(difference).sum(axis=0)
        growth = np.ma.median(np.ma.masked_invalid(difference), axis=0).filled(0)
        strong = mature & ~stations.index.isin(EVENT_STATIONS)
        pooled = np.median(growth[strong], axis=0)
        growth = (growth * available + 16 * pooled[None]) / (available + 16)
        # Without a full-year overlap use only the persistent station-wide
        # part of the change, blended toward the system change.
        growth = 0.25 * growth + 0.75 * growth.mean(axis=1, keepdims=True)
        analog += growth[None].clip(-0.18, 0.18)
    eligible_station = mature & ~stations.index.isin(EVENT_STATIONS)
    analog = np.where(regular[future_idx, None, None] & eligible_station[None, :, None], analog, np.nan)
    return pd.Series(analog.ravel(), index=index[prior * len(stations) * len(HOURS):]).reindex(pd.MultiIndex.from_frame(future[KEY]))


def run(args):
    data, _, _, _, stations, _, _ = setup(args.team_dir, args.external_zip, args.scoreboard,
                                           args.mobility_directory, args.station_addresses, 1)
    archive = pd.read_csv(args.validation, parse_dates=["date"])
    festivals = local_days(args.events)
    event_keys = pd.MultiIndex.from_frame(festivals[KEY[:2]].drop_duplicates())
    results, outputs = {}, []
    for fold, (cutoff, end) in WINDOWS.items():
        started = perf_counter()
        origin = pd.Timestamp(cutoff)
        history = data.loc[data["date"].le(origin)]
        future = data.loc[data["date"].gt(origin) & data["date"].le(end)].assign(boardings=np.nan)
        previous = archive.loc[archive["fold"].eq(fold)].set_index(KEY).reindex(pd.MultiIndex.from_frame(future[KEY]))
        truth, base = (previous[name].to_numpy() for name in ("y_log", "local_stack"))
        estimate = predict(history, future, stations, origin).to_numpy()
        valid = np.isfinite(estimate) & ~pd.MultiIndex.from_frame(future[KEY[:2]]).isin(event_keys)
        difference = np.where(valid, estimate - base, 0)
        level = by_station_day(future, difference)
        options = {"reference": base, "annual_blend": base + .35 * difference, "annual_level": base + .35 * level,
                   "annual_shape": base + .35 * (difference - level)}
        if args.match_winter:
            estimate = predict(history, future, stations, origin, match_winter=True).to_numpy()
            available = np.isfinite(estimate) & ~pd.MultiIndex.from_frame(future[KEY[:2]]).isin(event_keys)
            adjustment = np.where(available, estimate - base, 0)
            options["matched_winter"] = base + .35 * adjustment
            options["matched_winter_level"] = base + .35 * by_station_day(future, adjustment)
        if not np.isfinite(np.column_stack([truth, *options.values()])).all():
            raise ValueError(f"Incomplete forecasts in {fold}.")
        results[fold] = {"rmsle": {name: rmsle_logs(truth, values) for name, values in options.items()},
                         "matched_rows": int(valid.sum()), "seconds": round(perf_counter() - started, 2)}
        outputs.append(future[KEY].assign(fold=fold, y_log=truth, **options))
        args.results.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({fold: results[fold]}, allow_nan=False), flush=True)
    if args.predictions:
        pd.concat(outputs, ignore_index=True).to_csv(args.predictions, index=False, float_format="%.9f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team_dir", "external_zip", "scoreboard", "mobility_directory", "station_addresses", "validation", "events"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--match-winter", action="store_true")
    parser.add_argument("--predictions", type=Path)
    run(parser.parse_args())
