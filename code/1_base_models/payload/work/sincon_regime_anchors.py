"""Anchor immature stations and persistent station-level breaks to observed conditions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_housing_growth import forecast_changes, read_exposure
from sincon_joint_candidate import rmsle_logs
from sincon_station_state import EVENT_STATIONS, KEY, WINDOWS, read_team


EXCLUDE = ["is_holiday", "is_lunar_season", "is_bridge_day", "is_quasi_holiday", "is_pre_long_offday", "is_post_long_offday"]


def local_panel(frame, values, meta):
    daily = frame[[*KEY[:2], values]].groupby(KEY[:2])[values].mean().unstack("station_id").sort_index()
    centered = daily.copy()
    mature = meta["first_date"].eq(meta["first_date"].min()) & ~meta.index.isin(EVENT_STATIONS)
    for line, keys in meta.groupby("canonical_line").groups.items():
        station_ids = daily.columns.intersection(keys)
        controls = daily.columns.intersection(meta.index[mature & meta["canonical_line"].eq(line)])
        centered[station_ids] = daily[station_ids].sub(daily[controls].median(axis=1), axis=0)
    return centered


def classify_breaks(board, alight, cutoff, ordinary):
    dates = board.index
    # Detect changes on ordinary weekdays in the prior 18 weeks only. Both
    # directions must support the same sustained level shift.
    sample = dates.isin(ordinary) & (dates.dayofweek < 5) & (dates >= cutoff - pd.Timedelta(days=126))
    board, alight = board.loc[sample], alight.loc[sample]
    week = board.index.to_period("W")
    prior = np.stack([board.groupby(week).median(), alight.groupby(week).median()], axis=2)
    out = {}
    best = np.zeros(len(board.columns))
    for split in range(5, len(prior) - 3):
        before, after = prior[:split], prior[split:]
        old = np.nanmedian(before, axis=0)
        new = np.nanmedian(after, axis=0)
        move = new - old
        spread = np.nanmedian(np.abs(np.concatenate([before - old, after - new], axis=0)), axis=0)
        confidence = np.min(np.abs(move) / np.maximum(0.025, 1.4826 * spread), axis=1)
        agrees = (move[:, 0] * move[:, 1] > 0) & (np.min(np.abs(move), axis=1) >= .13)
        chosen = np.flatnonzero(agrees & (confidence >= 3.0) & (confidence > best))
        best[chosen] = confidence[chosen]
        weeks = board.groupby(week).size().index
        out.update({int(board.columns[i]): weeks[split].start_time for i in chosen})
    return out


def correction(past, keys, model, cutoff, meta, calendar, housing):
    if past["date"].max() > cutoff or keys["date"].min() <= cutoff or {"boardings", "y_log"} & set(keys):
        raise ValueError("Station states require pre-origin records and unlabeled forecast rows.")
    regular = calendar.index[calendar[EXCLUDE].eq(0).all(axis=1)]
    ordinary = past.loc[past["date"].isin(regular)]
    past_board = local_panel(ordinary.assign(log=np.log1p(ordinary["boardings"])), "log", meta)
    past_alight = local_panel(ordinary.assign(log=np.log1p(ordinary["alightings"])), "log", meta)
    breaks = classify_breaks(past_board, past_alight, cutoff, regular)
    early_days = (cutoff - meta["first_date"]).dt.days
    young = ((180 - early_days) / 90).clip(0, 1)
    future = keys.assign(base=model - 0.5 * housing)
    prediction = local_panel(future, "base", meta)
    observed_month = prediction.index.to_period("M")
    changed = np.zeros(len(keys))
    damped = np.zeros(len(keys))
    for station in prediction.columns:
        break_date = breaks.get(station)
        if break_date is None and not young.get(station, 0):
            continue
        actual_days = past_board.index >= cutoff - pd.Timedelta(days=56)
        if break_date is not None:
            actual_days &= past_board.index >= break_date
        for weekend in (False, True):
            past_days = actual_days & ((past_board.index.dayofweek >= 5) == weekend)
            predict_days = prediction.index.isin(regular) & ((prediction.index.dayofweek >= 5) == weekend)
            observed = past_board.loc[past_days, station].dropna()
            if len(observed) < (4 if weekend else 10):
                continue
            center = observed.median()
            forward = prediction.loc[predict_days, station].groupby(observed_month[predict_days]).median()
            change = np.clip(center - forward, -.38, .38)
            affected = keys["station_id"].eq(station) & keys["date"].isin(prediction.index[predict_days])
            values = keys.loc[affected, "date"].dt.to_period("M").map(change).fillna(0).to_numpy()
            if break_date is not None:
                changed[affected] = .8 * values
            damped[affected] = young.get(station, 0) * values
    return changed, damped, {str(station): str(date.date()) for station, date in breaks.items()}


def run(args):
    _, meta, calendar, _ = read_team(args.team_dir, args.external_zip)
    train = pd.read_csv(args.team_dir / "data/train.csv", parse_dates=["date"])
    archive = pd.read_csv(args.validation, parse_dates=["date"])
    exposure = read_exposure(args.housing, pd.read_csv(args.station_centers).set_index("station_id"))
    results, saved = {}, []
    for fold, (origin, _) in WINDOWS.items():
        cutoff = pd.Timestamp(origin)
        actual = archive.loc[archive["fold"].eq(fold)]
        keys = actual[KEY]
        base, true = (actual[name].to_numpy() for name in ("local_stack", "y_log"))
        history = train.loc[train["date"].le(cutoff)]
        housing, _, _ = forecast_changes(history, keys, cutoff, exposure, calendar, meta)
        break_shift, young_shift, breaks = correction(history, keys, base, cutoff, meta, calendar, housing)
        versions = {"reference": base, "detected_break": base + break_shift, "damp_young": base + young_shift,
                    "regime_anchors": base + np.where(break_shift != 0, break_shift, young_shift)}
        results[fold] = {"rmsle": {name: rmsle_logs(true, pred) for name, pred in versions.items()},
                         "detected_changes": breaks, "changed_rows": int(np.count_nonzero(versions["regime_anchors"] - base))}
        saved.append(keys.assign(fold=fold, y_log=true, **versions))
        print(json.dumps({fold: results[fold]}, allow_nan=False), flush=True)
    args.results.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.predictions:
        pd.concat(saved, ignore_index=True).to_csv(args.predictions, index=False, float_format="%.9f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team_dir", "external_zip", "validation", "housing", "station_centers"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    run(parser.parse_args())
