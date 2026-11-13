"""Fit hour-specific additive models with seasonal pooling across stations."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from sincon_joint_candidate import rmsle_logs, setup
from sincon_station_state import KEY, WINDOWS


def indicator(series):
    code, categories = pd.factorize(series, sort=True)
    return sparse.csr_matrix((np.ones(len(code)), (np.arange(len(code)), code)), shape=(len(code), len(categories)))


def standardized(values, fit):
    mean = np.nan_to_num(np.nanmean(values[fit], axis=0))
    scale = np.maximum(np.nan_to_num(np.nanstd(values[fit], axis=0)), 0.05)
    normalized = np.nan_to_num((values - mean) / scale)
    return np.clip(normalized, -5, 5)


def interactions(groups, values):
    groups = groups.tocoo()
    width = values.shape[1]
    columns = groups.col[:, None] * width + np.arange(width)[None]
    rows = np.repeat(groups.row, width)
    out = sparse.csr_matrix((values[groups.row].ravel(), (rows, columns.ravel())),
                            shape=(groups.shape[0], groups.shape[1] * width))
    out.eliminate_zeros()
    return out


def feature_blocks(panel, is_fit, cutoff):
    station = indicator(panel["station_id"])
    station_weekday = indicator(pd.MultiIndex.from_frame(panel[["station_id", "dow"]]))
    line = indicator(panel["line"])
    weekend_type = indicator(panel["dow"].clip(upper=5))
    day = panel["date"].dt.dayofyear.to_numpy()
    phase = 2 * np.pi * day / 365.2425
    annual = np.column_stack([np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)])
    since_start = (panel["date"].clip(upper=pd.Timestamp(cutoff)) - pd.Timestamp("2025-01-01")).dt.days.to_numpy()
    # The local level is flat after cutoff. Season terms are free to continue
    # periodically, with stronger shrinkage on station-specific deviations.
    trend = np.column_stack([since_start / 365, *[np.maximum(since_start - day, 0) / 365 for day in (90, 180, 270, 365)],
                             (since_start >= 178).astype(float), panel["months_after_extension"].to_numpy()])
    calendar_names = ["is_holiday", "is_lunar_day", "is_semi_holiday", "is_weekday_holiday", "is_lunar_block",
                      "is_bridge_day", "is_before_long_off", "is_after_long_off", "is_school_vacation",
                      "is_univ_vacation", "is_summer_peak"]
    lunar = panel["lunar_rel_day"].to_numpy()
    calendar = np.column_stack([panel[calendar_names].to_numpy(), *[(lunar == k) for k in range(-3, 4)]])
    weather = np.column_stack([panel[name].to_numpy() for name in (
        "is_raining", "is_rainy_day", "is_snow_day", "is_heatwave_day", "is_coldwave_day")])
    nonlinear = np.column_stack([np.log1p(panel[name].clip(lower=0).to_numpy()) for name in (
        "precip", "precip_prev1h", "precip_prev3h", "day_precip", "snowfall")])
    temperature = panel["temp_feel"].to_numpy()
    weather = np.column_stack([weather, nonlinear, np.maximum(temperature - 25, 0) / 8,
                               np.maximum(5 - temperature, 0) / 10, panel["wind"].to_numpy() / 5])
    external_names = [name for name in panel if name.startswith(("arrivals_", "daily_arrivals_", "hourly_share_"))]
    local = panel[["station_id", "dow", *external_names]]
    level = local.loc[is_fit].groupby(["station_id", "dow"])[external_names].mean()
    base = level.reindex(pd.MultiIndex.from_frame(local[["station_id", "dow"]])).to_numpy()
    mobility = standardized(local[external_names].to_numpy() - base, is_fit)
    weather = standardized(weather, is_fit)
    games = panel["mh_hours_from_end"].to_numpy()
    event_terms = np.column_stack([np.nan_to_num(np.exp(-np.square((games - shift) / 0.75)))
                                  for shift in (-3, -1, 0, 1, 2)])
    event_terms = np.column_stack([event_terms, panel[["ev_hours_from_start", "mh_hours_from_start"]].notna().to_numpy()])
    global_design = np.column_stack([annual, trend, calendar, weather, mobility, event_terms])
    return [
        (station, 0.001), (station_weekday, 0.2), (sparse.csr_matrix(global_design), 4),
        (interactions(weekend_type, np.column_stack([annual, calendar, weather, mobility])), 36),
        (interactions(line, np.column_stack([annual, trend, calendar, weather, mobility])), 40),
        (interactions(station, annual), 120), (interactions(station, trend), 35),
        (interactions(station, calendar), 18), (interactions(station, weather), 65),
        (interactions(station, mobility), 140),
    ]


def predict(past, future, cutoff):
    if past["date"].max() >= future["date"].min() or future["boardings"].notna().any():
        raise ValueError("Use only earlier ridership and unlabeled forecast rows.")
    result = pd.Series(index=future.index, dtype=float)
    for hour in range(6, 24):
        fit = past.loc[past["hour"].eq(hour)]
        test = future.loc[future["hour"].eq(hour)]
        panel = pd.concat([fit, test], ignore_index=True)
        is_fit = np.arange(len(panel)) < len(fit)
        blocks = feature_blocks(panel, is_fit, cutoff)
        design = sparse.hstack([block for block, _ in blocks], format="csr")
        penalty = sparse.diags(np.concatenate([np.full(block.shape[1], strength) for block, strength in blocks]), format="csc")
        x = design[is_fit]
        y = np.log1p(fit["boardings"].to_numpy(dtype=float))
        coefficients = spsolve((x.T @ x).tocsc() + penalty, x.T @ y)
        result.loc[test.index] = np.maximum(design[~is_fit] @ coefficients, 0)
    return result.to_numpy()


def run(args):
    data, _, _, _, _, _, _ = setup(args.team_dir, args.external_zip, args.scoreboard,
                                    args.mobility_directory, args.station_addresses, 1)
    archived = pd.read_csv(args.validation, parse_dates=["date"])
    report, parts = {}, []
    for fold, (cutoff, end) in WINDOWS.items():
        started = perf_counter()
        past = data.loc[data["date"].le(cutoff)]
        future = data.loc[data["date"].gt(cutoff) & data["date"].le(end)].assign(boardings=np.nan)
        previous = archived.loc[archived["fold"].eq(fold)].set_index(KEY).reindex(pd.MultiIndex.from_frame(future[KEY]))
        actual, joint = (previous[key].to_numpy() for key in ("y_log", "joint"))
        estimate = predict(past, future, cutoff)
        if not np.isfinite(np.column_stack([actual, joint, estimate])).all():
            raise ValueError(f"Incomplete forecast in {fold}.")
        difference = estimate - joint
        level = future[KEY[:2]].assign(change=difference).groupby(KEY[:2])["change"].transform("mean").to_numpy()
        versions = {"joint": joint, "additive": estimate, "blend": joint + 0.25 * difference,
                    "blend_shape": joint + 0.25 * (difference - level), "blend_level": joint + 0.25 * level}
        report[fold] = {"rmsle": {name: rmsle_logs(actual, values) for name, values in versions.items()},
                        "seconds": round(perf_counter() - started, 2)}
        parts.append(future[KEY].assign(fold=fold, **versions))
        args.results.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({fold: report[fold]}, allow_nan=False), flush=True)
    if args.predictions:
        pd.concat(parts, ignore_index=True).to_csv(args.predictions, index=False, float_format="%.7f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team_dir", "external_zip", "scoreboard", "mobility_directory", "station_addresses", "validation"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    run(parser.parse_args())
