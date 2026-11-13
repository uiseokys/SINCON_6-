"""Evaluate a synthetic-pretrained transformer for fixed-origin station day levels."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tabicl import TabICLRegressor

from sincon_joint_candidate import rmsle_logs, setup
from sincon_station_state import KEY, WINDOWS


FEATURES = ["dow", "month", "is_holiday", "is_offday", "is_bridge_day", "is_pre_long_offday", "is_post_long_offday",
            "is_quasi_holiday", "is_univ_vacation", "school_vac_share", "school_discretionary_share", "is_peak_summer",
            "day_temp_max", "day_temp_min", "day_precip", "is_snow_day", "annual_sin", "annual_cos", "days_since_start",
            "mh_game", "mh_audience_here", "ev_count", "daily_arrivals_regional_bus", "daily_arrivals_local_bus",
            "daily_arrivals_walking", "daily_arrivals_car"]


def station_days(data):
    # All predictions are directly conditioned on known date, weather and
    # non-subway covariates. The labels enter only in the pre-origin context.
    names = [name for name in FEATURES if name in data]
    mobility = [name for name in data if name.startswith("daily_arrivals_")]
    names = list(dict.fromkeys([*names, *mobility]))
    table = data[KEY + names + ["boardings"]].copy()
    table["daily_target"] = np.log1p(table.pop("boardings"))
    table["band"] = (table["hour"] - 6) // 6
    daily = table.drop_duplicates(KEY[:2]).set_index(KEY[:2])[names]
    return table, daily


def forecast(table, daily, cutoff, last, station_ids, model, bands):
    daily_targets = table.groupby(KEY[:2] + (["band"] if bands == 3 else []))["daily_target"].mean()
    outputs = []
    for station in station_ids:
        keys = daily.xs(station, level="station_id")
        train_keys = keys.loc[:cutoff]
        future_keys = keys.loc[(keys.index > cutoff) & (keys.index <= last)]
        for band in range(bands):
            truth = daily_targets.xs(station, level="station_id")
            if bands == 3:
                truth = truth.xs(band, level="band")
            y = truth.reindex(train_keys.index)
            if train_keys.index.max() > cutoff or future_keys.index.min() <= cutoff or y.isna().any():
                raise ValueError("Prediction requires distinct pre-origin labels and future covariates.")
            x = train_keys.loc[:, train_keys.nunique(dropna=True).gt(1)]
            model.fit(x, y)
            pred = model.predict(future_keys[x.columns])
            outputs.append(pd.DataFrame({"date": future_keys.index, "station_id": station, "band": band, "tabicl": pred}))
    return pd.concat(outputs, ignore_index=True)


def run(args):
    data, _, _, _, stations, _, _ = setup(args.team_dir, args.external_zip, args.scoreboard,
                                            args.mobility_directory, args.station_addresses, args.threads)
    table, daily = station_days(data)
    reference = pd.read_csv(args.validation, parse_dates=["date"])
    base_column = "local_stack" if "local_stack" in reference else "conditional_stack"
    station_ids = stations.index if not args.stations else args.stations
    model = TabICLRegressor(model_path=args.checkpoint, allow_auto_download=False, n_estimators=args.estimators,
                            device="cpu", n_jobs=args.threads, random_state=2026)
    result, outputs = {}, []
    for fold, (cutoff, end) in WINDOWS.items():
        started = perf_counter()
        origin, last = pd.Timestamp(cutoff), pd.Timestamp(end)
        forecasts = forecast(table, daily, origin, last, station_ids, model, args.bands)
        old = reference.loc[reference["fold"].eq(fold)].copy()
        old["band"] = (old["hour"] - 6) // 6 if args.bands == 3 else 0
        joined = old.merge(forecasts, on=[*KEY[:2], "band"], how="left", validate="many_to_one")
        base = joined[base_column].to_numpy()
        true = joined["y_log"].to_numpy()
        old_level = joined.groupby([*KEY[:2], "band"])[base_column].transform("mean")
        delta = joined["tabicl"] - old_level
        covered = delta.notna().to_numpy()
        if not covered.any() or not np.isfinite(delta[covered]).all():
            raise ValueError(f"Missing TabICL predictions in {fold}.")
        variants = {"reference": base, "tabicl_all": base + delta.fillna(0).to_numpy(),
                    "tabicl_blend": base + 0.25 * delta.fillna(0).to_numpy()}
        result[fold] = {"rmsle": {key: rmsle_logs(true, pred) for key, pred in variants.items()},
                        "covered_rmsle": {key: rmsle_logs(true[covered], pred[covered]) for key, pred in variants.items()},
                        "covered_rows": int(covered.sum()), "seconds": round(perf_counter() - started, 2)}
        outputs.append(joined[KEY].assign(fold=fold, y_log=true, **variants))
        args.results.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({fold: result[fold]}, allow_nan=False), flush=True)
    if args.predictions:
        pd.concat(outputs, ignore_index=True).to_csv(args.predictions, index=False, float_format="%.9f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team_dir", "external_zip", "scoreboard", "mobility_directory", "station_addresses", "validation", "checkpoint"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--bands", choices=[1, 3], type=int, default=1)
    parser.add_argument("--estimators", type=int, default=2)
    parser.add_argument("--stations", type=int, nargs="+")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    run(parser.parse_args())
