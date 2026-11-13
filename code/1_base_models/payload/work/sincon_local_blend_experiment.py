"""Test independent housing, calendar, and mobility adjustments jointly."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_festival_experiment import local_days
from sincon_joint_candidate import rmsle_logs
from sincon_stacked_experiment import interval
from sincon_station_state import KEY, WINDOWS, read_team


SOURCES = {
    "housing": ("sincon_housing_walking_validation.csv.gz", "walking_ramp_full"),
    "holiday": ("sincon_holiday_transfer_validation.csv.gz", "pooled_same_holiday"),
    "bus": ("sincon_seoul_bus_validation.csv.gz", "bus_daily"),
    "annual": ("sincon_matched_year_validation.csv.gz", "annual_blend"),
    "regime": ("sincon_regime_anchors_validation.csv.gz", "detected_break"),
}
JOIN = ["fold", *KEY]


def attach(archive, directory):
    for name, (file, column) in SOURCES.items():
        model = pd.read_csv(directory / file, usecols=[*JOIN, "reference", "y_log", column], parse_dates=["date"])
        model = model.rename(columns={"y_log": "source_y", column: name})
        archive = archive.merge(model, on=JOIN, validate="one_to_one", how="left")
        if not (np.allclose(archive.pop("reference"), archive["local_stack"], atol=1e-8, rtol=0)
                and np.allclose(archive.pop("source_y"), archive["y_log"], atol=1e-8, rtol=0)):
            raise ValueError(f"Mismatched rows, labels, or reference forecasts in {file}.")
        archive[name] -= archive["local_stack"]
    return archive


def event_keys(festival_path, baseball_path, football_path):
    festival = local_days(festival_path)[KEY[:2]]
    baseball = pd.read_csv(baseball_path, usecols=["date"], parse_dates=["date"]).assign(station_id=3127)
    football = pd.read_csv(football_path, usecols=["date"], parse_dates=["date"]).assign(station_id=3754)
    return pd.MultiIndex.from_frame(pd.concat([festival, baseball, football]).drop_duplicates())


def run(args):
    _, _, calendar, _ = read_team(args.team_dir, args.external_zip)
    events = event_keys(args.festivals, args.baseball, args.football)
    archive = pd.read_csv(args.directory / "sincon_local_validation.csv.gz", usecols=[*JOIN, "y_log", "local_stack"], parse_dates=["date"])
    archive = attach(archive, args.directory)
    results, saved = {}, []
    for fold, (cutoff, _) in WINDOWS.items():
        frame = archive.loc[archive["fold"].eq(fold)]
        actual, reference = (frame[column].to_numpy() for column in ("y_log", "local_stack"))
        ordinary_place = ~pd.MultiIndex.from_frame(frame[KEY[:2]]).isin(events)
        ordinary_date = frame["date"].isin(calendar.index[calendar["is_holiday"].eq(0)]).to_numpy()
        annual_pairs = (pd.Timestamp(cutoff) - calendar.index.min()).days >= 373
        housing = frame["housing"].to_numpy()
        change = housing + frame["holiday"].to_numpy() + ordinary_date * frame["bus"].to_numpy()
        if annual_pairs:
            change += frame["annual"].to_numpy()
        combined = reference + ordinary_place * change
        # A sustained break is an alternate level estimate for a station
        # with no meaningful change in apartment exposure that day.
        regime = np.where(ordinary_place & ordinary_date & (np.abs(housing) < 0.01), frame["regime"], 0)
        variants = {"reference": reference, "independent_sources": combined, "with_regime": combined + regime}
        results[fold] = {"rmsle": {name: rmsle_logs(actual, pred) for name, pred in variants.items()},
                         "paired_week_interval": {name: interval(frame, actual, reference, pred)
                                                  for name, pred in variants.items() if name != "reference"},
                         "annual_adjustment_supported": bool(annual_pairs)}
        saved.append(frame[JOIN].assign(y_log=actual, **variants))
        print(json.dumps({fold: results[fold]}, allow_nan=False), flush=True)
    args.results.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.concat(saved, ignore_index=True).to_csv(args.predictions, index=False, float_format="%.9f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team_dir", "external_zip", "directory", "festivals", "baseball", "football"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    run(parser.parse_args())
