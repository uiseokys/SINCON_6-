"""Estimate conservative weights for the fixed blend using earlier forecasts only."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from sincon_complementary_blend import JOIN, historical_station_shift, rmsle
from sincon_stadium_concert_analogs import read_inputs as concert_inputs
from sincon_station_state import KEY, WINDOWS
from sincon_weekly_variability import past_episodes, protected_days


SOURCES = {
    "housing": ("sincon_housing_walking_validation.csv.gz", "walking_ramp_full"),
    "holiday": ("sincon_holiday_transfer_validation.csv.gz", "pooled_same_holiday"),
    "bus": ("sincon_seoul_bus_validation.csv.gz", "bus_daily"),
    "annual": ("sincon_matched_year_validation.csv.gz", "annual_blend"),
    "regime": ("sincon_regime_anchors_validation.csv.gz", "detected_break"),
}
FIELDS = [*SOURCES, "weekly_and_peak"]


def load(work, team):
    rows = pd.read_csv(work / "sincon_complementary_blend_validation.csv.gz", usecols=[
        *JOIN, "y_log", "local_stack", "fixed_blend",
    ], parse_dates=["date"])
    for field, (file, column) in SOURCES.items():
        source = pd.read_csv(work / file, usecols=[*JOIN, "y_log", "reference", column], parse_dates=["date"])
        source = source.rename(columns={"y_log": "source_y", column: field})
        rows = rows.merge(source, on=JOIN, how="left", validate="one_to_one")
        if (not np.allclose(rows.pop("source_y"), rows.y_log, atol=1e-8, rtol=0)
                or not np.allclose(rows.pop("reference"), rows.local_stack, atol=1e-8, rtol=0)):
            raise ValueError(f"{file} has different target labels or baseline values.")
        rows[field] -= rows.local_stack

    days = KEY[:2]
    with ZipFile(work / "sincon_local_repro.zip") as bundle:
        root = "external_inputs/"
        calendar = pd.read_csv(bundle.open(root + "calendar_features.csv"), parse_dates=["date"])
    festivals = pd.read_csv(work / "sincon_festival_station_days.csv", usecols=days, parse_dates=["date"])
    baseball = pd.read_csv(work / "sincon_kbo_game_data.csv", usecols=["date"], parse_dates=["date"]).assign(station_id=3127)
    football = pd.read_csv(work / "sincon_bucheon_football_data.csv", usecols=["date"], parse_dates=["date"]).assign(station_id=3754)
    concerts = concert_inputs(work, team)[2][days]
    event = pd.MultiIndex.from_frame(pd.concat([festivals, baseball, football, concerts]).drop_duplicates())
    ordinary_place = ~pd.MultiIndex.from_frame(rows[days]).isin(event)
    ordinary_date = rows.date.isin(calendar.loc[calendar.is_holiday.eq(0), "date"])
    origin = pd.to_datetime(rows.fold.map({name: dates[0] for name, dates in WINDOWS.items()}))
    rows["origin"] = origin
    years = (origin - calendar.date.min()).dt.days.ge(373)
    structural = ordinary_place & ordinary_date & rows.housing.abs().lt(.01)
    rows["regime"] = rows.regime.where(structural, 0.)
    rows["annual"] = rows.annual.where(years, 0.)
    rows["bus"] = rows.bus.where(ordinary_date, 0.)
    rows[list(SOURCES)] = rows[list(SOURCES)].mul(ordinary_place, axis=0)
    rows["weekly_and_peak"] = rows.fixed_blend - rows.local_stack - rows[list(SOURCES)].sum(axis=1)
    return rows


def fit_weights(past):
    if past.date.nunique() < 45:
        return np.ones(len(FIELDS))
    features = past[FIELDS].to_numpy()
    actual = (past.y_log - past.local_stack - past.station_shift).to_numpy().clip(-.65, .65)
    supported = (np.abs(features) > 1e-10).sum(axis=0) >= 200
    actual -= features[:, ~supported].sum(axis=1)
    x = features[:, supported]
    prior = np.diag(np.sqrt(np.square(x).sum(axis=0) * .35))
    fit = lsq_linear(np.vstack([x, prior]), np.r_[actual, prior.sum(axis=1)], bounds=(0., 1.5))
    if not fit.success:
        raise ValueError("Source reweighting optimization did not converge.")
    weights = np.ones(len(FIELDS))
    weights[supported] = fit.x
    return weights


def prior_rows(rows, origin):
    known = rows.loc[rows.origin.lt(origin) & rows.date.le(origin)]
    return known.sort_values("origin").drop_duplicates(KEY, keep="first")


def evaluate(work, team, earlier):
    rows = load(work, team)
    holidays, events = protected_days(work)
    _, archive = past_episodes(work / "sincon_local_validation.csv.gz", earlier, holidays, events)
    rows["station_shift"] = 0.
    for fold, (cutoff, _) in WINDOWS.items():
        mask = rows.fold.eq(fold)
        rows.loc[mask, "station_shift"], _ = historical_station_shift(archive, rows.loc[mask], pd.Timestamp(cutoff))
    report = {"folds": {}}
    for fold, (cutoff, _) in WINDOWS.items():
        origin = pd.Timestamp(cutoff)
        known = prior_rows(rows, origin)
        weights = fit_weights(known)
        target = rows.loc[rows.fold.eq(fold)]
        raw = target.local_stack.to_numpy()
        shift = target.station_shift.to_numpy()
        forecast = raw + target[FIELDS].to_numpy() @ weights
        fixed = target.fixed_blend.to_numpy()
        actual = target.y_log.to_numpy()
        report["folds"][fold] = {
            "rows": len(target),
            "weights": dict(zip(FIELDS, weights.tolist())),
            "pre_origin_days": int(known.date.nunique()),
            "uncalibrated": {k: rmsle(actual, p) for k, p in (("baseline", raw), ("fixed", fixed), ("learned", forecast))},
            "prior_only_calibration": {k: rmsle(actual, p + shift)
                                       for k, p in (("baseline", raw), ("fixed", fixed), ("learned", forecast))},
        }
    report["adoption_folds"] = [
        fold for fold, f in report["folds"].items()
        if f["uncalibrated"]["baseline"] - f["uncalibrated"]["learned"] >= .003
    ]
    report["prior_only_calibration_adoption_folds"] = [
        fold for fold, f in report["folds"].items()
        if f["prior_only_calibration"]["baseline"] - f["prior_only_calibration"]["learned"] >= .003
    ]
    report["accepted"] = len(report["adoption_folds"]) >= 2
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("team", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    work = args.work_dir
    report, _ = evaluate(work, args.team, args.earlier)
    (work / "sincon_source_reweight_results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
