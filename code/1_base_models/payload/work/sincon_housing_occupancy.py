"""Test complex-level utility fees as evidence of apartment occupancy growth."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from sincon_housing_growth import forecast_changes, read_exposure
from sincon_joint_candidate import rmsle_logs
from sincon_station_state import KEY, WINDOWS, read_team


METERS = np.array([111_195, 111_195 * np.cos(np.deg2rad(37.5))])
UTILITY = ["private_water", "private_electric"]


def inferred_occupancy(fees, stock, complexes):
    details = fees.merge(stock[["complex_id", "units", "approval_date"]], on="complex_id", how="inner", validate="many_to_one")
    details = details.loc[details["units"].ge(100)].copy()
    for field in UTILITY:
        details[field] = details[field] / (details["units"] * details["month"].dt.days_in_month)
    old = details["approval_date"].lt(pd.Timestamp("2024-01-01"))
    coverage = details.loc[old].groupby(["province", "month"])["complex_id"].nunique()
    normal = coverage.ge(0.75 * coverage.groupby("province").transform("max"))
    reference = details.loc[old, ["province", "month", *UTILITY]].copy()
    reference[UTILITY] = reference[UTILITY].where(reference[UTILITY].gt(0))
    median = reference.groupby(["province", "month"])[UTILITY].median().loc[normal]
    eligible = details.loc[details["complex_id"].isin(complexes.index)].copy()
    units = eligible["complex_id"].map(complexes["units"])
    if not (units == eligible["units"]).all():
        raise ValueError("The fee archive and approval register must agree on new complex unit counts.")
    denominators = median.reindex(pd.MultiIndex.from_frame(eligible[["province", "month"]])).to_numpy()
    relative = eligible[UTILITY].to_numpy() / denominators
    # Zero charges are meaningful only for utilities the complex reports.
    billed = eligible.groupby("complex_id")[UTILITY].transform("max").gt(0).to_numpy()
    relative[~billed] = np.nan
    water, electricity = relative.T
    use_both = np.isfinite(relative).all(axis=1)
    ratio = np.where(use_both, 0.75 * water + 0.25 * electricity, np.where(np.isfinite(water), water, electricity))
    return eligible[["complex_id", "month"]].assign(ratio=ratio).dropna(subset="ratio")


def projected_households(complexes, utilization, days, origin, use_actual_through, mix):
    center = complexes["first_month"].fillna(complexes["approval_date"] + pd.Timedelta(days=21))
    start = pd.concat([center - pd.Timedelta(days=30), complexes["approval_date"]], axis=1).max(axis=1)
    maturity = (days.to_numpy()[:, None] - start.to_numpy()[None]) / np.timedelta64(1, "D")
    expected = np.clip(maturity / 60, 0, 1)
    adjustment = np.zeros_like(expected)
    adjusted = {}
    for complex_id, values in utilization.groupby("complex_id"):
        column = complexes.index.get_loc(complex_id)
        centers = values["month"] + pd.Timedelta(days=14)
        age = (centers - start[complex_id]).dt.days.to_numpy()
        month_is_observed = centers.dt.to_period("M").dt.end_time.dt.normalize().le(use_actual_through).to_numpy()
        available = (age >= -21) & (age <= 150) & month_is_observed
        stable = (age >= 120) & centers.le(origin).to_numpy()
        stable_ratio = values["ratio"].to_numpy()[stable]
        # If the complex already has a mature usage record at the forecast
        # origin, normalize its prior ramp against its own billing baseline.
        scale = (np.median(stable_ratio) + 2) / 3 if stable_ratio.size else 1.
        if not available.any():
            continue
        measured = np.clip(values["ratio"].to_numpy()[available] / np.clip(scale, .5, 2), 0, 1)
        curve = np.maximum.accumulate(measured)
        observed_age = age[available]
        residual = curve - np.clip(observed_age / 60, 0, 1)
        knots = np.r_[observed_age.min() - 25, observed_age, observed_age.max() + 45]
        change = np.interp(maturity[:, column], knots, np.r_[0, residual, 0], left=0, right=0)
        adjustment[:, column] = np.where(maturity[:, column] >= 0, change, 0)
        adjusted[complex_id] = int(available.sum())
    exposure = (expected + mix * adjustment).clip(0, 1)
    return exposure * complexes["units"].to_numpy(), adjusted


def exposure_panel(complexes, centers, utilization, days, origin, last_reported, mix, walking=None):
    coordinates = ["latitude", "longitude"]
    distance = cdist(centers[coordinates].to_numpy() * METERS, complexes[coordinates].to_numpy() * METERS)
    if walking is not None:
        network = walking.reindex(index=centers.index, columns=complexes.index).to_numpy()
        # Rescale the kernel's geometric bandwidth for ordinary walking
        # detours. Disconnected or unmeasured routes retain a geometric prior.
        distance = np.where(np.isfinite(network), network / 1.22, distance)
    weights = np.exp(-0.5 * (distance / 550) ** 2) * (distance <= 1_600)
    weights /= np.maximum(1, weights.sum(axis=0))
    occupancy, used = projected_households(complexes, utilization, days, origin, last_reported, mix)
    return pd.DataFrame(occupancy @ weights.T, index=days, columns=centers.index), used


def run(args):
    _, meta, calendar, _ = read_team(args.team_dir, args.external_zip)
    train = pd.read_csv(args.team_dir / "data/train.csv", parse_dates=["date"])
    archive = pd.read_csv(args.validation, parse_dates=["date"])
    centers = pd.read_csv(args.station_centers).set_index("station_id").reindex(meta.index)
    complexes = pd.read_csv(args.housing, parse_dates=["approval_date"], dtype={"first_month": str}).set_index("complex_id")
    complexes["first_month"] = pd.to_datetime(complexes["first_month"], format="%Y%m", errors="coerce") + pd.Timedelta(days=14)
    fees = pd.read_csv(args.fees, parse_dates=["month"])
    stock = pd.read_csv(args.fee_stock, parse_dates=["approval_date"])
    usage = inferred_occupancy(fees, stock, complexes)
    walking = pd.read_csv(args.walking).pivot(index="station_id", columns="complex_id", values="walk_m") if args.walking else None
    base_exposure = read_exposure(args.housing, centers)
    results, outputs = {}, []
    for fold, (cutoff, end) in WINDOWS.items():
        started = perf_counter()
        origin, last = pd.Timestamp(cutoff), pd.Timestamp(end)
        history = train.loc[train["date"].le(origin)]
        valid = archive.loc[archive["fold"].eq(fold)]
        keys = valid[KEY]
        truth, base = (valid[name].to_numpy() for name in ("y_log", "local_stack"))
        old, _, _ = forecast_changes(history, keys, origin, base_exposure, calendar, meta)
        prior_fees = fees.loc[fees["month"].dt.to_period("M").dt.end_time.dt.normalize().le(origin)]
        prior_usage = inferred_occupancy(prior_fees, stock, complexes)
        versions = {"reference": base}
        details = {}
        hidden = last - pd.offsets.MonthEnd(1)
        variants = [("utility_ramp", last, 0.75, None), ("utility_with_last_month_hidden", hidden, 0.75, None)]
        if walking is not None:
            variants += [("walking_ramp", last, 0., walking), ("walking_utility", last, 0.75, walking),
                         ("walking_utility_last_month_hidden", hidden, 0.75, walking)]
        for label, last_reported, mix, network in variants:
            historical_exposure, _ = exposure_panel(complexes, centers, prior_usage, base_exposure.index, origin, origin, mix, network)
            new_exposure, available = exposure_panel(complexes, centers, usage, base_exposure.index, origin, last_reported, mix, network)
            new_exposure.loc[:origin] = historical_exposure.loc[:origin]
            new, _, _ = forecast_changes(history, keys, origin, new_exposure, calendar, meta)
            versions[label] = base + 0.5 * (new - old)
            versions[f"{label}_full"] = base + new - 0.5 * old
            details[label] = {"complexes_with_utility": len(available), "affected_rows": int(np.count_nonzero(abs(new - old) > 0.01))}
        results[fold] = {"rmsle": {name: rmsle_logs(truth, pred) for name, pred in versions.items()},
                         "details": details, "seconds": round(perf_counter() - started, 2)}
        outputs.append(keys.assign(fold=fold, y_log=truth, **versions))
        args.results.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({fold: results[fold]}, allow_nan=False), flush=True)
    if args.predictions:
        pd.concat(outputs, ignore_index=True).to_csv(args.predictions, index=False, float_format="%.9f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team_dir", "external_zip", "validation", "housing", "station_centers", "fees", "fee_stock"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--walking", type=Path)
    run(parser.parse_args())
