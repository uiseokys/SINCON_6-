"""Validate two bounded blends of independent team forecasts and the local stack."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from sincon_complementary_blend import historical_station_shift
from sincon_station_state import KEY, WINDOWS
from sincon_weekly_variability import past_episodes, protected_days


MODELS = ["xgb", "catboost", "dlinear", "nlinear"]
JOIN = ["fold", *KEY]
MIN_HISTORY_DAYS = 45
TOTAL_WEIGHT_CAP = .30
PENALTY = .35
SEASON_WEIGHT = .10


def verified_archive(work, team, local_validation=None):
    archive = pd.read_csv(local_validation or work / "sincon_local_validation.csv.gz",
                          parse_dates=["date"], usecols=[*JOIN, "y_log", "local_stack"])
    zoo = pd.read_csv(team / "experiments/valid_predictions_zoo.csv.gz", parse_dates=["date"], usecols=[*JOIN, "y_log", *MODELS])
    merged = archive.merge(zoo.rename(columns={"y_log": "zoo_y"}), on=JOIN, validate="one_to_one", how="outer", indicator=True)
    official = pd.read_csv(team / "data/train.csv", parse_dates=["date"], usecols=[*KEY, "boardings"])
    if official.date.max() != pd.Timestamp("2026-06-30") or official.duplicated(KEY).any():
        raise ValueError("Expected the historical official competition data.")
    truth = np.log1p(official.set_index(KEY).boardings)
    exact = truth.reindex(pd.MultiIndex.from_frame(merged[KEY])).to_numpy()
    if (not merged.pop("_merge").eq("both").all() or not np.allclose(merged.y_log, exact, atol=1e-8, rtol=0)
            or not np.allclose(merged.pop("zoo_y"), exact, atol=5.1e-7, rtol=0)
            or not np.isfinite(merged[["local_stack", *MODELS]].to_numpy()).all()):
        raise ValueError("Zoo and local forecasts need complete matching keys and official labels.")
    for fold, (cutoff, end) in WINDOWS.items():
        sample = merged.loc[merged.fold.eq(fold)]
        dates = pd.date_range(pd.Timestamp(cutoff) + pd.Timedelta(days=1), end)
        if len(sample) != len(dates) * 71 * 18 or set(sample.date) != set(dates):
            raise ValueError(f"Unexpected date grid for fold {fold}.")
    merged["y_log"] = exact
    merged["origin"] = pd.to_datetime(merged.fold.map({fold: span[0] for fold, span in WINDOWS.items()}))
    return merged, truth


def features(rows, models, holidays, events):
    ordinary = ~rows.date.isin(holidays) & ~pd.MultiIndex.from_frame(rows[KEY[:2]]).isin(events)
    difference = rows[models].sub(rows.local_stack, axis=0)
    station_day = [rows.origin, rows.date, rows.station_id]
    shape = difference - difference.groupby(station_day).transform("mean")
    return ordinary, difference.where(ordinary, 0., axis=0), shape.where(ordinary, 0., axis=0)


def fit_weights(rows, prior):
    if rows.date.nunique() < MIN_HISTORY_DAYS:
        return np.zeros(prior.shape[1])
    design = prior.to_numpy()
    residual = (rows.y_log - rows.local_stack).to_numpy()
    shrink = np.diag(np.sqrt(PENALTY * np.square(design).sum(axis=0)))
    weights, _ = nnls(np.vstack([design, shrink]), np.r_[residual, np.zeros(len(shrink))])
    return weights * min(1., TOTAL_WEIGHT_CAP / max(weights.sum(), TOTAL_WEIGHT_CAP))


def fit_before(all_rows, origin, ordinary):
    prior = all_rows.loc[ordinary & all_rows.date.le(origin) & all_rows.origin.lt(origin)]
    return prior.sort_values("origin").drop_duplicates(KEY, keep="first")


def scores(rows, forecasts):
    masks = {"all": np.ones(len(rows), dtype=bool), "late_21_23": rows.hour.ge(21).to_numpy(),
             "mature_68": ~rows.station_id.isin([3107, 3108, 3109]).to_numpy()}
    return {method: {name: float(np.sqrt(np.square(rows.y_log.to_numpy()[mask] - np.maximum(values[mask], 0)).mean()))
                     for name, mask in masks.items()} for method, values in forecasts.items()}


def weekly_comparison(rows, baseline, candidate):
    blocks = ((rows.date - rows.date.min()).dt.days // 7).to_numpy()
    records = []
    for block in np.unique(blocks):
        selected = blocks == block
        subset = rows.loc[selected]
        results = scores(subset, {"local_stack": baseline[selected], "xgb_only_10pct": candidate[selected]})
        gains = {metric: results["local_stack"][metric] - results["xgb_only_10pct"][metric]
                 for metric in results["local_stack"]}
        records.append({"first": str(subset.date.min().date()), "last": str(subset.date.max().date()),
                        "rows": len(subset), "rmsle_gain": gains})
    return records


def run(team, work, earlier, season_path, recent_paths):
    frame, official = verified_archive(work, team)
    holidays, events = protected_days(work)
    _, prior_shifts = past_episodes(work / "sincon_local_validation.csv.gz", earlier, holidays, events)
    ordinary, direct, shape = features(frame, MODELS, holidays, events)
    designs = {"direct_blend": direct, "within_station_day_shape": shape}
    runs, segments = {}, []
    for fold, (cutoff, _) in WINDOWS.items():
        origin = pd.Timestamp(cutoff)
        target = frame.loc[frame.fold.eq(fold)]
        prior = fit_before(frame, origin, ordinary)
        base = target.local_stack.to_numpy()
        forecasts = {"local_stack": base}
        coefficients = {}
        for method, design in designs.items():
            weights = fit_weights(prior, design.loc[prior.index])
            delta = design.loc[target.index].to_numpy() @ weights
            if (delta[~ordinary.loc[target.index].to_numpy()] != 0).any():
                raise ValueError("The model blend changed a protected date or station-day.")
            if method == "within_station_day_shape":
                sums = pd.Series(delta, index=target.index).groupby([target.date, target.station_id]).sum()
                if sums.abs().max() > 1e-12:
                    raise ValueError("Within-day blending must preserve the daily mean in log space.")
            forecasts[method] = base + delta
            coefficients[method] = dict(zip(MODELS, weights))
        xgb_weight, = fit_weights(prior, direct.loc[prior.index, ["xgb"]])
        forecasts["xgb_only_prior"] = base + xgb_weight * direct.loc[target.index, "xgb"].to_numpy()
        forecasts["xgb_only_10pct"] = base + SEASON_WEIGHT * direct.loc[target.index, "xgb"].to_numpy()
        coefficients["xgb_only_prior"] = {"xgb": xgb_weight}
        coefficients["xgb_only_10pct"] = {"xgb": SEASON_WEIGHT}
        station_shift, _ = historical_station_shift(prior_shifts, target, origin)
        calibrated = {k: v + station_shift for k, v in forecasts.items()}
        result = {"rows": len(target), "origin": cutoff, "weight_fit_days": prior.date.nunique(),
                  "weight_fit_last_label": str(prior.date.max().date()) if len(prior) else None,
                  "weights": coefficients, "raw": scores(target, forecasts),
                  "prior_only_station_calibration": scores(target, calibrated),
                  "standalone": scores(target, {name: target[name].to_numpy() for name in MODELS}),
                  "weekly_fixed_10pct_raw": weekly_comparison(target, base, forecasts["xgb_only_10pct"]),
                  "weekly_fixed_10pct_prior_calibrated": weekly_comparison(
                      target, calibrated["local_stack"], calibrated["xgb_only_10pct"],
                  )}
        runs[fold] = result
        segments.append(target[JOIN + ["y_log"]].assign(**calibrated))
    joined = pd.concat(segments, ignore_index=True)
    disjoint = joined.loc[joined.fold.isin(["A", "C"])]
    if disjoint.duplicated(KEY).any():
        raise ValueError("A/C must contain disjoint scored dates.")
    result = {"folds": runs, "disjoint_ac_prior_calibrated": scores(disjoint, {
        name: disjoint[name].to_numpy() for name in forecasts}),
        "adoption": {name: [fold for fold, record in runs.items() if (
            record["prior_only_station_calibration"]["local_stack"]["all"]
            - record["prior_only_station_calibration"][name]["all"] >= .003)]
            for name in [*designs, "xgb_only_prior", "xgb_only_10pct"]},
        "target_season": {"status": "pending"}, "xgb_only_follow_up": True}
    if season_path:
        result["target_season"] = fixed_origin_scores(
            season_path, work / "sincon_summer_replay_validation.csv.gz", "2025-06-30", "2025-10-31",
            official, holidays, events,
        )
    if recent_paths:
        cutoff, end = "2026-04-30", "2026-06-30"
        prior = fit_before(frame, pd.Timestamp(cutoff), ordinary)
        xgb_weight, = fit_weights(prior, direct.loc[prior.index, ["xgb"]])
        result["recent_historical_origin"] = fixed_origin_scores(
            recent_paths[1], recent_paths[0], cutoff, end, official, holidays, events, xgb_weight,
        )
        result["recent_historical_origin"]["weight_fit_days"] = prior.date.nunique()
    return result


def fixed_origin_scores(path, archive, origin, end, official, holidays, events, weight=None):
    xgb = pd.read_csv(path, parse_dates=["origin", "date"], usecols=["origin", *KEY, "xgb"])
    fields = [*KEY, "y_log", "local_stack"]
    declared_origin = "origin" in pd.read_csv(archive, nrows=0)
    local = pd.read_csv(archive, parse_dates=["date"], usecols=fields + (["origin"] if declared_origin else []))
    if declared_origin and not pd.to_datetime(local.pop("origin")).eq(pd.Timestamp(origin)).all():
        raise ValueError("The local forecast archive has a different origin.")
    rows = local.merge(xgb, on=KEY, how="outer", indicator=True, validate="one_to_one")
    labels = official.reindex(pd.MultiIndex.from_frame(rows[KEY])).to_numpy()
    dates = pd.date_range(pd.Timestamp(origin) + pd.Timedelta(days=1), end)
    if (not rows.pop("_merge").eq("both").all() or not np.allclose(rows.y_log, labels, atol=1e-8, rtol=0)
            or not rows.origin.eq(pd.Timestamp(origin)).all() or len(rows) != len(dates) * 71 * 18
            or set(rows.date) != set(dates) or not np.isfinite(rows.xgb).all()):
        raise ValueError("XGBoost and local archive need complete matching fixed-origin rows.")
    _, direct, shape = features(rows, ["xgb"], holidays, events)
    base = rows.local_stack.to_numpy()
    predictions = {"local_stack": base, "xgb_standalone": rows.xgb.to_numpy(),
                   "direct_blend_10pct": base + SEASON_WEIGHT * direct.xgb.to_numpy(),
                   "within_station_day_shape_10pct": base + SEASON_WEIGHT * shape.xgb.to_numpy()}
    if weight is not None:
        predictions["direct_blend_prior_only"] = base + weight * direct.xgb.to_numpy()
    return {"status": "verified", "origin": origin, "first": str(dates.min().date()), "last": end,
            "rows": len(rows), "model": "xgb", "fixed_weight": SEASON_WEIGHT, "prior_only_weight": weight,
            "scores": scores(rows, predictions),
            "weekly_fixed_10pct_raw": weekly_comparison(rows, base, predictions["direct_blend_10pct"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team", type=Path)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("--season-xgb", type=Path)
    parser.add_argument("--recent", type=Path, nargs=2, metavar=("LOCAL", "XGBOOST"))
    parser.add_argument("--work", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    report = run(args.team, args.work, args.earlier, args.season_xgb, args.recent)
    output = args.work / "sincon_zoo_transfer_results.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
