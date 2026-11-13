"""Test whether the 2025 Geomdan extension identifies transferable station shares."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from sincon_complementary_blend import historical_station_shift, interval, rmsle
from sincon_station_state import KEY, WINDOWS
from sincon_weekly_variability import past_episodes, protected_days


OPENED = pd.Timestamp("2025-06-28")
PLACEBO = pd.Timestamp("2025-05-17")
OLD = [3110, 3111]
NEW = [3107, 3108, 3109]
DONORS = [station for station in range(3115, 3139) if station not in [3125, 3127]]
STATIONS = OLD + NEW + DONORS
HOURS = [6, 10, 16, 21]
REQUIRED_GAIN = .003


def bands(hours):
    return np.asarray(HOURS)[np.searchsorted(HOURS, hours, side="right") - 1]


def panel(frame, value, holidays, events):
    source = frame.loc[frame.station_id.isin(STATIONS) & ~frame.date.isin(holidays), [*KEY, value]].copy()
    source = source.loc[~pd.MultiIndex.from_frame(source[KEY[:2]]).isin(events)]
    source["band"] = bands(source.hour)
    by_band = source.groupby(["date", "band", "station_id"])[value].sum().unstack("station_id")
    return np.log1p(by_band.reindex(columns=STATIONS))


def normalize(observed, means):
    dates = observed.index.get_level_values("date")
    indexes = pd.MultiIndex.from_arrays([dates.dayofweek, observed.index.get_level_values("band")], names=["weekday", "band"])
    typical = means.reindex(indexes).set_axis(observed.index)
    return observed - typical, typical


def fit_control(before):
    dates = before.index.get_level_values("date")
    if before.empty or dates.max() >= OPENED:
        raise ValueError("Fit the synthetic controls before the rail extension opened.")
    means = before.groupby([dates.dayofweek.rename("weekday"), "band"]).median()
    differences, _ = normalize(before, means)
    weights = {}
    for hour, rows in differences.groupby("band"):
        good = rows[[*OLD, *DONORS]].dropna()
        x = good[DONORS].to_numpy()
        # Nonnegative station weights make a station's forecast insensitive to
        # cancellation between a few unstable stations. Shrink toward equal shares.
        n = len(DONORS)
        regularization = np.eye(n) * np.sqrt(.12)
        design = np.vstack([x, regularization, np.full((1, n), 8.)])
        for station in OLD:
            outcomes = np.r_[good[station], np.full(n, np.sqrt(.12) / n), 8.]
            coefficients, _ = nnls(design, outcomes)
            weights[hour, station] = coefficients / coefficients.sum()
    return means, weights


def control_prediction(observed, model):
    means, weights = model
    residuals, typical = normalize(observed, means)
    estimates = typical[OLD].copy()
    for hour, positions in residuals.groupby("band").indices.items():
        peers = residuals[DONORS].iloc[positions].to_numpy()
        observed_peers = np.isfinite(peers)
        for column, station in enumerate(OLD):
            weight = weights[hour, station]
            available = observed_peers @ weight
            relative = np.divide(np.nan_to_num(peers) @ weight, available,
                                 out=np.zeros(len(peers)), where=available >= .75)
            estimates.iloc[positions, column] += relative
    return estimates


def relative_effect(observed, counterfactual, minimum):
    errors = (observed[OLD] - counterfactual).stack(future_stack=True).rename("effect").reset_index().dropna()
    errors["weekend"] = errors.date.dt.dayofweek.ge(5)
    values = errors.groupby(["station_id", "band", "weekend"]).effect.agg(["median", "count"])
    enough = values["count"].ge(np.where(values.index.get_level_values("weekend"), minimum[1], minimum[0]))
    return values["median"].where(enough)


def historical_check(observed, model):
    dates = observed.index.get_level_values("date")
    expected = control_prediction(observed, model)
    era = {
        "first_stable_month": ("2025-07-14", "2025-08-08"),
        "september_2025": ("2025-09-01", "2025-09-30"),
        "january_2026": ("2026-01-05", "2026-02-12"),
        "before_forecast": ("2026-05-01", "2026-06-25"),
    }
    diagnostics = {}
    for name, (first, last) in era.items():
        select = (dates >= first) & (dates <= last) & (dates.dayofweek < 5)
        old = np.expm1(observed.loc[select, OLD]).groupby("date").sum()
        cf = np.expm1(expected.loc[select]).groupby("date").sum()
        new = np.expm1(observed.loc[select, NEW]).groupby("date").sum().sum(axis=1)
        diagnostics[name] = {"weekdays": len(old), "ratio_to_no_opening_prediction": (old / cf).median().to_dict(),
                             "net_old_deficit_divided_by_new_boardings": float(((cf - old).sum(axis=1) / new).median())}
    before = observed.loc[dates < PLACEBO]
    holdout = observed.loc[(dates >= PLACEBO) & (dates < OPENED)]
    estimate = control_prediction(holdout, fit_control(before))
    residual = holdout[OLD] - estimate
    diagnostics["preopening_placebo"] = {"origin": str((PLACEBO - pd.Timedelta(days=1)).date()),
                                         "median_log_error": residual.median().to_dict(),
                                         "mean_log_error": residual.mean().to_dict()}
    return diagnostics


def predict(history, future, origin, holidays, events, model):
    if (history.index.get_level_values("date").max() > origin or future.date.min() <= origin
            or {"y_log", "boardings", "alightings"} & set(future)):
        raise ValueError("Counterfactual forecasts require prior counts and unlabeled target rows.")
    dates = history.index.get_level_values("date")
    age = (dates - OPENED).days
    recent = (age >= 14) & ((origin - dates).days < 56)
    early = (age >= 14) & (age < 70)
    recent_effect = relative_effect(history.loc[recent], control_prediction(history.loc[recent], model), (10, 4))
    early_effect = relative_effect(history.loc[early], control_prediction(history.loc[early], model), (10, 4))
    stable = (np.sign(recent_effect) == np.sign(early_effect)) & (recent_effect - early_effect).abs().le(.20)
    query = future.assign(count=np.expm1(future.base.clip(lower=0.)))
    query_panel = panel(query, "count", holidays, events)
    control = control_prediction(query_panel, model)
    panel_keys = pd.MultiIndex.from_frame(future[["date", "station_id"]].assign(band=bands(future.hour)))
    grid = pd.MultiIndex.from_frame(query[["date"]].assign(band=bands(query.hour)))
    changes = {}
    for name, effect in (("incumbents", recent_effect), ("stable_opening_effect_only", recent_effect.where(stable))):
        gaps = (control - query_panel[OLD]).stack(future_stack=True).rename("level_gap").reset_index()
        gaps["weekend"] = gaps.date.dt.dayofweek.ge(5)
        effects = effect.reindex(pd.MultiIndex.from_frame(gaps[["station_id", "band", "weekend"]])).to_numpy()
        gaps["delta"] = .65 * (gaps.level_gap + effects).clip(-.35, .35)
        adjustment = gaps.set_index(["date", "station_id", "band"]).delta.reindex(panel_keys).fillna(0.).to_numpy()
        allowed = ~pd.MultiIndex.from_frame(future[KEY[:2]]).isin(events)
        changes[name] = np.where(allowed, adjustment, 0.)
    # A separate transfer scenario keeps the five-station count total fixed
    # by applying the opposite count adjustment to the three new stations.
    count = query["count"].to_numpy()
    moved = count * np.expm1(changes["incumbents"])
    shift_total = pd.Series(moved).groupby(grid).transform("sum").to_numpy()
    new = future.station_id.isin(NEW).to_numpy()
    eligible_new = new & ~pd.MultiIndex.from_frame(future[KEY[:2]]).isin(events)
    new_total = pd.Series(np.where(eligible_new, count, 0.)).groupby(grid).transform("sum").to_numpy()
    feasible = (new_total > 0.) & (shift_total < new_total)
    relative = np.divide(shift_total, new_total, out=np.zeros_like(count), where=feasible)
    counter = np.zeros_like(count)
    counter[eligible_new] = np.log1p(-relative[eligible_new])
    changes["count_neutral_transfer"] = np.where(feasible, changes["incumbents"] + counter, 0.)
    estimates = {name: np.log1p(count * np.exp(delta)) for name, delta in changes.items()}
    neutrality = pd.Series(np.expm1(estimates["count_neutral_transfer"]) - count).groupby(grid).sum()
    if neutrality.abs().max() > 1e-6:
        raise ValueError("Neutral forecast must preserve boarding totals within each date and time band.")
    return estimates, {"recent_effect_cells": int(recent_effect.notna().sum()), "stable_cells": int(stable.sum())}


def incumbent_error_budget(rows, baseline):
    # Held-out labels are used only for this diagnostic bound.
    actual = rows.y_log.to_numpy()
    residual = pd.Series(actual - baseline).where(rows.station_id.isin(OLD))
    by_station_band = [rows.station_id, bands(rows.hour)]
    result = {}
    for name, grouping in (("by_month_and_weekend", [rows.date.dt.to_period("M"), rows.date.dt.dayofweek.ge(5)]),
                           ("by_date", [rows.date])):
        oracle = residual.groupby([*by_station_band, *grouping]).transform("mean").fillna(0.).to_numpy()
        result[name] = rmsle(actual, baseline) - rmsle(actual, baseline + oracle)
    return result


def run(work, team, earlier, newer):
    holidays, events = protected_days(work)
    current, archive = past_episodes(work / "sincon_local_validation.csv.gz", earlier, holidays, events)
    train = pd.read_csv(team / "data/train.csv", usecols=[*KEY, "boardings"], parse_dates=["date"])
    truth = np.log1p(train.set_index(KEY).boardings)
    observed = panel(train, "boardings", holidays, events)
    dates = observed.index.get_level_values("date")
    model = fit_control(observed.loc[dates < OPENED])
    results = {"opening": str(OPENED.date()), "opening_source": "https://www.incheon.go.kr/mayor_8/MA040101/view?curPage=255&repSeq=DOM_0000000012620245",
               "diagnostics": historical_check(observed, model), "folds": {}}
    spans = {name: (pd.Timestamp(WINDOWS[name][0]), rows) for name, rows in current.items()}
    warm = work / "sincon_summer_replay_validation.csv.gz"
    for start, path, end in [("2025-06-30", warm, "2025-10-31"),
                             ("2025-07-31", earlier / "origin_2025-07-31.csv.gz", "2025-10-31"),
                             ("2025-08-31", earlier / "origin_2025-08-31.csv.gz", "2025-10-31"),
                             ("2026-03-31", newer / "origin_2026-03-31.csv.gz", "2026-06-30"),
                             ("2026-04-30", newer / "origin_2026-04-30.csv.gz", "2026-06-30")]:
        rows = pd.read_csv(path, usecols=[*KEY, "y_log", "local_stack"], parse_dates=["date"])
        spans[start] = pd.Timestamp(start), rows.loc[rows.date.le(end)].rename(columns={"local_stack": "base"})
    for name, (origin, rows) in spans.items():
        rows = rows.reset_index(drop=True)
        if (rows.duplicated(KEY).any() or not np.allclose(rows.y_log, truth.reindex(pd.MultiIndex.from_frame(rows[KEY])), atol=1e-8, rtol=0)):
            raise ValueError(f"Invalid validation rows for {name}.")
        if name in WINDOWS or origin.year == 2026:
            station_shift, calibration_days = historical_station_shift(archive, rows, origin)
        else:
            station_shift, calibration_days = 0., 0
        baseline = (rows.base + station_shift).clip(lower=0.).to_numpy()
        variants, details = predict(observed.loc[dates <= origin], rows[KEY].assign(base=baseline), origin, holidays, events, model)
        actual = rows.y_log.to_numpy()
        local = rows.station_id.isin(OLD + NEW).to_numpy()
        scores = {}
        for variant, forecast in {"baseline": baseline, **variants}.items():
            scores[variant] = {"overall": rmsle(actual, forecast), "corridor": rmsle(actual[local], forecast[local]),
                               "week_gain_interval": interval(rows, actual, baseline, forecast),
                               "changed_rows": int(np.count_nonzero(np.abs(forecast - baseline) > 1e-10))}
        results["folds"][name] = {"origin": str(origin.date()), "end": str(rows.date.max().date()), "n": len(rows),
                                   "calibration_days": calibration_days, **details, "scores": scores,
                                   "held_out_incumbents_oracle_gain": incumbent_error_budget(rows, baseline)}
    results["accepted"] = {variant: sum(
        results["folds"][name]["scores"]["baseline"]["overall"] - results["folds"][name]["scores"][variant]["overall"]
        >= REQUIRED_GAIN for name in WINDOWS) >= 2 for variant in variants}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "earlier", "newer"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    started = perf_counter()
    result = run(args.work_dir, args.team, args.earlier, args.newer)
    result["cpu_seconds"] = round(perf_counter() - started, 2)
    text = json.dumps(result, indent=2, allow_nan=False)
    (args.work_dir / "sincon_geomdan_opening_control_results.json").write_text(text + "\n")
    print(text)
