"""Fit intraday forecast error from earlier completed ordinary station-days."""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from sincon_geomdan_morning_anchor import forecast as geomdan_morning
from sincon_lunar_network_level import calendars, combined_replay, proposal, verified_lunar_phases
from sincon_munhak_ordinary import WARM
from sincon_network_level_audit import STABLE_EXCLUSIONS, ordinary
from sincon_station_state import KEY
from sincon_weather_sensitivity import FEATURES, context, score
from sincon_weekly_variability import protected_days


DAY = KEY[:2]
METHODS = ("anomaly_ridge", "conditional_tree", "tail_shrinkage")
LATE_DIAGNOSTICS = ("late_tree_full", "late_tree_shape")
BAND = (6, 10, 16, 21)
WEATHER = tuple(FEATURES)
ANOMALIES = ("shape_change", "daily_change", *[f"{name}_change" for name in WEATHER], "school_change")
DESIGN = (*ANOMALIES, "forecast_shape", "baseline", "station_level", "hour", "weekend", "saturday", "lead", "line", "school")


def latest_validation(args, history, protected, tabicl=None):
    archive = combined_replay(args, tabicl)
    phases = calendars(args.work_dir)
    official = history.set_index(KEY).observed
    results = []
    for label, frame in archive.groupby("label", sort=False):
        origin = frame.origin.iloc[0]
        index = pd.MultiIndex.from_frame(frame[KEY])
        if not index.is_unique or not frame.date.gt(origin).all() or not np.allclose(official.reindex(index), frame.y_log, atol=1e-8, rtol=0.):
            raise ValueError(f"Invalid historical forecast keys or labels for {label}.")
        known = history.loc[history.date.le(origin)]
        query = frame[[*KEY, "baseline"]]
        if frame.date.isin(phases.date).any():
            delta, _ = proposal(known, query, phases, origin, protected)
            frame = frame.assign(baseline=frame.baseline + verified_lunar_phases(frame, delta, phases))
            query = frame[[*KEY, "baseline"]]
        shifts, _ = geomdan_morning(known, query, origin, protected)
        results.append(frame.assign(baseline=(frame.baseline + shifts["relative_peak_share"]).clip(lower=0.)))
    return pd.concat(results, ignore_index=True)


def design(rows, origin, settings, protected):
    if rows.date.min() <= origin or {"y_log", "boardings"} & set(rows):
        raise ValueError("Build the calibration design using unlabeled future station-hours.")
    station, calendar, weather, _, _ = settings
    active = ordinary(rows, protected) & ~rows.station_id.isin(STABLE_EXCLUSIONS)
    active &= ~(rows.station_id.eq(3131) & rows.date.dt.day_of_week.eq(6))
    frame = rows.loc[active, [*KEY, "baseline"]].copy()
    frame["weekday"] = frame.date.dt.day_of_week
    frame["month"] = frame.date.dt.month
    frame["weekend"] = frame.weekday.ge(5).astype(float)
    frame["saturday"] = frame.weekday.eq(5).astype(float)
    frame["lead"] = (frame.date - origin).dt.days / 123.
    frame["line"] = frame.station_id.map(station.canonical_line).astype(float)
    frame["school"] = frame.date.map(calendar)
    frame["weather_zone"] = frame.station_id.map(station.weather_zone)
    lookup = pd.MultiIndex.from_frame(frame[["date", "weather_zone", "hour"]])
    frame[list(WEATHER)] = weather.reindex(lookup)[list(WEATHER)].to_numpy()
    day = frame.groupby(DAY, sort=False).baseline
    if not day.size().eq(18).all():
        raise ValueError("Ordinary comparisons must include complete station-days.")
    frame["station_level"] = day.transform("mean")
    frame["forecast_shape"] = frame.baseline - frame.station_level
    groups = ["station_id", "month", "weekday"]
    frame["shape_change"] = frame.forecast_shape - frame.groupby([*groups, "hour"]).forecast_shape.transform("mean")
    frame["daily_change"] = frame.station_level - frame.groupby(groups).station_level.transform("mean")
    weather_groups = ["weather_zone", "month", "hour"]
    frame[[f"{name}_change" for name in WEATHER]] = frame[list(WEATHER)] - frame.groupby(weather_groups)[list(WEATHER)].transform("mean")
    frame["school_change"] = frame.school - frame.groupby("month").school.transform("mean")
    frame["band"] = np.searchsorted(BAND, frame.hour, side="right") - 1
    if not np.isfinite(frame[list(DESIGN)].to_numpy()).all():
        raise ValueError("Every calibration row requires complete weather and schedule context.")
    return frame


def ridge_matrix(rows):
    numeric = rows[list(ANOMALIES)].to_numpy().clip(-2.5, 2.5)
    bands = np.eye(len(BAND))[rows.band.to_numpy()]
    interactions = [numeric * bands[:, index:index + 1] for index in range(len(BAND))]
    weekend = numeric * rows.weekend.to_numpy()[:, None]
    return np.column_stack([numeric, *interactions, weekend])


def shrink_tails(prior, future):
    """Dampen unusually strong forecast shifts only when older blocks agree."""
    correction = pd.Series(0., index=future.index)
    split = prior.date.drop_duplicates().sort_values().iloc[prior.date.nunique() // 2]
    for (band, weekend), fit in prior.groupby(["band", "weekend"]):
        # Train a one-sided slope on the part beyond the historical threshold.
        threshold = max(.025, float(fit.shape_change.abs().quantile(.90)))
        magnitude = (fit.shape_change.abs() - threshold).clip(lower=0.)
        signal = -np.sign(fit.shape_change) * magnitude
        tail = magnitude.gt(0.)
        betas = []
        for block in (fit.date.lt(split), fit.date.ge(split)):
            sample = tail & block
            if fit.date[sample].nunique() < 10:
                break
            x, y = signal[sample].to_numpy(), fit.target[sample].clip(-.65, .65).to_numpy()
            betas.append(np.dot(x, y) / (np.dot(x, x) + 2.))
        if len(betas) != 2 or min(betas) <= 0.:
            continue
        use = future.band.eq(band) & future.weekend.eq(weekend)
        feature = future.loc[use, "shape_change"]
        correction.loc[use] = -min(min(betas), .45) * np.sign(feature) * (feature.abs() - threshold).clip(lower=0.)
    return correction.to_numpy()


def estimate(archive, future, origin, threads):
    if future.date.min() <= origin or "target" in future:
        raise ValueError("Only pre-origin errors may be used to fit a future correction.")
    prior = archive.loc[archive.date.le(origin) & archive.origin.lt(origin)].sort_values(["reliability", "origin"])
    prior = prior.drop_duplicates(KEY, keep="last")
    if not prior.date.gt(prior.origin).all():
        raise ValueError("Every residual must come from an earlier issued forecast.")
    support = int(prior.date.nunique())
    if support < 24:
        return {method: np.zeros(len(future)) for method in METHODS}, {"fit_dates": support, "fit_rows": len(prior)}
    age = (origin - prior.date).dt.days.to_numpy()
    weights = prior.reliability.to_numpy() * np.exp2(-age / 170.)
    response = prior.target.to_numpy().clip(-.65, .65)
    x, z = ridge_matrix(prior), ridge_matrix(future)
    wx = x * weights[:, None]
    gram = x.T @ wx
    beta = np.linalg.solve(gram + np.diag(.5 * gram.diagonal() + weights.sum() * .012), wx.T @ response)
    ridge = z @ beta
    settings = {"objective": "regression_l2", "verbosity": -1, "num_threads": threads, "seed": 2026,
                "learning_rate": .04, "num_leaves": 11, "min_data_in_leaf": 1500,
                "lambda_l2": 90., "max_bin": 63, "force_col_wise": True, "deterministic": True}
    training = lgb.Dataset(prior[list(DESIGN)], label=response, weight=weights)
    tree = lgb.train(settings, training, num_boost_round=110).predict(future[list(DESIGN)], num_threads=threads)
    adjustments = {}
    for method, values in zip(METHODS, (ridge, tree, shrink_tails(prior, future))):
        raw = pd.Series(values, index=future.index).clip(-.20, .20)
        centered = .45 * (raw - raw.groupby([future.date, future.station_id]).transform("mean"))
        worst = centered.abs().groupby([future.date, future.station_id]).transform("max")
        centered *= (.12 / worst.replace(0., 1.)).clip(upper=1.)
        floor = future.baseline / (-centered).where(centered < 0.)
        scale = floor.groupby([future.date, future.station_id]).transform("min").fillna(1.).clip(upper=1.)
        centered *= scale
        if (not np.isfinite(centered).all() or centered.groupby([future.date, future.station_id]).sum().abs().max() > 1e-10
                or (future.baseline + centered).min() < -1e-12):
            raise ValueError("The calibrated shape must preserve the original station-day log level.")
        adjustments[method] = centered.to_numpy()
    return adjustments, {"fit_dates": support, "fit_rows": len(prior), "latest_fit_date": str(prior.date.max().date())}


def archive_design(archive, settings, protected):
    designed = []
    for label, part in archive.groupby("label", sort=False):
        origin = part.origin.iloc[0]
        frame = design(part[[*KEY, "baseline"]], origin, settings, protected)
        remaining = part.y_log.loc[frame.index] - frame.baseline
        outcome = remaining - remaining.groupby([frame.date, frame.station_id]).transform("mean")
        designed.append(frame.assign(target=outcome, label=label, origin=origin, reliability=part.reliability.iloc[0]))
    return pd.concat(designed)


def evaluate(frame):
    groups = {"all": frame, "late_21_23": frame.loc[frame.hour.ge(21)]}
    return {group: {name: {**score(rows.y_log.to_numpy(), rows.baseline.to_numpy(),
                                 (rows.baseline + rows[name]).to_numpy(), rows.date),
                           "changed_rows": int(rows[name].ne(0.).sum())} for name in (*METHODS, *LATE_DIAGNOSTICS)}
            for group, rows in groups.items()}


def late_diagnostics(frame, tree):
    late = tree.where(frame.hour.ge(21), 0.)
    hours = late.loc[frame.hour.ge(21)]
    centered = hours - hours.groupby([frame.date, frame.station_id]).transform("mean")
    return dict(zip(LATE_DIAGNOSTICS, (late, centered.reindex(frame.index, fill_value=0.))))


def run(args):
    raw = pd.read_csv(args.team / "data/train.csv", parse_dates=["date"], usecols=[*KEY, "boardings"])
    history = raw.assign(observed=np.log1p(raw.pop("boardings")))
    protected, settings = protected_days(args.work_dir), context(args.work_dir, args.team)
    archive = latest_validation(args, history, protected)
    features = archive_design(archive, settings, protected)
    expected = json.loads((args.work_dir / "sincon_geomdan_morning_anchor_results.json").read_text())
    cases, report = {}, {}
    for label, frame in archive.groupby("label", sort=False):
        origin = frame.origin.iloc[0]
        base = np.sqrt(np.square(frame.y_log - frame.baseline).mean())
        if not np.isclose(base, expected[label]["all"]["relative_peak_share"]["candidate"], atol=1e-10, rtol=0.):
            raise ValueError(f"The latest incumbent replay differs in {label}.")
        selected = features.loc[features.label.eq(label)].drop(columns="target")
        deltas, support = estimate(features, selected, origin, args.threads)
        changes = {m: pd.Series(v, index=selected.index).reindex(frame.index, fill_value=0.) for m, v in deltas.items()}
        changes.update(late_diagnostics(frame, changes["conditional_tree"]))
        cases[label] = frame.assign(**changes)
        report[label] = {"origin": str(origin.date()), **support, **evaluate(cases[label])}
        print(label, report[label]["all"], flush=True)
    warm = pd.concat([cases[label].loc[cases[label].date.dt.month.eq(month)] for label, month in WARM])
    for label, rows in (("disjoint_warm", warm), ("disjoint_A_C", pd.concat([cases["A"], cases["C"]]))):
        if rows.duplicated(KEY).any():
            raise ValueError("Pooled results must use nonoverlapping station-hour keys.")
        report[label] = evaluate(rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "archive", "tabicl_oof", "retail_oof", "results"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    report = run(args)
    args.results.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for label in ("disjoint_A_C", "disjoint_warm"):
        print(label, report[label]["all"], flush=True)


if __name__ == "__main__":
    main()
