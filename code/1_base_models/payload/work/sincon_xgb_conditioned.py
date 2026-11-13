"""Learn how much XGBoost contributes to separate forecast components."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from sincon_complementary_blend import historical_station_shift
from sincon_station_state import KEY, WINDOWS
from sincon_team_bias_proxy import earlier_team_errors
from sincon_weather_sensitivity import context, forecast, score
from sincon_weekly_variability import past_episodes, protected_days
from sincon_zoo_transfer import SEASON_WEIGHT, verified_archive


COMPONENTS = ["station_level", "day_level", "weekday_profile", "hour_residual"]
JUNE = pd.Timestamp("2025-06-30")
MIN_DATES = 24
UPPER_WEIGHT = .28
MODEL_NAMES = ["fixed_10pct", "scalar", "components", "seasonal_components"]


def read_pair(local, xgb, origin):
    base = pd.read_csv(local, parse_dates=["date"], usecols=[*KEY, "y_log", "local_stack"])
    other = pd.read_csv(xgb, parse_dates=["date", "origin"], usecols=[*KEY, "xgb", "origin"])
    paired = base.merge(other, on=KEY, how="outer", indicator=True, validate="one_to_one")
    if not paired.pop("_merge").eq("both").all() or not paired.origin.eq(origin).all():
        raise ValueError(f"Forecast archives do not align for {origin.date()}.")
    return paired


def design(rows, station, protected):
    holidays, events = protected
    ordinary = ~rows.date.isin(holidays) & ~pd.MultiIndex.from_frame(rows[KEY[:2]]).isin(events)
    mature = rows.station_id.map(station.first_date).eq(station.first_date.min())
    eligible = ordinary & mature
    observed_difference = rows.xgb - rows.local_stack
    difference = observed_difference.where(eligible)
    daily = difference.groupby([rows.station_id, rows.date]).transform("mean")
    station_level = daily.groupby(rows.station_id).transform("mean")
    shape = difference - daily
    weekday = shape.groupby([rows.station_id, rows.hour, rows.date.dt.dayofweek]).transform("mean")
    frame = rows.assign(eligible=eligible, station_level=station_level, day_level=daily - station_level,
                        weekday_profile=weekday, hour_residual=shape - weekday)
    frame[COMPONENTS] = frame[COMPONENTS].where(eligible, 0., axis=0).fillna(0.)
    if not np.allclose(frame[COMPONENTS].sum(axis=1), observed_difference.where(eligible, 0), atol=1e-12, rtol=0):
        raise ValueError("The additive components must reconstruct the original XGBoost difference.")
    return frame


def fit_prior(past, origin, seasonal_day=None, scalar=False):
    width = 1 if scalar else len(COMPONENTS)
    if past.date.nunique() < MIN_DATES:
        return np.full(width, SEASON_WEIGHT)
    predictors = past[COMPONENTS].to_numpy()
    if scalar:
        predictors = predictors.sum(axis=1, keepdims=True)
    residual = (past.y_log - past.reference).to_numpy().clip(-.55, .55)
    weights = past.reliability.to_numpy() * np.exp2(-(origin - past.date).dt.days.to_numpy() / 210.)
    if seasonal_day is not None:
        phase = (past.date.dt.dayofyear.to_numpy() - seasonal_day) * np.pi / 365.25
        weights *= .3 + .7 * np.exp(-3 * np.sin(phase)**2)
    root = np.sqrt(weights)
    x, y = predictors * root[:, None], residual * root
    ridge = np.diag(np.sqrt(.65 * np.square(x).sum(axis=0) + .3))
    result = lsq_linear(np.vstack([x, ridge]), np.r_[y, SEASON_WEIGHT * ridge.sum(axis=1)], bounds=(0, UPPER_WEIGHT))
    if not result.success:
        raise ValueError("The constrained least-squares fit did not converge.")
    return result.x


def proposals(target, archive, origin):
    fit = archive.loc[archive.origin.lt(origin) & archive.date.le(origin) & archive.eligible].copy()
    fit = fit.sort_values(["reliability", "origin"], ascending=[False, False]).drop_duplicates(KEY)
    x = target[COMPONENTS].to_numpy()
    reference = target.reference.to_numpy()
    scalar = fit_prior(fit, origin, scalar=True)
    common = fit_prior(fit, origin)
    seasonal = np.zeros(len(target))
    month_weights = {}
    months = target.date.dt.to_period("M")
    for month in sorted(months.unique()):
        selected = months.eq(month)
        middle = target.loc[selected, "date"].sort_values().iloc[selected.sum() // 2].dayofyear
        coefficients = fit_prior(fit, origin, seasonal_day=middle)
        month_weights[str(month)] = dict(zip(COMPONENTS, coefficients.tolist()))
        seasonal[selected] = x[selected] @ coefficients
    final = target[[*KEY, "y_log"]].assign(
        fixed_10pct=reference + SEASON_WEIGHT * x.sum(axis=1),
        scalar=reference + scalar[0] * x.sum(axis=1),
        components=reference + x @ common,
        seasonal_components=reference + seasonal,
    )
    return final, {"days_used": fit.date.nunique(), "scalar_weight": float(scalar[0]),
                   "weights": dict(zip(COMPONENTS, common.tolist())), "weights_by_month": month_weights}


def metrics(rows, metadata):
    mature = rows.station_id.map(metadata.first_date).eq(metadata.first_date.min()).to_numpy()
    late = rows.hour.ge(21).to_numpy()
    groups = {"all": np.ones(len(rows), dtype=bool), "mature_68": mature, "late_21_23": late}
    return {name: {method: score(rows.y_log.to_numpy()[keep], rows.fixed_10pct.to_numpy()[keep],
                                 rows[method].to_numpy()[keep], rows.date[keep])
                   for method in MODEL_NAMES[1:]} for name, keep in groups.items()}


def load_episodes(args):
    work = args.work_dir
    inputs = context(work, args.team)
    metadata = inputs[0]
    protected = protected_days(work)
    raw = pd.read_csv(args.team / "data/train.csv", parse_dates=["date"], usecols=[*KEY, "boardings"])
    raw["observed"] = np.log1p(raw.pop("boardings"))
    team, _ = verified_archive(work, args.team)
    cols = [*KEY, "y_log", "team"]
    team_main = pd.read_csv(work / "sincon_local_validation.csv.gz", usecols=[*cols, "fold"], parse_dates=["date"])
    team_summer = pd.read_csv(work / "sincon_summer_replay_validation.csv.gz", usecols=cols, parse_dates=["date"])
    origins = {name: pd.Timestamp(span[0]) for name, span in WINDOWS.items()}
    calibrated = pd.concat([team_main.assign(origin=team_main.fold.map(origins)), team_summer.assign(origin=JUNE)], ignore_index=True)
    calibrated["station_age"] = (calibrated.origin - calibrated.station_id.map(metadata.first_date)).dt.days
    _, fallback_archive = past_episodes(work / "sincon_local_validation.csv.gz", args.earlier, *protected)
    episodes = {label: (team.loc[team.fold.eq(label)].copy(), cutoff, 1.) for label, cutoff in origins.items()}
    episodes["summer"] = (read_pair(work / "sincon_summer_replay_validation.csv.gz", args.june_xgb, JUNE), JUNE, 1.)
    for name in ("2025-07-31", "2025-08-31", "2025-09-30", "2026-03-31", "2026-04-30"):
        origin = pd.Timestamp(name)
        file = f"origin_{name}.csv.gz"
        local = (args.earlier if origin.year == 2025 else args.recent) / file
        xgb = args.april_xgb if name == "2026-04-30" else args.extra_xgb / file
        episodes[name] = (read_pair(local, xgb, origin), origin, .55)
    return episodes, raw, inputs, protected, calibrated, fallback_archive


def prepare_episode(rows, origin, raw, inputs, protected, calibrated, fallback_archive):
    frame = design(rows, inputs[0], protected)
    weather = forecast(raw.loc[raw.date.le(origin)], rows[[*KEY, "local_stack"]], origin, inputs)
    late = np.where(frame.eligible & frame.hour.ge(21), .45 * weather, 0.)
    offset, days = earlier_team_errors(calibrated, origin)
    if days >= 60:
        fallback, _ = historical_station_shift(fallback_archive, rows, origin)
        shift = rows.station_id.map(offset).fillna(pd.Series(fallback, index=rows.index)).to_numpy()
    else:
        shift = 0.
    return frame.assign(reference=rows.local_stack.to_numpy() + late + shift, station_offset=shift), days


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "earlier", "recent", "extra_xgb", "june_xgb", "april_xgb"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    work = args.work_dir
    episodes, raw, inputs, protected, calibrated, fallback_archive = load_episodes(args)
    truth = raw.set_index(KEY).observed
    metadata = inputs[0]

    reports, previous, scored = {}, [], {}
    for label, (rows, origin, reliability) in sorted(episodes.items(), key=lambda item: item[1][1]):
        rows = rows.reset_index(drop=True)
        index = pd.MultiIndex.from_frame(rows[KEY])
        if (not index.is_unique or rows.date.min() <= origin or not np.allclose(truth.reindex(index), rows.y_log, atol=1e-8, rtol=0)
                or not np.isfinite(rows[["xgb", "local_stack"]]).all().all()):
            raise ValueError(f"Invalid pre-origin forecast and validation keys: {label}")
        frame, days = prepare_episode(rows, origin, raw, inputs, protected, calibrated, fallback_archive)
        fit_columns = [*KEY, "origin", "y_log", "eligible", "reference", *COMPONENTS]
        past = pd.concat(previous, ignore_index=True) if previous else frame[fit_columns].iloc[:0].assign(reliability=1.)
        scored[label], details = proposals(frame, past, origin)
        previous.append(frame[fit_columns].assign(reliability=reliability))
        reports[label] = {"origin": str(origin.date()), "calibration_days": days, **details, **metrics(scored[label], metadata)}

    disjoint_warm = pd.concat([scored[name].loc[scored[name].date.dt.month.eq(month)] for name, month in
                              (("2025-07-31", 8), ("2025-08-31", 9), ("2025-09-30", 10))], ignore_index=True)
    disjoint_ac = pd.concat([scored[name] for name in ("A", "C")], ignore_index=True)
    for label, sample in (("disjoint_warm", disjoint_warm), ("disjoint_A_C", disjoint_ac)):
        if sample.duplicated(KEY).any():
            raise ValueError(f"Pooled validation contains repeated dates/keys in {label}.")
        reports[label] = {"days": sample.date.nunique(), **metrics(sample, metadata)}
    path = work / "sincon_xgb_conditioned_results.json"
    path.write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: part["all"] for name, part in reports.items()}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
