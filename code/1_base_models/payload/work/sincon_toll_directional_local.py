"""Test toll-specific direction and route-share shocks against local forecast errors."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sincon_district_mobility import CODE_CHANGES, station_districts
from sincon_fuel_shock import band, read_archive
from sincon_source_reweight import FIELDS, evaluate, fit_weights, prior_rows, rmsle
from sincon_station_state import KEY, WINDOWS
from sincon_weekly_variability import protected_days


MODES = ["car", "local_bus", "regional_bus", "walking"]
SLOTS = [0, 1, 2, 3]
DAY = KEY[:2]


def same_weekday_shock(values, ordinary):
    """Subtract the trailing median of 3+ ordinary occurrences of the same weekday."""
    observed = values.where(ordinary)
    peers = np.stack([observed.shift(7 * lag).to_numpy() for lag in range(1, 7)])
    supported = np.isfinite(peers).sum(axis=0) >= 3
    median = np.nanmedian(np.where(supported[None], peers, 0.), axis=0)
    shock = (values - median).where(supported)
    return shock.where(ordinary)


def exogenous(work, ordinary):
    toll = pd.read_csv(work / "sincon_tcs_daily.csv", parse_dates=["date"])
    grid = toll.pivot(index="date", columns="toll", values=["entries", "exits"])
    if grid.isna().any().any() or grid.le(0).any().any() or not grid.index.equals(ordinary.index):
        raise ValueError("Toll shocks require positive daily counts for both directions on a consecutive grid.")
    local_total = grid.entries + grid.exits
    route_share = np.log(local_total.div(local_total.sum(axis=1), axis=0))
    entry_share = grid.entries.div(grid.entries.sum(axis=1), axis=0)
    exit_share = grid.exits.div(grid.exits.sum(axis=1), axis=0)
    direction = np.log(entry_share / exit_share)
    traffic = same_weekday_shock(pd.concat([route_share.add_prefix("share_"), direction.add_prefix("direction_")], axis=1), ordinary)

    with ZipFile(work / "sincon_local_repro.zip") as archive:
        weather = pd.read_csv(archive.open("external_inputs/external/weather_hourly.csv"),
                              usecols=["date", "hour", "precip", "temp_feel", "snowfall"], parse_dates=["date"])
    weather = weather.loc[weather.hour.between(6, 23)].assign(slot=lambda x: band(x.hour))
    weather = weather.groupby(["date", "slot"])[["precip", "temp_feel", "snowfall"]].mean()
    weather[["precip", "snowfall"]] = np.log1p(weather[["precip", "snowfall"]])
    weather = weather.unstack("slot")

    monthly = []
    for path in sorted((work / "mobility_monthly").glob("*.csv.gz")):
        part = pd.read_csv(path, parse_dates=["date"], dtype={"district": str})
        part = part.loc[part.hour.between(6, 23)].assign(
            slot=lambda x: band(x.hour), district=lambda x: x.district.replace(CODE_CHANGES))
        monthly.append(part.groupby(["date", "slot", "mode", "district"]).arrivals.sum())
    mobility = pd.concat(monthly).unstack(["mode", "district", "slot"]).reindex(ordinary.index)
    if set(mobility.columns.get_level_values("mode")) != set(MODES):
        raise ValueError("Only the four existing non-rail mobile modes are supported.")
    total = mobility.T.groupby(level=["mode", "slot"]).sum().T
    city = np.log1p(total).rename(columns=lambda mode: f"city_{mode}", level="mode")
    share = np.log1p(mobility)
    share = share - np.log1p(total.reindex(columns=share.columns.droplevel("district"))).set_axis(share.columns, axis=1)
    return traffic, tuple(same_weekday_shock(frame.reindex(ordinary.index), ordinary) for frame in (weather, city, share))


def history_and_baselines(work, earlier, rows):
    _, archive = read_archive(work / "sincon_local_validation.csv.gz", earlier)
    # Replay source weights at each forecast's own origin for the A/B/C archives.
    # Older archived predictions predate the source-level components.
    archive["forecast"] = archive.local_stack
    rows["learned"] = rows.local_stack
    for fold, (cutoff, _) in WINDOWS.items():
        origin = pd.Timestamp(cutoff)
        selected = rows.fold.eq(fold)
        frame = rows.loc[selected]
        estimate = frame.local_stack.to_numpy() + frame[FIELDS].to_numpy() @ fit_weights(prior_rows(rows, origin))
        rows.loc[selected, "learned"] = estimate
        by_key = frame.set_index(KEY).assign(forecast=estimate).forecast
        old = archive.origin.eq(origin)
        archive.loc[old, "forecast"] = by_key.reindex(pd.MultiIndex.from_frame(archive.loc[old, KEY])).to_numpy()
    if archive.forecast.isna().any():
        raise ValueError("Source-weight archive could not be aligned to all of its folds.")
    return archive


def project_local(archive, target, traffic, context, districts, ordinary, events, origin):
    observed = archive.loc[archive.origin.lt(origin) & archive.date.le(origin)]
    observed = observed.sort_values("origin").drop_duplicates(KEY, keep="last")
    observed = observed.loc[ordinary.reindex(observed.date).to_numpy() & ~pd.MultiIndex.from_frame(observed[DAY]).isin(events)]
    observed = observed.assign(district=observed.station_id.map(districts), slot=band(observed.hour),
                               error=(observed.y_log - observed.forecast.clip(lower=0)).clip(-.45, .45))
    observed["error"] -= observed.groupby(["date", "slot"]).error.transform("mean")
    response = observed.groupby(["district", "slot", "date"]).error.mean()

    if target.date.min() <= origin or observed.date.max() > origin:
        raise ValueError("Fit rows must end at origin and precede every forecast row.")
    days = pd.DatetimeIndex(target.date.unique()).sort_values()
    index = pd.MultiIndex.from_product([districts.unique(), SLOTS, days], names=response.index.names)
    result = pd.Series(0., index=index)
    fitted = []
    weather, city, share = context
    for district in districts.unique():
        for slot in SLOTS:
            controls = pd.concat([weather.xs(slot, axis=1, level="slot"), city.xs(slot, axis=1, level="slot"),
                                  share.xs((district, slot), axis=1, level=["district", "slot"])], axis=1)
            y = response.loc[district, slot].rename("error")
            design = pd.concat([controls, traffic], axis=1)
            fit = design.join(y).dropna()
            eligible = design.reindex(days).dropna()
            if len(fit) < 60:
                continue
            mean = fit[design.columns].mean()
            scale = fit[design.columns].std().clip(lower=.002)
            x = ((fit[design.columns] - mean) / scale).to_numpy()
            z = np.clip(((eligible - mean) / scale).to_numpy(), -3., 3.)
            n = controls.shape[1]
            a, b = x[:, :n], z[:, :n]
            v, w = x[:, n:], z[:, n:]
            residualization = np.linalg.solve(a.T @ a + 15. * np.eye(n), a.T @ v)
            v, w = v - a @ residualization, w - b @ residualization
            labels = fit.error.to_numpy() - fit.error.mean()
            labels -= a @ np.linalg.solve(a.T @ a + 15. * np.eye(n), a.T @ labels)
            coef = np.linalg.solve(v.T @ v + 60. * np.eye(v.shape[1]), v.T @ labels)
            result.loc[(district, slot, eligible.index)] = np.clip(w @ coef, -.10, .10)
            fitted.append(len(fit))

    keys = pd.MultiIndex.from_arrays([target.station_id.map(districts), band(target.hour), target.date])
    shifts = target.assign(slot=band(target.hour), adjustment=result.reindex(keys).to_numpy())
    shifts.adjustment -= shifts.groupby(["date", "slot"]).adjustment.transform("mean")
    blocked = pd.MultiIndex.from_frame(shifts[DAY]).isin(events)
    return shifts.adjustment.where(~blocked, 0.).to_numpy() * .5, fitted


def run(work, team, earlier):
    started = perf_counter()
    holidays, events = protected_days(work)
    dates = pd.date_range("2025-01-01", "2026-10-31", name="date")
    ordinary = pd.Series(~dates.isin(holidays), index=dates)
    districts = station_districts(work / "sincon_station_addresses.csv", team)
    traffic, context = exogenous(work, ordinary)
    _, rows = evaluate(work, team, earlier)
    archive = history_and_baselines(work, earlier, rows)
    results = {}
    for fold, (cutoff, _) in WINDOWS.items():
        target = rows.loc[rows.fold.eq(fold)]
        forecast = target.learned.to_numpy()
        actual = target.y_log.to_numpy()
        delta, fitted = project_local(archive, target[KEY], traffic, context, districts, ordinary, events, pd.Timestamp(cutoff))
        scores = {}
        for name, subset in (("all", np.ones(len(target), dtype=bool)), ("late", target.hour.ge(21).to_numpy()),
                             ("ordinary", ordinary.reindex(target.date).to_numpy())):
            base = rmsle(actual[subset], forecast[subset])
            candidate = rmsle(actual[subset], forecast[subset] + delta[subset])
            scores[name] = {"learned": base, "with_local_tolls": candidate, "gain": base - candidate}
        results[fold] = {"scores": scores, "district_band_models": len(fitted), "fit_days_min": min(fitted),
                         "fit_days_max": max(fitted), "affected_rows": int(np.count_nonzero(delta)),
                         "rms_log_adjustment": float(np.sqrt(np.mean(delta ** 2)))}
        print(json.dumps({fold: results[fold]}, allow_nan=False), flush=True)
    results["elapsed_seconds"] = round(perf_counter() - started, 2)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team", type=Path)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    metrics = run(args.work_dir, args.team, args.earlier)
    (args.work_dir / "sincon_toll_directional_local_results.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
