"""Test a fixed TabICL daily deviation blend on the latest held-out forecasts."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sincon_apply_final_calibration import PUBLIC_LOG_LEVEL
from sincon_asiad_concert_spillover import proposals as asiad_proposals
from sincon_munhak_ordinary import WARM, home_dates
from sincon_munhak_spectator import HISTORY, earlier_profiles
from sincon_network_level_audit import STABLE_EXCLUSIONS, ordinary
from sincon_overlay_candidate import replay, trained_inputs
from sincon_spectator_month_split import gain_summary, month_halves
from sincon_station_profile_candidate import add_spectator_events, station_adjustments
from sincon_station_state import KEY
from sincon_weather_sensitivity import score
from sincon_xgb_conditioned import load_episodes


DAY = KEY[:2]
WEIGHT = .10
DISAGREEMENT_CAP = .30


def deviations(future, protected):
    if set(future.columns) != {*KEY, "incumbent", "alternative"}:
        raise ValueError("Computing deviations requires unlabeled forecast rows.")
    ordinary_scope = ordinary(future, protected) & ~future.station_id.isin(STABLE_EXCLUSIONS)
    sample = future.loc[ordinary_scope].copy()
    sample["difference"] = sample.alternative - sample.incumbent
    by_day = sample.groupby(DAY).agg(disagreement=("difference", "mean"), minimum=("incumbent", "min")).reset_index()
    by_day["month"] = by_day.date.dt.to_period("M")
    by_day["weekday"] = by_day.date.dt.day_of_week
    cells = ["station_id", "month", "weekday"]
    by_day["capped"] = by_day.disagreement.clip(-DISAGREEMENT_CAP, DISAGREEMENT_CAP)
    centered = by_day.capped - by_day.groupby(cells).capped.transform("mean")
    by_day["offset"] = WEIGHT * centered
    # If a group has a near-zero hour, shrink the entire group together to retain its mean.
    by_day["scale"] = np.divide(by_day.minimum, -by_day.offset, out=np.ones(len(by_day)), where=by_day.offset.lt(0.)).clip(0., 1.)
    by_day["offset"] *= by_day.groupby(cells).scale.transform("min")
    if not np.allclose(by_day.groupby(cells).offset.sum(), 0., atol=1e-12, rtol=0.):
        raise ValueError("Daily deviations changed a station's calendar profile.")
    shifts = by_day.set_index(DAY).offset.reindex(pd.MultiIndex.from_frame(future[DAY]), fill_value=0.).to_numpy()
    candidate = future.incumbent.to_numpy() + shifts
    if (not np.isfinite(candidate).all() or (candidate < -1e-12).any()
            or np.any(shifts[~ordinary_scope]) or np.max(np.abs(shifts)) > 2 * WEIGHT * DISAGREEMENT_CAP + 1e-12):
        raise ValueError("The blend must preserve protected rows and valid log predictions.")
    return candidate.clip(min=0.)


def metrics(frame):
    if frame.duplicated(KEY).any():
        raise ValueError("The comparison cannot score overlapping date-station-hours.")
    base = frame.forecast.to_numpy()
    candidate = frame.candidate.to_numpy()
    overall = score(frame.y_log.to_numpy(), base, candidate, frame.date)
    late = frame.hour.ge(21)
    daily = frame[["date"]].assign(base=(frame.y_log - base)**2, blend=(frame.y_log - candidate)**2,
                                   rows=1).groupby("date").sum()
    mask = month_halves(daily.index, 10_000, 116).astype(float)
    errors = daily[["base", "blend"]].to_numpy()
    selected = mask @ errors
    counts = (mask @ daily.rows.to_numpy())[:, None]
    left = np.sqrt(selected / counts)
    right = np.sqrt((errors.sum(axis=0) - selected) / (daily.rows.sum() - counts))
    gains = np.column_stack((left[:, 0] - left[:, 1], right[:, 0] - right[:, 1]))
    return {
        "dates": int(frame.date.nunique()),
        "changed_rows": int(frame.forecast.ne(frame.candidate).sum()),
        "all": {**overall, "gain": overall["reference"] - overall["candidate"]},
        "late": score(frame.y_log.to_numpy()[late], base[late], candidate[late], frame.date[late]),
        "month_halves": {**gain_summary(gains.ravel()),
                         "both_halves_improve_share": float(np.mean(np.all(gains > 0., axis=1)))},
    }


def daily_archive(args):
    path = args.daily_archive or args.work_dir / "sincon_tabicl_daily_validation.csv.gz"
    return pd.read_csv(path, usecols=[*KEY, "fold", "y_log", "tabicl_all"], parse_dates=["date"])


def replay_warm(args, episodes, archived, labels=tuple(label for label, _ in WARM)):
    from tabicl import TabICLRegressor

    from sincon_joint_candidate import setup
    from sincon_tabicl_experiment import forecast, station_days

    work = args.work_dir
    data, _, _, _, stations, _, _ = setup(args.team, args.external, work / "sincon_kbo_game_data.csv",
                                          work / "mobility_monthly", work / "sincon_station_addresses.csv", 3)
    table, daily = station_days(data)
    model = TabICLRegressor(model_path=args.checkpoint, allow_auto_download=False, n_estimators=2,
                            device="cpu", n_jobs=3, random_state=2026)
    parts = [archived]
    for label in labels:
        rows, origin, _ = episodes[label]
        days = forecast(table, daily, origin, rows.date.max(), stations.index, model, 1).drop(columns="band")
        predictions = rows[[*KEY, "y_log"]].merge(days, on=DAY, how="left", validate="many_to_one")
        parts.append(predictions.rename(columns={"tabicl": "tabicl_all"}).assign(fold=label))
        print(f"Replayed TabICL for {label} through {rows.date.max().date()}.", flush=True)
    return pd.concat(parts, ignore_index=True)


def evaluate(args):
    episodes, raw, inputs, protected, calibrated, fallback = load_episodes(args)
    fitted = trained_inputs(raw, args.work_dir, args.earlier, *protected)
    games, _ = home_dates(args.work_dir)
    profiles = earlier_profiles(raw, protected, games)
    truth = raw.set_index(KEY).observed
    archived = daily_archive(args)
    if args.checkpoint:
        archived = replay_warm(args, episodes, archived)
    expected = json.loads((args.work_dir / "sincon_network_level_audit_results.json").read_text())
    results, replays = {}, {}
    for label in archived.fold.unique():
        rows, origin, _ = episodes[label]
        rows = rows.reset_index(drop=True)
        keys = pd.MultiIndex.from_frame(rows[KEY])
        tabicl = archived.loc[archived.fold.eq(label)].set_index(KEY).reindex(keys)
        if (not keys.is_unique or rows.date.min() <= origin or not np.allclose(truth.reindex(keys), rows.y_log, atol=1e-8, rtol=0.)
                or not np.allclose(tabicl.y_log, rows.y_log, atol=1e-8, rtol=0.)
                or not np.isfinite(tabicl.tabicl_all).all()):
            raise ValueError(f"The forecasts and labels must match the full chronological grid in {label}.")
        base, _ = replay(rows, origin, raw, inputs, protected, calibrated, fallback, fitted)
        query = base[[*KEY, "combined"]]
        history = raw.loc[raw.date.le(origin)]
        changes, _ = station_adjustments(history, query, origin, protected, games)
        changes, _ = add_spectator_events(changes, query, HISTORY, origin, profiles)
        spectator = base.combined + changes["joint_with_spectator"]
        asiad, _ = asiad_proposals(base[KEY].assign(incumbent=spectator), history, origin, protected, include_placebos=False)
        normal = ordinary(base, protected) & ~base.station_id.isin(STABLE_EXCLUSIONS)
        incumbent = (asiad.three_quarters_excess + np.where(normal, 0., PUBLIC_LOG_LEVEL)).clip(lower=0.)
        if not np.isclose(np.sqrt(np.square(rows.y_log - incumbent).mean()),
                          expected[label]["without_public_level"]["candidate"], atol=1e-10, rtol=0.):
            raise ValueError(f"Exact chronological level-neutral reference failed in {label}.")
        future = base[KEY].assign(incumbent=incumbent, alternative=tabicl.tabicl_all.to_numpy())
        replays[label] = base[[*KEY, "y_log"]].assign(forecast=incumbent, candidate=deviations(future, protected))
        results[label] = metrics(replays[label])
    results["disjoint_A_C"] = metrics(pd.concat([replays["A"], replays["C"]], ignore_index=True))
    if all(label in replays for label, _ in WARM):
        warm = [replays[label].loc[replays[label].date.dt.month.eq(month)] for label, month in WARM]
        results["disjoint_warm"] = metrics(pd.concat(warm, ignore_index=True))
    if args.oof_output:
        pd.concat([
            frame[KEY].assign(label=label, origin=episodes[label][1], log_delta=frame.candidate - frame.forecast)
            for label, frame in replays.items()
        ], ignore_index=True).to_pickle(args.oof_output)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "earlier", "recent", "extra_xgb", "june_xgb", "april_xgb", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--daily-archive", type=Path)
    parser.add_argument("--oof-output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--external", type=Path)
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.external):
        parser.error("Warm replay needs both --checkpoint and --external.")
    started = perf_counter()
    results = evaluate(args)
    results["settings"] = {"weight": WEIGHT, "disagreement_cap": DISAGREEMENT_CAP,
                           "seconds": round(perf_counter() - started, 2)}
    args.report.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps({label: result["all"] for label, result in results.items() if "all" in result}))


if __name__ == "__main__":
    main()
