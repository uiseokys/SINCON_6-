"""Replay fixed forecasts on complementary month-stratified halves of historical days."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_asiad_concert_spillover import proposals as asiad_proposals
from sincon_munhak_ordinary import WARM, home_dates
from sincon_munhak_spectator import HISTORY, earlier_profiles
from sincon_overlay_candidate import replay, trained_inputs
from sincon_station_profile_candidate import add_spectator_events, station_adjustments
from sincon_station_state import KEY
from sincon_xgb_conditioned import load_episodes


def daily_errors(args):
    episodes, raw, inputs, protected, calibrated, fallback = load_episodes(args)
    fitted = trained_inputs(raw, args.work_dir, args.earlier, *protected)
    games, _ = home_dates(args.work_dir)
    profiles = earlier_profiles(raw, protected, games)
    truth = raw.set_index(KEY).observed
    expected = json.loads((args.work_dir / "sincon_station_profile_spectator_results.json").read_text())
    expected_asiad = json.loads((args.work_dir / "sincon_asiad_concert_spillover_results.json").read_text())
    forecasts = {}
    for name in ("A", "B", "C", "summer", *(name for name, _ in WARM)):
        rows, origin, _ = episodes[name]
        rows = rows.reset_index(drop=True)
        keys = pd.MultiIndex.from_frame(rows[KEY])
        if not keys.is_unique or rows.date.min() <= origin or not np.allclose(truth.reindex(keys), rows.y_log, atol=1e-8, rtol=0):
            raise ValueError(f"Unexpected validation keys, labels or dates in {name}.")
        base, _ = replay(rows, origin, raw, inputs, protected, calibrated, fallback, fitted)
        future = base[[*KEY, "combined"]]
        history = raw.loc[raw.date.le(origin)]
        changes, _ = station_adjustments(history, future, origin, protected, games)
        changes, _ = add_spectator_events(changes, future, HISTORY, origin, profiles)
        spectator = (base.combined + changes["joint_with_spectator"]).clip(lower=0.)
        asiad, _ = asiad_proposals(base[KEY].assign(incumbent=spectator), history, origin, protected,
                                  include_placebos=False)
        asiad = asiad.three_quarters_excess
        forecasts[name] = pd.DataFrame({
            "date": base.date,
            "overlay": np.square(base.y_log - base.combined.clip(lower=0.)),
            "joint": np.square(base.y_log - (base.combined + changes["joint"]).clip(lower=0.)),
            "candidate": np.square(base.y_log - spectator),
            "asiad": np.square(base.y_log - asiad),
            "spectator_rows": changes["spectator"] != 0.,
            "asiad_rows": asiad.ne(spectator),
        })
    forecasts["disjoint_A_C"] = pd.concat([forecasts["A"], forecasts["C"]])
    forecasts["disjoint_warm"] = pd.concat([forecasts[name].loc[forecasts[name].date.dt.month.eq(month)] for name, month in WARM])
    if set(forecasts["A"].date) & set(forecasts["C"].date) or len({month for _, month in WARM}) != len(WARM):
        raise ValueError("The combined cases must each use disjoint dates.")
    for name, rows in forecasts.items():
        archived = expected[name]["joint_with_spectator"]
        scores = np.sqrt(rows[["overlay", "candidate"]].mean())
        if not np.allclose(scores, [archived["reference"], archived["candidate"]], atol=1e-10, rtol=0.):
            raise ValueError(f"Forecast replay differs from prior exact validation in {name}.")
        expected_final = expected_asiad[name]["all"]["rmsle"]["three_quarters_excess"]
        if not np.isclose(np.sqrt(rows.asiad.mean()), expected_final, atol=1e-10, rtol=0.):
            raise ValueError(f"The Asiad replay differs from prior exact validation in {name}.")
    return {name: rows.groupby("date").agg(overlay=("overlay", "sum"), joint=("joint", "sum"),
                                          candidate=("candidate", "sum"), asiad=("asiad", "sum"), count=("overlay", "size"),
                                          spectator_rows=("spectator_rows", "sum"), asiad_rows=("asiad_rows", "sum"))
            for name, rows in forecasts.items()}


def month_halves(dates, trials, seed):
    rng = np.random.default_rng(seed)
    months = dates.to_period("M")
    groups = [np.flatnonzero(months == month) for month in months.unique()]
    odd = [i for i, group in enumerate(groups) if len(group) % 2]
    bonus = np.zeros((trials, len(groups)), dtype=bool)
    if len(odd) >= 2:
        positions = np.asarray(odd)[np.argsort(rng.random((trials, len(odd))), axis=1)[:, :len(odd) // 2]]
        bonus[np.arange(trials)[:, None], positions] = True
    half = np.zeros((trials, len(dates)), dtype=bool)
    for index, group in enumerate(groups):
        ranking = np.argsort(rng.random((trials, len(group))), axis=1)
        half[:, group] = ranking < (len(group) // 2 + bonus[:, index, None])
    if not np.all(half.sum(axis=1) == len(dates) // 2):
        raise ValueError("Monthly date halves do not add up to half the dates.")
    return half


def gain_summary(gains):
    return {"mean": float(gains.mean()), "quantiles_025_50_975": np.quantile(gains, [.025, .5, .975]).tolist(),
            "improves_share": float(np.mean(gains > 1e-12)), "worsens_share": float(np.mean(gains < -1e-12)),
            "improves_by_0_00126_share": float(np.mean(gains >= .00126)),
            "improves_by_0_003901_share": float(np.mean(gains >= .003901))}


def summarize(days, trials, seed):
    columns = ["overlay", "joint", "candidate", "asiad"]
    date_sse = days[columns].to_numpy()
    n = days["count"].to_numpy()
    full = np.sqrt(date_sse.sum(axis=0) / n.sum())
    chosen = month_halves(days.index, trials, seed).astype(float)
    selected_sse = chosen @ date_sse
    selected_n = chosen @ n
    primary = np.sqrt(selected_sse / selected_n[:, None])
    complement = np.sqrt((date_sse.sum(axis=0) - selected_sse) / (n.sum() - selected_n[:, None]))
    rmsle = np.concatenate([primary, complement])
    event = days.spectator_rows.ne(0).to_numpy()
    event_count = chosen @ event
    event_count = np.concatenate([event_count, event.sum() - event_count]).astype(int)
    full_gain = rmsle[:, 0] - rmsle[:, 2]
    spectator_gain = rmsle[:, 1] - rmsle[:, 2]
    asiad_gain = rmsle[:, 2] - rmsle[:, 3]
    event_days = {}
    for index in np.flatnonzero(event):
        without_day = np.sqrt((date_sse.sum(axis=0) - date_sse[index]) / (n.sum() - n[index]))
        event_days[days.index[index].date().isoformat()] = {
            "changed_rows": int(days.spectator_rows.iloc[index]),
            "spectator_sse_gain": float(date_sse[index, 1] - date_sse[index, 2]),
            "candidate_gain_excluding_day": float(without_day[0] - without_day[2]),
            "spectator_gain_excluding_day": float(without_day[1] - without_day[2]),
        }
    asiad_days = {}
    asiad_event = days.asiad_rows.ne(0).to_numpy()
    asiad_count = chosen @ asiad_event
    asiad_count = np.concatenate([asiad_count, asiad_event.sum() - asiad_count]).astype(int)
    for index in np.flatnonzero(asiad_event):
        without_day = np.sqrt((date_sse.sum(axis=0) - date_sse[index]) / (n.sum() - n[index]))
        asiad_days[days.index[index].date().isoformat()] = {
            "changed_rows": int(days.asiad_rows.iloc[index]),
            "asiad_sse_gain": float(date_sse[index, 2] - date_sse[index, 3]),
            "asiad_gain_excluding_day": float(without_day[2] - without_day[3]),
        }
    gap = np.abs(primary[:, 0] - complement[:, 0])
    return {"days": len(days), "months": days.groupby(days.index.to_period("M")).size().rename(index=str).to_dict(),
            "full_rmsle": dict(zip(columns, full.tolist())),
            "overall_gain": gain_summary(full_gain), "ordinary_only_gain": gain_summary(rmsle[:, 0] - rmsle[:, 1]),
            "spectator_only_increment": gain_summary(spectator_gain),
            "overall_gain_with_asiad": gain_summary(rmsle[:, 0] - rmsle[:, 3]),
            "asiad_only_increment": gain_summary(asiad_gain),
            "both_halves_improve_with_asiad_share": float(np.mean((primary[:, 0] > primary[:, 3]) & (complement[:, 0] > complement[:, 3]))),
            "both_halves_improve_share": float(np.mean((primary[:, 0] > primary[:, 2]) & (complement[:, 0] > complement[:, 2]))),
            "absolute_baseline_half_gap_quantiles_025_50_975": np.quantile(gap, [.025, .5, .975]).tolist(),
            "baseline_absolute_half_gap_at_least_0_006254_share": float(np.mean(gap >= .006254)),
            "gain_by_spectator_days_in_half": {
                str(int(count)): {"half_share": float(np.mean(event_count == count)),
                                  "overall_gain": gain_summary(full_gain[event_count == count]),
                                  "spectator_only_increment": gain_summary(spectator_gain[event_count == count])}
                for count in np.unique(event_count)}, "spectator_days": event_days,
            "gain_by_asiad_days_in_half": {
                str(int(count)): {"half_share": float(np.mean(asiad_count == count)),
                                  "asiad_only_increment": gain_summary(asiad_gain[asiad_count == count])}
                for count in np.unique(asiad_count)}, "asiad_days": asiad_days}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "earlier", "recent", "extra_xgb", "june_xgb", "april_xgb", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    samples = daily_errors(args)
    selections = ("A", "B", "C", "summer", "disjoint_A_C", "disjoint_warm")
    result = {"trials": args.trials, "seed": args.seed,
              "kbo_timing_included": False,
              "samples": {name: summarize(samples[name], args.trials, args.seed) for name in selections}}
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
