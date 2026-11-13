"""Apply the validated vector-XGBoost intraday anomaly to the final guarded forecast."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sincon_archived_intraday_anomalies import MAX_SHIFT, WEIGHT, proposal, vector_inputs, vector_signal
from sincon_network_level_audit import STABLE_EXCLUSIONS, ordinary
from sincon_songdo_schedule import save_submission
from sincon_station_state import KEY
from sincon_tabicl_day_weight import exclude_retail
from sincon_tabicl_final_delta import aligned_counts, digest, protect_previous_layers


ORIGIN = pd.Timestamp("2026-06-30")
LAST = pd.Timestamp("2026-10-31")
CONDITIONAL_SHA256 = "5496cced5a69b990e0237fa3ba0a109def8415c2537be87349a485e2c4ee6ec6"
CANCELLATION_SHA256 = "cc49c30814dec15c56c3557b4ff4827b75b65e47f79842b8c73c0fc914c9ea93"
CANCELLATION_BASE = "submission_local_stack_level_station_events_xgb_weather_geomdan_late.csv"
CANCELLATION_STEPS = (
    ("submission_station_profiles_spectator.csv", "submission_station_profiles_cancelled.csv", 72),
    ("submission_station_profiles_cancelled.csv", "submission_station_profiles_event_audit.csv", 24),
)


def verified_results(work):
    reports = [json.loads((work / name).read_text()) for name in
               ("sincon_archived_intraday_anomalies_results.json", "sincon_archived_intraday_anomalies_warm_results.json")]
    labels = (("A", "B", "C", "disjoint_A_C", "strict_summer_mature_68"),
              ("2025-07-31", "2025-08-31", "2025-09-30", "disjoint_warm"))
    metrics = {}
    for report, cases in zip(reports, labels):
        if report["frozen_weight"] != WEIGHT or report["max_absolute_log_change"] != MAX_SHIFT:
            raise ValueError("The final weight and cap must match the chronological experiment.")
        for case in cases:
            result = report["cases"][case]["all"]["vector_xgb"]
            if result["gain"] <= 0. or result["week_block_gain_interval"][0] <= 0.:
                raise ValueError(f"The fixed intraday anomaly did not replicate in {case}.")
            metrics[case] = result
    return metrics


def cancellation_rows(work, grid, predictions):
    base = np.log1p(aligned_counts(work / CANCELLATION_BASE, grid))
    expected = np.zeros(len(grid))
    for before, after, count in CANCELLATION_STEPS:
        step = np.log1p(aligned_counts(work / after, grid)) - np.log1p(aligned_counts(work / before, grid))
        changed = step.abs().gt(1e-9).to_numpy()
        if changed.sum() != count or np.any(changed & (expected != 0.)):
            raise ValueError("The cancellation and relocation deltas must match their verified edit sets.")
        expected[changed] = step[changed]
    actual = np.log1p(predictions) - base
    if not np.allclose(actual, expected, atol=1e-12, rtol=0.) or actual.abs().gt(1e-9).sum() != 96:
        raise ValueError("The alternate incumbent must reproduce exactly 96 verified schedule edits.")
    return expected != 0.


def raw_signal(args, grid, conditional):
    if args.raw_signal:
        raw = pd.read_csv(args.raw_signal, parse_dates=["date"], float_precision="round_trip")
        key = pd.MultiIndex.from_frame(raw[[*KEY, "fold"]])
        expected = pd.MultiIndex.from_frame(grid[KEY].assign(fold="final"))
        if (not key.is_unique or not key.sort_values().equals(expected.sort_values())
                or not np.isfinite(raw.raw_delta).all()):
            raise ValueError("The cached vector signal must match every final station-hour exactly once.")
        return {"vector_xgb": raw.raw_delta.set_axis(key)}
    model = vector_inputs(args)
    if model[0].loc[model[0].date.gt(ORIGIN), "boardings"].notna().any():
        raise ValueError("Vector forecasting accepts no observed test-period boardings.")
    reference = pd.Series(np.log1p(conditional.to_numpy()), index=pd.MultiIndex.from_frame(grid[KEY]))
    return vector_signal(model, ORIGIN, "final", reference)


def run(args):
    started = perf_counter()
    work = args.work_dir
    metrics = verified_results(work)
    guarded = json.loads((work / "sincon_tabicl_day_weight_results.json").read_text())
    if digest(args.incumbent) != guarded["export"]["sha256"] or digest(args.conditional) != CONDITIONAL_SHA256:
        raise ValueError("The guarded incumbent and conditional source must match their recorded forecasts.")
    if args.submission.exists() or args.report.exists():
        raise FileExistsError("Choose new output paths for the forecast and its report.")
    grid = pd.read_csv(args.team / "data/test.csv", parse_dates=["date"], usecols=["example_id", *KEY])
    if (len(grid) != 157_194 or not grid.example_id.is_unique or grid.duplicated(KEY).any()
            or grid.date.min() != ORIGIN + pd.Timedelta(days=1) or grid.date.max() != LAST
            or not grid.groupby(KEY[:2]).size().eq(18).all()):
        raise ValueError("Use the complete official four-month test grid.")
    guarded_counts = aligned_counts(args.incumbent, grid)
    conditional = aligned_counts(args.conditional, grid)
    protected, layers = protect_previous_layers(grid, guarded_counts, work, args.team)
    incumbent = guarded_counts
    cancellation = np.zeros(len(grid), dtype=bool)
    if args.baseline:
        if digest(args.baseline) != CANCELLATION_SHA256:
            raise ValueError("The baseline override must match the verified 0.14774 file.")
        incumbent = aligned_counts(args.baseline, grid)
        cancellation = cancellation_rows(work, grid, incumbent)
        protected = protected[0], protected[1].union(pd.MultiIndex.from_frame(grid.loc[cancellation, KEY[:2]]))
    known_events = grid.station_id.eq(3754) & grid.date.eq("2026-10-17")
    eligible = ordinary(grid, protected) & ~grid.station_id.isin(STABLE_EXCLUSIONS) & ~exclude_retail(grid) & ~known_events
    signal = raw_signal(args, grid, conditional)
    frame = grid.assign(label="final", eligible=eligible, baseline=np.log1p(incumbent))
    delta = proposal(frame, signal)["vector_xgb"]
    changed = delta != 0.
    final = incumbent.to_numpy(copy=True)
    final[changed] = np.expm1(frame.baseline.to_numpy()[changed] + delta[changed])
    retained = pd.DataFrame(layers).any(axis=1)
    if (not changed.any() or np.any(changed & ~eligible) or np.any(changed & (retained | cancellation))
            or np.abs(delta).max() > MAX_SHIFT + 1e-12 or not np.isfinite(final).all() or (final < 0.).any()):
        raise ValueError("Only eligible rows may receive a bounded, nonnegative vector forecast.")
    save_submission(args.submission, grid[["example_id"]], grid, final)
    verified = aligned_counts(args.submission, grid)
    if not np.array_equal(final, verified.to_numpy()) or not np.array_equal(verified[~changed], incumbent[~changed]):
        raise ValueError("The exported forecast does not reproduce its predictions and protected rows.")
    return {"submission": str(args.submission.resolve()), "sha256": digest(args.submission), "rows": len(grid),
            "weight": WEIGHT, "changed_rows": int(verified.ne(incumbent).sum()),
            "changed_station_days": len(grid.loc[changed, KEY[:2]].drop_duplicates()),
            "prior_event_rows_preserved": int(retained.sum()), "bucheon_oct17_rows_preserved": int(known_events.sum()),
            "baseline_cancellation_rows_preserved": int(cancellation.sum()),
            "max_absolute_log_change": float(np.abs(delta).max()),
            "rms_log_change": float(np.sqrt(np.square(delta).mean())),
            "incumbent_sha256": digest(args.baseline) if args.baseline else guarded["export"]["sha256"],
            "conditional_sha256": CONDITIONAL_SHA256,
            "raw_signal_sha256": digest(args.raw_signal) if args.raw_signal else None,
            "seconds": round(perf_counter() - started, 2), "validation": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "external", "incumbent", "conditional", "submission", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--baseline", type=Path, help="Apply the vector correction to the checked 0.14774 incumbent.")
    parser.add_argument("--raw-signal", type=Path, help="Reuse a keyed raw vector forecast from the same official inputs.")
    args = parser.parse_args()
    result = run(args)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "validation"}))


if __name__ == "__main__":
    main()
