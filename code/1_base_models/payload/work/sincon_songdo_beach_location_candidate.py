"""Remove the late demand attributed to the wrong Songdo Beach Festival station."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_munhak_cancellation_candidate import read_response
from sincon_songdo_beach_location_lgb import FINAL_DAYS, KEY, LATE, WRONG_STATION
from sincon_songdo_schedule import save_submission


def evaluate(args):
    grid = pd.read_csv(args.team / "data/test.csv", usecols=[*KEY, "example_id"])
    incumbent = pd.read_csv(args.baseline, float_precision="round_trip")
    if (len(grid) != 157_194 or not grid.example_id.is_unique or grid.duplicated(KEY).any()
            or list(incumbent.columns) != ["example_id", "boardings"] or not incumbent.example_id.equals(grid.example_id)
            or not np.isfinite(incumbent.boardings).all() or incumbent.boardings.lt(0.).any()):
        raise ValueError("The baseline and official test grid must share all valid IDs in order.")
    days = grid.station_id.eq(WRONG_STATION) & grid.date.isin(FINAL_DAYS)
    target = grid.loc[days].reset_index(drop=True)
    tree_report = json.loads(args.tree.read_text())
    tree = pd.DataFrame(tree_report["final_changed_scope"])
    if (len(target) != 144 or len(tree) != len(target) or not tree.example_id.is_unique
            or set(tree.example_id) != set(target.example_id)
            or tree_report["final_archive_max_absolute_log_difference"] > 1e-8):
        raise ValueError("Expected exact three-seed tree responses for 3139 on eight dates.")
    tree = tree.set_index("example_id").reindex(target.example_id).reset_index()
    archived = pd.read_csv(args.team / "ensemble_parts.csv").set_index("example_id").reindex(target.example_id)
    if not tree[KEY].equals(target[KEY]) or not np.allclose(archived.log_lgb, tree.original_log_lgb, atol=1e-8, rtol=0.):
        raise ValueError("The tree scenario must match archived forecasts and official keys.")
    neural = [read_response(path, target, ["archived_nn", "original_nn", "station_delta", "combined_delta"])
              for path in args.neural]
    for frame in neural:
        if (not np.allclose(archived.log_nn, frame.archived_nn, atol=1e-8, rtol=0.)
                or not np.array_equal(frame.station_delta, frame.combined_delta)):
            raise ValueError("The relocated festival must preserve the city event count.")
    tree_effect = tree.delta_log.to_numpy()
    nn_effect = np.mean([frame.station_delta.to_numpy() for frame in neural], axis=0)
    late = target.hour.isin(LATE).to_numpy()
    if late.sum() != 24 or not np.isfinite(np.column_stack((tree_effect, nn_effect))).all():
        raise ValueError("The two models must cover the complete late-hour venue adjustment.")
    delta = .5 * (tree_effect + nn_effect)
    if (tree_effect[late] >= 0.).any() or (delta[late] >= 0.).any():
        raise ValueError("Removing the incorrect event flag did not lower every late-hour forecast.")
    old = incumbent.boardings.loc[days].to_numpy()[late]
    new = np.expm1((np.log1p(old) + delta[late]).clip(min=0.))
    forecasts = incumbent.boardings.to_numpy(copy=True)
    selected = days & grid.hour.isin(LATE)
    forecasts[selected] = new
    rows = target.loc[late].assign(previous=old, corrected=new, lgb_log_delta=tree_effect[late], nn_log_delta=nn_effect[late])
    if not np.array_equal(forecasts[~selected], incumbent.boardings.to_numpy()[~selected]) or len(rows) != 24:
        raise ValueError("Only 3139 at 21–23 on the eight wrong-venue dates may change.")
    if args.submission:
        save_submission(args.submission, incumbent, grid, forecasts)
    return {"baseline": str(args.baseline.resolve()), "rows": len(grid), "changed_rows": int(np.count_nonzero(new != old)),
            "daily_late_totals": rows.groupby("date")[["previous", "corrected"]].sum().to_dict(orient="index"),
            "historical_venue_audit": tree_report["historical_scores"], "sources": tree_report["sources"],
            "neural_original_archive_rmse": [float(np.sqrt(np.square(frame.original_nn - frame.archived_nn).mean()))
                                             for frame in neural], "hourly": rows.to_dict(orient="records"),
            "submission": str(args.submission.resolve()) if args.submission else None,
            "sha256": hashlib.sha256(args.submission.read_bytes()).hexdigest() if args.submission else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "baseline", "tree", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--neural", nargs="+", required=True, type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    result = evaluate(args)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("changed_rows", "daily_late_totals", "neural_original_archive_rmse", "sha256")}), flush=True)


if __name__ == "__main__":
    main()
