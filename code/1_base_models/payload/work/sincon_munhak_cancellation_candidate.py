"""Apply paired model responses to four verified concert cancellations."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_munhak_cancelled_nn import CANCELLED, KEY, STATION
from sincon_songdo_schedule import save_submission


SOURCES = {
    "objet_both_days": "https://btobcompany.kr/notice/?bmode=view&category=&idx=174155451&q=YToxOntzOjEyOiJrZXl3b3JkX3R5cGUiO3M6MzoiYWxsIjt9&t=board",
    "objet_september_27": "https://weverse.io/qwer/notice/39354",
    "ghostival_ticket_seller": "https://nol.yanolja.com/ticket/products/26014180",
    "ghostival_cancellation_report": "https://www.asiatoday.co.kr/kn/view.php?key=20261021010007097",
    "kbo_september_games": "https://koreabaseball.com/Record/Crowd/GraphDaily.aspx",
    "kbo_october_postseason": "https://www.koreabaseball.com/Record/TeamRank/TeamRankDaily.aspx?gameDate=20261012",
}


def read_response(path, target, columns, tree=False):
    if tree:
        stored = json.loads(path.read_text())
        if stored["seeds"] != [0, 1, 2] or stored["archive_max_absolute_difference"] > 1e-8:
            raise ValueError("The original three-seed LightGBM forecasts must be reproduced.")
        data = pd.DataFrame(stored["station_only"])
    else:
        data = pd.read_csv(path, dtype={"date": str}, float_precision="round_trip")
    if not data.example_id.is_unique or set(data.example_id) != set(target.example_id):
        raise ValueError(f"Wrong IDs in {path}.")
    data = data.set_index("example_id").reindex(target.example_id).reset_index()
    if not data[KEY].equals(target[KEY]) or not np.isfinite(data[columns].to_numpy()).all():
        raise ValueError(f"Forecast scenario keys or outputs differ in {path}.")
    return data


def run(args):
    grid = pd.read_csv(args.team / "data/test.csv", usecols=[*KEY, "example_id"])
    reference = pd.read_csv(args.baseline, float_precision="round_trip")
    if (len(grid) != 157_194 or not grid.example_id.is_unique or grid.duplicated(KEY).any()
            or not grid.example_id.equals(reference.example_id) or list(reference.columns) != ["example_id", "boardings"]
            or reference.boardings.lt(0.).any() or not np.isfinite(reference.boardings).all()):
        raise ValueError("The incumbent and full official grid must have the same test IDs and valid counts.")
    scope = grid.station_id.eq(STATION) & grid.date.isin(CANCELLED)
    target = grid.loc[scope].reset_index(drop=True)
    if len(target) != len(CANCELLED) * 18:
        raise ValueError("Expected one station's eighteen hours on each cancelled event day.")
    lgb = read_response(args.tree, target, ["original_log_lgb", "station_and_city_delta_log"], tree=True)
    original = pd.read_csv(args.team / "ensemble_parts.csv").set_index("example_id").reindex(target.example_id)
    if not np.allclose(original.log_lgb, lgb.original_log_lgb, atol=1e-8, rtol=0.):
        raise ValueError("Tree responses must be anchored to the archived ensemble member.")
    neural = [read_response(path, target, ["archived_nn", "original_nn", "combined_delta"]) for path in args.neural]
    for frame in neural:
        if not np.allclose(original.log_nn, frame.archived_nn, atol=1e-8, rtol=0.):
            raise ValueError("Neural responses must reference the archived ensemble member.")
    nn_effect = np.mean([frame.combined_delta.to_numpy() for frame in neural], axis=0)
    lgb_effect = lgb.station_and_city_delta_log.to_numpy()
    baseline = reference.boardings.loc[scope].to_numpy()
    log = np.log1p(baseline)
    paired = np.expm1((log + .5 * (lgb_effect + nn_effect)).clip(min=0.))
    tree_only = np.expm1((log + .5 * lgb_effect).clip(min=0.))
    for result in (paired, tree_only):
        if not np.isfinite(result).all() or (result < 0.).any():
            raise ValueError("Cancellation forecasts must be finite and nonnegative.")
    report = target.assign(baseline=baseline, paired=paired, tree_only=tree_only,
                           lgb_log_delta=lgb_effect, nn_log_delta=nn_effect)
    day = report.groupby("date")[["baseline", "paired", "tree_only"]].sum().to_dict(orient="index")
    for date, data in day.items():
        if data["paired"] >= data["baseline"] or data["tree_only"] >= data["baseline"]:
            raise ValueError(f"Removing a cancelled event unexpectedly increases aggregate demand on {date}.")
    result = reference.boardings.to_numpy(copy=True)
    result[scope] = paired
    if np.any(result[~scope] != reference.boardings[~scope]):
        raise ValueError("The cancellation change crossed its four verified station-days.")
    if args.submission:
        save_submission(args.submission, reference, grid, result)
    return {"sources": SOURCES, "baseline": str(args.baseline.resolve()), "grid_rows": len(grid),
            "changed_rows": int(np.count_nonzero(result != reference.boardings)),
            "neural_archive_rmse": [float(np.sqrt(np.square(frame.original_nn - frame.archived_nn).mean())) for frame in neural],
            "daily_totals": day, "hourly": report.to_dict(orient="records"),
            "submission": str(args.submission.resolve()) if args.submission else None,
            "sha256": hashlib.sha256(args.submission.read_bytes()).hexdigest() if args.submission else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "baseline", "tree", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--neural", required=True, nargs="+", type=Path)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    result = run(args)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("changed_rows", "daily_totals", "neural_archive_rmse", "sha256")}), flush=True)


if __name__ == "__main__":
    main()
