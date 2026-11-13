"""Apply locally validated event corrections to the team's 0.14774 submission."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_songdo_schedule import save_submission
from sincon_station_state import KEY


BASE_NAME = "submission_0p15126_event_cancel_fix.csv"
BASE_SHA256 = "cc49c30814dec15c56c3557b4ff4827b75b65e47f79842b8c73c0fc914c9ea93"
PREVIOUS_NAME = "submission_local_stack_level_station_events_xgb_weather_geomdan_late.csv"
EPSILON = 1e-9
LAYER_FILES = {
    "spectator_base": "submission_station_profile_munhak_geomdan.csv",
    "spectator": "submission_station_profiles_spectator.csv",
    "program_base": "submission_station_profiles_event_audit.csv",
    "program": "submission_station_profiles_event_program.csv",
}
PROGRAM_WINDOWS = {
    "psy_egress": [("2026-07-11", 3210, range(21, 23))],
    "rain_cancellation_timing": [("2026-08-22", 3127, range(15, 18)), ("2026-09-25", 3127, range(14, 17))],
    "science_festival": [(date, station, range(15, 18)) for date in ("2026-10-23", "2026-10-24") for station in (3209, 3210)],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submission(path, grid):
    frame = pd.read_csv(path, float_precision="round_trip")
    if (list(frame.columns) != ["example_id", "boardings"] or not frame.example_id.equals(grid.example_id)
            or not np.isfinite(frame.boardings).all() or frame.boardings.lt(0.).any()):
        raise ValueError(f"Expected a finite nonnegative forecast in official test order: {path}")
    return frame


def window(grid, windows):
    keys = {(date, station, hour) for date, station, hours in windows for hour in hours}
    mask = pd.MultiIndex.from_frame(grid[KEY]).isin(keys)
    if mask.sum() != len(keys):
        raise ValueError("Every documented event hour must occur exactly once in test.csv.")
    return mask


def masked_deltas(grid, before, after, masks):
    delta = np.log1p(after.boardings.to_numpy()) - np.log1p(before.boardings.to_numpy())
    changed = np.abs(delta) > EPSILON
    expected = np.column_stack(list(masks.values())).any(axis=1)
    if not np.array_equal(changed, expected):
        raise ValueError("The archived stage changed rows outside its documented event windows.")
    return {name: np.where(mask, delta, 0.) for name, mask in masks.items()}


def run(args):
    work = args.work_dir
    grid = pd.read_csv(args.team / "data/test.csv", usecols=["example_id", *KEY])
    if len(grid) != 157_194 or not grid.example_id.is_unique or grid.duplicated(KEY).any():
        raise ValueError("The complete official test grid is required.")
    base_path = args.team / BASE_NAME
    if digest(base_path) != BASE_SHA256:
        raise ValueError("The submitted 0.14774 file does not match its frozen checksum.")
    base = submission(base_path, grid)
    previous = submission(work / PREVIOUS_NAME, grid)
    claude_windows = [(f"2026-08-{day:02d}", 3139, range(21, 24)) for day in range(8, 16)]
    claude_windows += [(date, 3127, range(6, 24)) for date in ("2026-09-26", "2026-09-27", "2026-10-24", "2026-10-25")]
    protected = window(grid, claude_windows)
    masked_deltas(grid, previous, base, {"confirmed_previous": protected})

    stages = {name: submission(work / file, grid) for name, file in LAYER_FILES.items()}
    spectator = {
        "ssg_landers_special": window(grid, [("2026-10-17", 3127, range(16, 23))]),
        "burn_to_win": window(grid, [("2026-10-31", 3127, range(12, 20))]),
    }
    layers = masked_deltas(grid, stages["spectator_base"], stages["spectator"], spectator)
    program = {name: window(grid, scope) for name, scope in PROGRAM_WINDOWS.items()}
    layers.update(masked_deltas(grid, stages["program_base"], stages["program"], program))

    wonmi_path = work / "sincon_wonmi_sports_day_final_delta.csv"
    wonmi = pd.read_csv(wonmi_path).set_index("example_id")
    target = grid.loc[window(grid, [("2026-10-17", 3754, range(13, 17))])].set_index("example_id")
    if not wonmi[KEY].equals(target[KEY]) or not wonmi.log_delta.between(0., .10, inclusive="right").all():
        raise ValueError("Wonmi correction must match exactly the four documented positive adjustments.")
    layers["wonmi_sports_day"] = wonmi.log_delta.reindex(grid.example_id, fill_value=0.).to_numpy()

    songdo_path = work / "sincon_songdo_beach_evening_audit_results.json"
    songdo_report = json.loads(songdo_path.read_text())
    songdo = pd.DataFrame(songdo_report["keyed_increments"]).set_index("example_id")
    scope = window(grid, [(f"2026-08-{day:02d}", 3139, (19, 20)) for day in range(8, 16)])
    if (not songdo[KEY].equals(grid.loc[scope].set_index("example_id")[KEY])
            or not np.allclose(songdo.boardings, base.boardings[scope], atol=1e-10, rtol=0.)
            or songdo_report["transfer"] != .25 or not songdo.log_delta.between(-.5, 0., inclusive="left").all()):
        raise ValueError("Songdo may reduce only the sixteen audited 19:00–20:00 forecasts.")
    layers["songdo_beach_evening"] = songdo.log_delta.reindex(grid.example_id, fill_value=0.).to_numpy()

    nonzero = np.column_stack([offset != 0. for offset in layers.values()])
    if (nonzero.sum(axis=1) > 1).any() or nonzero[protected].any():
        raise ValueError("Event layers must be disjoint and retain the 96 previously corrected rows.")
    active = nonzero.any(axis=1)
    forecast = base.boardings.to_numpy(copy=True)
    delta = np.sum(list(layers.values()), axis=0)
    forecast[active] = np.expm1(np.log1p(forecast[active]) + delta[active])
    if not np.array_equal(forecast != base.boardings.to_numpy(), active):
        raise ValueError("The exported changes must be exactly the documented event rows.")

    save_submission(args.output, base, grid, forecast)
    rows = []
    for layer, offset in layers.items():
        affected = offset != 0.
        group = grid.loc[affected].assign(layer=layer, before=base.boardings, after=forecast[affected], log_delta=offset[affected])
        rows.append(group)
    changes = pd.concat(rows).sort_values(KEY)
    report = {
        "base": str(base_path), "base_sha256": BASE_SHA256,
        "reported_base_public_rmsle": .14774,
        "output": str(args.output.resolve()), "sha256": digest(args.output),
        "rows": len(grid), "previous_event_rows_preserved": int(protected.sum()),
        "changed_rows": int(active.sum()), "source_sha256": {name: digest(work / f) for name, f in LAYER_FILES.items()},
        "wonmi_sha256": digest(wonmi_path), "songdo_audit_sha256": digest(songdo_path),
        "changes": json.loads(changes.to_json(orient="records", double_precision=15)),
    }
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['changed_rows']} event rows, {len(grid)} predictions, SHA-256 {report['sha256']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
