"""Apply the validated Cup and walking-event forecasts to the frozen Night v4 release."""

import argparse
import io
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bottleneck", "graph_clean")]

from apply_firebaseball import CUTOFF, read_official, sha256
from apply_incheon_walk import apply_incheon_walk, schedule
from bucheon_cup import KEY, apply_cup, read_facts

REFERENCE_SHA = "4c85257294be267c47396587f3a71a2d1e238dc93f957a634cfe662f8e379965"
FIXED_INPUTS = {
    "calendar": "514fe1ba45a539ed06dc81747117288c99b6e81dbace4a7c68e671b4c3cd7fc0",
    "protected": "be94d181e8b423f37264515158270daf9c39c3623a13106d85ff8d509ad2fadd",
    "cup": "1e4a11e72bc71f5695dc73363b402c28f001a9b1daa2738953dc326eabfd6f9c",
    "walk": "f5519f1f3798982cd08a65d14aacee98b6b747ac27b01e3b97a98c9359f9d93b",
}
OFFICIAL_ZIP_SHA = "e86cdc0599e7a62a4d722b3053a3f9566c0bf93c2f49f8f910c3258b69adc18a"
OFFICIAL_CSV_SHA = {
    "train.csv": "1b3b21e8f1fe22a44c0ff4adba315784a5c21caea25098a15869c4ed370822bc",
    "test.csv": "d257fed268db7919fd8fbc45b6c78dfbdb09b6bb40d4ae1d3f2a666c6e69fefa",
}


def read_competition(path):
    if path.is_file():
        digest = sha256(path)
        if digest != OFFICIAL_ZIP_SHA:
            raise ValueError("Expected the original competition ZIP.")
        return *read_official(path), {"official": digest}
    hashes = {name: sha256(path / name) for name in OFFICIAL_CSV_SHA}
    if hashes != OFFICIAL_CSV_SHA:
        raise ValueError("Expected the exact official train.csv and test.csv.")
    train = pd.read_csv(path / "train.csv", usecols=[*KEY, "boardings"], parse_dates=["date"])
    test = pd.read_csv(path / "test.csv", parse_dates=["date"])
    return train, test, hashes


def read_forecast(contents, test):
    forecast = pd.read_csv(io.BytesIO(contents), float_precision="round_trip")
    if list(forecast) != ["example_id", "boardings"] or not forecast.example_id.equals(test.example_id):
        raise ValueError("Expected the exact official test IDs and order.")
    values = forecast.boardings.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("All forecasts must be finite and nonnegative.")
    return values


def run(args):
    names = ("base", "calendar", "protected", "cup", "walk")
    paths = {name: getattr(args, name) for name in names}
    if args.reference:
        paths["reference"] = args.reference
    official_paths = [args.official]
    if args.official.is_dir():
        official_paths.extend(args.official / name for name in OFFICIAL_CSV_SHA)
    if args.output.suffix != ".csv" or args.output.resolve() in {
        path.resolve() for path in (*paths.values(), *official_paths)
    }:
        raise ValueError("Use a separate .csv output path.")
    hashes = {name: sha256(path) for name, path in paths.items()}
    for name, expected in FIXED_INPUTS.items():
        if hashes[name] != expected:
            raise ValueError(f"Unexpected {name} SHA-256.")
    if hashes.get("reference", hashes["base"]) != REFERENCE_SHA:
        raise ValueError("Expected the frozen Night v4 reference.")
    train, test, official_hashes = read_competition(args.official)
    hashes.update(official_hashes)
    original = args.base.read_bytes()
    base = read_forecast(original, test)
    difference = 0.
    if args.reference:
        reference = read_forecast(args.reference.read_bytes(), test)
        difference = float(np.abs(np.log1p(base) - np.log1p(reference)).max())
        if difference > 1e-10:
            raise ValueError("Regenerated base differs from the verified Night v4 release.")

    cal = pd.read_csv(args.calendar, usecols=["date", "is_holiday"], parse_dates=["date"]).set_index("date")
    protected = pd.read_csv(args.protected, usecols=KEY[:2], parse_dates=["date"])
    cup_counts, cup, evidence = apply_cup(train, test[KEY], base, CUTOFF, read_facts(args.cup))
    walk_counts, walk = apply_incheon_walk(
        train, test[KEY], base, CUTOFF, cal, pd.MultiIndex.from_frame(protected), schedule(args.walk)
    )
    expected_cup = test.index[test.date.eq("2026-07-29") & test.station_id.eq(3754) & test.hour.ge(20)]
    expected_walk = test.index[test.date.eq("2026-10-17") & test.station_id.eq(3226) & test.hour.between(12, 18)]
    if (len(expected_cup) != 4 or len(expected_walk) != 7 or not cup.index.equals(expected_cup)
            or not walk.index.equals(expected_walk) or not cup.index.intersection(walk.index).empty):
        raise ValueError("Expected only the four Cup rows and the seven walking-event rows.")

    values = base.copy()
    values[cup.index] = cup_counts[cup.index]
    values[walk.index] = walk_counts[walk.index]
    changed = cup.index.union(walk.index)
    lines = original.splitlines(keepends=True)
    if len(lines) != len(test) + 1:
        raise ValueError("Expected one header and one line per test ID.")
    for index in changed:
        lines[index + 1] = f"{test.example_id.iat[index]},{float(values[index])!r}\n".encode("ascii")
    payload = b"".join(lines)
    if not np.array_equal(read_forecast(payload, test), values):
        raise ValueError("Serialized output differs from the validated forecasts.")
    report = {
        "cutoff": str(CUTOFF.date()),
        "rows": len(test),
        "changed_rows": len(changed),
        "unchanged_rows_byte_identical": len(test) - len(changed),
        "base_max_log1p_difference_from_reference": difference,
        "inputs_sha256": hashes,
        "cup_donors": evidence,
        "changes": [
            {"example_id": test.example_id.iat[index], "date": str(test.date.iat[index].date()),
             "station_id": int(test.station_id.iat[index]), "hour": int(test.hour.iat[index]),
             "previous": float(base[index]), "updated": float(values[index])}
            for index in changed
        ],
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    args.output.write_bytes(payload)
    report["output_sha256"] = sha256(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("official", "base", "calendar", "protected", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Frozen CSV for checking a numerically equivalent rebuilt base.")
    parser.add_argument("--cup", type=Path, default=ROOT / "graph_clean" / "bucheon_cup_facts.csv")
    parser.add_argument("--walk", type=Path, default=ROOT / "bottleneck" / "incheon_walk.csv")
    result = run(parser.parse_args())
    print(f"{result['rows']} rows, {result['changed_rows']} changes, SHA-256 {result['output_sha256']}")
