"""Reproduce the clean refined forecast with official CSVs and frozen model forecasts."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pandas as pd


WORK = Path(__file__).resolve().parent
CACHE_FILES = (
    "sincon_clean_event_oof.csv.gz",
    "sincon_clean_tabicl_final.csv",
    "sincon_clean_tabicl_oof.csv",
    "sincon_clean_vector_oof_warm.csv.gz",
)


def extract_cache(bundle, directory):
    with ZipFile(bundle) as source:
        manifest = json.loads(source.read("manifest.json"))
        for name in CACHE_FILES:
            data = source.read(name)
            if hashlib.sha256(data).hexdigest() != manifest[name]:
                raise ValueError(f"Unexpected checksum for {name}.")
            (directory / name).write_bytes(data)


def run(team, bundle, output, upstream):
    output.mkdir(parents=True, exist_ok=True)

    def execute(script, *args):
        subprocess.run([sys.executable, str(WORK / script), *map(str, args)], cwd=WORK, check=True)

    with TemporaryDirectory(prefix="sincon_clean_") as path:
        cache = Path(path)
        extract_cache(bundle, cache)
        xgb = cache / "origin_2026-06-30.csv.gz"
        with ZipFile(WORK / "sincon_xgb_forecasts.zip") as source:
            xgb.write_bytes(source.read(xgb.name))
        early, events, model = (cache / f"{name}.csv" for name in ("early", "events", "model"))
        protected = cache / "protected.csv"
        oof, daily, tabicl, vector_warm = (cache / name for name in CACHE_FILES)
        earlier = upstream.earlier or WORK / "sincon_source_reweight_portable.zip"
        replacements = []
        for name in ("local_stack", "conditional_stack", "local_historical_abc"):
            path = getattr(upstream, name)
            if path:
                replacements.extend((f"--{name.replace('_', '-')}", path.resolve()))
        execute("sincon_clean_15126_replay.py", team, earlier.resolve(), upstream.xgb or xgb,
                early, output / "early.json", *replacements)
        execute("sincon_clean_event_replay.py", team, early, oof, events, output / "events_before_model.json",
                "--stage", "before_model", "--protected", protected)
        validation = ("--validation-archive", upstream.validation_archive) if upstream.validation_archive else ()
        execute("sincon_clean_model_layers.py", team, events, protected, model, output / "model.json",
                "--clean-oof", oof, "--oof-forecast-column", "baseline", "--tabicl-daily", daily,
                "--tabicl-oof", tabicl, "--vector", WORK / "sincon_vector_shape_final_raw.csv.gz",
                "--vector-oof-warm", vector_warm, *validation)
        final = output / "submission_clean_refined.csv"
        execute("sincon_clean_event_replay.py", team, model, oof, final, output / "events_after_model.json",
                "--stage", "after_model", "--early-protected", protected, "--protected", output / "event_keys.csv")
        base = pd.read_csv(events, float_precision="round_trip").set_index("example_id").boardings
        actual = pd.read_csv(final, float_precision="round_trip").set_index("example_id").boardings
        ids = pd.read_csv(protected, usecols=["example_id"]).example_id
        later = json.loads((output / "events_after_model.json").read_text())["changes"]
        unchanged = ids.loc[~ids.isin({row["example_id"] for row in later})]
        if not base.reindex(unchanged).equals(actual.reindex(unchanged)):
            raise ValueError("Final forecasts must preserve earlier event keys outside the documented refinements.")
        print(f"{final} {hashlib.sha256(final.read_bytes()).hexdigest()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "bundle", "output"):
        parser.add_argument(name, type=Path)
    for name in ("earlier", "xgb", "local-stack", "conditional-stack", "local-historical-abc"):
        parser.add_argument(f"--{name}", type=lambda value: Path(value).resolve())
    parser.add_argument("--validation-archive", type=lambda value: Path(value).resolve())
    args = parser.parse_args()
    run(args.team.resolve(), args.bundle.resolve(), args.output.resolve(), args)
