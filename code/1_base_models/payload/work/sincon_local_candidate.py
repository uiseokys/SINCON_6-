"""Reproduce the validated festival and apartment additions to the SINCON forecast."""
import argparse
from pathlib import Path

import pandas as pd
import torch

from sincon_conditional_candidate import Candidate, validate, write_submission
from sincon_festival_experiment import local_days, transferred
from sincon_housing_growth import forecast_changes, read_exposure
from sincon_station_state import KEY


HOUSING_WEIGHT = 0.5


class LocalCandidate(Candidate):
    def __init__(self, args):
        super().__init__(args)
        self.festivals = local_days(args.events)
        self.meta = pd.read_csv(args.team_dir / "data/metaData.csv", parse_dates=["first_date"]).set_index("station_id")
        centers = pd.read_csv(args.station_centers).set_index("station_id").reindex(self.meta.index)
        self.housing_level = read_exposure(args.housing, centers)

    def predict(self, history, future, neural, reference, cutoff, with_missing_test=False):
        predictions = super().predict(history, future, neural, reference, cutoff, with_missing_test)
        keys = future[KEY]
        housing, _, _ = forecast_changes(history, keys, pd.Timestamp(cutoff), self.housing_level, self.calendar, self.meta)
        base = predictions["conditional_stack"]
        with_festival, _ = transferred(history, keys, self.festivals, base)
        local = with_festival + HOUSING_WEIGHT * housing
        predictions["local_stack"] = local
        if with_missing_test:
            withheld, _ = transferred(history, keys, self.festivals, predictions["withheld_bus_month"])
            predictions["local_without_last_bus_month"] = withheld + HOUSING_WEIGHT * housing
        return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team_dir", "external_zip", "scoreboard", "mobility_directory", "station_addresses", "bus", "football",
                 "events", "housing", "station_centers"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model = LocalCandidate(args)
    results = validate(model, args.results, args.predictions, candidate_key="local_stack", reference_key="conditional_stack")
    if args.submission:
        write_submission(args, model, results, candidate_key="local_stack")
