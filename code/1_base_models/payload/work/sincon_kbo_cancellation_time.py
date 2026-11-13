"""Backtest stadium demand around the public announcement of a cancelled game."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sincon_munhak_ordinary import STADIUM, category, home_dates, robust_profile
from sincon_overlay_candidate import replay, trained_inputs
from sincon_station_state import KEY
from sincon_xgb_conditioned import load_episodes


@dataclass(frozen=True)
class Cancellation:
    day: str
    game: float
    announced: float
    source: str

    @property
    def date(self):
        return pd.Timestamp(self.day)

    @property
    def lead(self):
        return self.game - self.announced


HISTORY = (
    Cancellation("2025-04-05", 17, 16 + 58/60, "https://v.daum.net/v/zLaeo5MU5x"),
    Cancellation("2025-05-09", 18.5, 16 + 16/60, "https://v.daum.net/v/20250509161604328"),
    Cancellation("2025-06-13", 18.5, 18 + 8/60, "https://v.daum.net/v/20250613181630000"),
    Cancellation("2025-06-20", 18.5, 15 + 33/60, "https://www.osen.co.kr/article/G1112596586"),
    Cancellation("2025-07-17", 18.5, 17 + 27/60, "https://sports.donga.com/sports/article/all/20250717/132022652/1"),
    Cancellation("2025-07-18", 18.5, 15 + 54/60, "https://news.mt.co.kr/mtview.php?VN=&no=2025071814274759402"),
    Cancellation("2025-08-13", 18.5, 15 + 40/60, "https://www.chosun.com/sports/baseball/2025/08/13/CBNMEV54KP6ESUVATIUQ3PZSCQ/"),
    Cancellation("2025-09-06", 17, 16 + 40/60, "https://www.sportschosun.com/baseball/2025-09-06/202509060100043850007081"),
    Cancellation("2025-09-19", 18.5, 16 + 13/60, "https://v.daum.net/v/20250919171446512"),
    Cancellation("2025-09-24", 18.5, 16 + 23/60, "https://v.daum.net/v/WOSegHGkwx"),
    Cancellation("2025-10-10", 18.5, 16 + 37/60, "https://www.sportsseoul.com/news/read/1552567"),
    Cancellation("2026-04-09", 18.5, 15 + 39/60, "https://www.newspim.com/news/view/20260409001150"),
    Cancellation("2026-05-26", 18.5, 18 + 1/60, "https://v.daum.net/v/79wS4xb0CH"),
)
FINAL = (
    Cancellation("2026-08-22", 19, 16, "https://www.mk.co.kr/news/sports/12133816"),
    Cancellation("2026-09-25", 17, 15 + 26/60, "https://m.mk.co.kr/amp/12161370"),
)
HOURS = np.arange(6, 24)
ORIGIN = pd.Timestamp("2026-06-30")


def typical(ordinary, rows, origin):
    return robust_profile(ordinary.loc[ordinary.date.le(origin)], rows[KEY], origin)


def ordinary_level(profile, values, cancellation):
    early = HOURS + .5 <= cancellation.announced - 3.
    return profile + np.clip(np.median(values[early] - profile[early]), -.65, .40)


def build_history(train, ordinary):
    profiles = {}
    for event in HISTORY:
        rows = train.loc[train.date.eq(event.date)].sort_values("hour")
        if not np.array_equal(rows.hour, HOURS):
            raise ValueError(f"Missing official observations on {event.day}.")
        value = rows.observed.to_numpy()
        normal = typical(ordinary, rows, event.date - pd.Timedelta(days=1))
        baseline = ordinary_level(normal, value, event)
        active = np.abs(HOURS + .5 - event.announced - .5) <= 1.5
        profiles[event] = np.where(active, np.maximum(np.expm1(value) - np.expm1(baseline), 0.), 0.)
    return profiles


def proposal(event, rows, origin, ordinary, profiles, timing_shift=0.):
    if event.date <= origin or rows.date.nunique() != 1 or not np.array_equal(rows.hour, HOURS):
        raise ValueError("The station-day must follow its origin and include all eighteen hours.")
    base = rows.forecast.to_numpy().clip(min=0.)
    # Approximate admission exposure with a pre-specified boundary. No
    # observed ridership is used to select historical precedents.
    prior = [donor for donor in profiles if donor.date <= origin and (donor.lead <= 1.25) == (event.lead <= 1.25)
             and abs(donor.lead - event.lead) <= 1.25]
    if len(prior) < 2:
        return base.copy(), []
    level = ordinary_level(typical(ordinary, rows, origin), base, event)
    moved = [np.interp(HOURS - (event.announced - donor.announced) - timing_shift, HOURS,
                       profiles[donor], left=0., right=0.) for donor in prior]
    transferred = np.median(moved, axis=0)
    active = np.abs(HOURS + .5 - event.announced - .5) <= 1.5
    estimate = np.log1p(np.expm1(level).clip(min=0.) + transferred)
    return base + .5 * np.where(active, estimate - base, 0.), [donor.day for donor in prior]


def run(args):
    episodes, raw, inputs, protected, calibrated, fallback = load_episodes(args)
    games, _ = home_dates(args.work_dir)
    station = raw.loc[raw.station_id.eq(STADIUM)]
    ordinary = station.loc[category(station, *protected, games) == "ordinary"]
    history = build_history(station, ordinary)
    fitted = trained_inputs(raw, args.work_dir, args.earlier, *protected)
    exact, audits = {}, {}
    truth = station.set_index(KEY).observed
    for event in HISTORY:
        available = [(name, origin) for name, (rows, origin, _) in episodes.items()
                     if origin < event.date <= rows.date.max()]
        if not available:
            continue
        label, origin = max(available, key=lambda item: item[1])
        if label not in exact:
            full, _ = replay(episodes[label][0].reset_index(drop=True), origin, raw, inputs, protected, calibrated, fallback, fitted)
            exact[label] = full.loc[full.station_id.eq(STADIUM)]
        rows = exact[label].loc[exact[label].date.eq(event.date)].sort_values("hour").rename(columns={"combined": "forecast"})
        y = truth.reindex(pd.MultiIndex.from_frame(rows[KEY])).to_numpy()
        if not np.allclose(y, rows.y_log, atol=1e-8, rtol=0.):
            raise ValueError(f"Archived labels do not match train.csv on {event.day}.")
        pred, donors = proposal(event, rows[[*KEY, "forecast"]], origin, ordinary, history)
        early, _ = proposal(event, rows[[*KEY, "forecast"]], origin, ordinary, history, -.5)
        late, _ = proposal(event, rows[[*KEY, "forecast"]], origin, ordinary, history, .5)
        methods = (rows.forecast.to_numpy(), pred, early, late)
        audits[event.day] = {"origin": str(origin.date()), "precedents": donors, "lead_hours": event.lead,
                             "sse": {name: float(np.square(y - values.clip(min=0.)).sum()) for name, values in
                                     zip(("reference", "fixed", "earlier", "later"), methods)}}

    grid = pd.read_csv(args.team / "data/test.csv", parse_dates=["date"], usecols=[*KEY, "example_id"])
    baseline = pd.read_csv(args.baseline, float_precision="round_trip")
    if len(grid) != 157_194 or not grid.example_id.equals(baseline.example_id) or not grid.example_id.is_unique:
        raise ValueError("The current candidate must match official test IDs.")
    proposed = {}
    for event in FINAL:
        scope = grid.date.eq(event.date) & grid.station_id.eq(STADIUM)
        rows = grid.loc[scope].assign(forecast=np.log1p(baseline.boardings[scope])).sort_values("hour")
        pred, donors = proposal(event, rows[[*KEY, "forecast"]], ORIGIN, ordinary, history)
        detail = rows[["station_id", "hour", "example_id"]].assign(previous=np.expm1(rows.forecast), candidate=np.expm1(pred))
        proposed[event.day] = {"precedents": donors, "lead_hours": event.lead,
                               "hourly": detail.to_dict(orient="records")}

    support = [item for item in audits.values() if item["precedents"] and item["lead_hours"] > 1.25]
    scores = {name: float(np.sqrt(sum(item["sse"][name] for item in support) / (18 * len(support))))
              for name in ("reference", "fixed", "earlier", "later")} if support else {}
    return {"sources": {event.day: event.source for event in (*HISTORY, *FINAL)}, "historical": audits,
            "earlier_announcement_case_count": len(support), "earlier_announcement_rmsle": scores,
            "test_sensitivity": proposed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("team", "earlier", "recent", "extra_xgb", "june_xgb", "april_xgb", "baseline"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = run(args)
    (args.work_dir / "sincon_kbo_cancellation_time_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("earlier_announcement_case_count", "earlier_announcement_rmsle", "historical")}), flush=True)


if __name__ == "__main__":
    main()
