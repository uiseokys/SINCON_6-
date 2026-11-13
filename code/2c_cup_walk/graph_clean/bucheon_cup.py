"""Apply a fixed, chronologically validated Korea Cup egress floor."""
import numpy as np
import pandas as pd

KEY = ["date", "station_id", "hour"]
STATION = 3754
HOURS = np.arange(6, 24)
EGRESS = HOURS >= 20
MATCH = pd.Timestamp("2026-07-29")
EXTRA_SHIFT_MINUTES = 35


def read_facts(path):
    facts = pd.read_csv(path, parse_dates=["date"])
    if (not facts.date.is_unique or not facts.station_id.eq(STATION).all()
            or not facts.license.eq("CC BY-NC-SA 2.0 KR").all()
            or not facts.match_type.isin(["regulation", "extra_time"]).all()):
        raise ValueError("Expected the attributed Bucheon home Cup match facts.")
    clock = pd.to_datetime(facts.kickoff_kst, format="%H:%M")
    return facts.assign(kickoff=clock.dt.hour * 60 + clock.dt.minute).sort_values("date")


def profile(counts, date, fixture_dates):
    dates = counts.index
    controls = dates[(dates < date) & (dates.day_of_week == date.day_of_week) & ~dates.isin(fixture_dates)][-8:]
    if len(controls) < 6:
        raise ValueError(f"Insufficient same-weekday support for {date.date()}.")
    return counts.loc[controls].median().to_numpy(), controls


def move(excess, minutes):
    edges = HOURS * 60
    starts = edges + minutes
    overlap = np.minimum(edges[:, None] + 60, starts[None, :] + 60)
    overlap -= np.maximum(edges[:, None], starts[None, :])
    return np.maximum(overlap, 0) @ excess / 60


def predictions(history, origin, day, facts):
    """Return both match scenarios using only previously completed regulation games."""
    origin, day = pd.Timestamp(origin), pd.Timestamp(day)
    if history.date.max() != origin or origin >= day:
        raise ValueError("Use only observations through the fixed cutoff before the match.")
    station_rows = history.loc[history.station_id.eq(STATION)]
    counts = station_rows.pivot(index="date", columns="hour", values="boardings").reindex(columns=HOURS)
    if not np.isfinite(counts).all().all():
        raise ValueError("Expected a complete hourly ridership panel.")
    kickoff = facts.set_index("date").kickoff
    normal, controls = profile(counts, day, kickoff.index)
    donors = facts.loc[facts.date.le(origin) & facts.match_type.eq("regulation")]
    if donors.empty:
        raise ValueError("Expected at least one earlier regulation match.")
    excess_by_donor = []
    evidence = []
    for donor in donors.itertuples(index=False):
        typical, donor_controls = profile(counts, donor.date, kickoff.index)
        excess = np.maximum(counts.loc[donor.date].to_numpy() - typical, 0.)
        first = donor.kickoff // 60 + 1
        excess[(HOURS < first) | (HOURS > first + 3)] = 0.
        excess_by_donor.append(excess)
        evidence.append({"date": str(donor.date.date()), "controls": donor_controls.strftime("%Y-%m-%d").tolist()})
    result = {"ordinary": normal}
    for label, delay in (("regulation", 0), ("extra_time", EXTRA_SHIFT_MINUTES)):
        shifts = kickoff.loc[day] - donors.kickoff + delay
        aligned = [move(excess, shift) for excess, shift in zip(excess_by_donor, shifts, strict=True)]
        result[label] = normal + np.where(EGRESS, np.median(aligned, axis=0), 0.)
    return result, {"origin": str(origin.date()), "controls": controls.strftime("%Y-%m-%d").tolist(), "donors": evidence}


def apply_cup(history, target, base_counts, origin, facts, day=MATCH):
    """Return patched counts, changed keyed rows, and chronological donor evidence."""
    day = pd.Timestamp(day)
    base = np.asarray(base_counts, dtype=float)
    if (len(base) != len(target) or not np.isfinite(base).all() or (base < 0).any()
            or target.date.le(pd.Timestamp(origin)).any() or "boardings" in target):
        raise ValueError("Expected earlier history and finite nonnegative unlabeled forecasts.")
    rows = target.date.eq(day) & target.station_id.eq(STATION) & target.hour.isin(HOURS[EGRESS])
    hours = target.loc[rows, "hour"].to_numpy()
    if not np.array_equal(np.sort(hours), HOURS[EGRESS]):
        raise ValueError("Expected each of the four egress forecast hours.")
    estimates, evidence = predictions(history, origin, day, facts)
    floor = estimates["regulation"][hours - HOURS[0]]
    revised = base.copy()
    revised[rows] = np.maximum(base[rows], floor)
    patch = target.loc[rows, KEY].assign(previous=base[rows], floor=floor, updated=revised[rows])
    return revised, patch.loc[patch.updated.ne(patch.previous)], evidence
