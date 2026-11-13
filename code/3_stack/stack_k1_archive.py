"""Pure functions extracted unchanged from goal_014/graph_clean/k1_residual_replay.py."""
import numpy as np
import pandas as pd
KEY = ['date', 'station_id', 'hour']
FIT_FIELDS = ['station_id', 'dow', 'hour']
EXCLUDED_STATIONS = {3107, 3108, 3109, 3127, 3139, 3209, 3754}

def k1_stats(archive, origin):
    mask = (archive.date.gt(origin - pd.Timedelta(days=84)) & archive.date.le(origin)
            & archive.origin.lt(origin) & archive.quiet)
    history = archive.loc[mask].sort_values("origin").drop_duplicates(KEY, keep="last").copy()
    history["error"] = history.y_log - history.forecast
    group = history.groupby(KEY[:2]).error
    history["error"] = (history.error - group.transform("mean")).where(group.transform("size").eq(18))
    table = history.groupby(FIT_FIELDS).error.agg(["mean", "count"])
    return (.25 * table["mean"] * table["count"] / (table["count"] + 12)).where(table["count"].ge(6))


def k1_apply(target, stats):
    values = stats.reindex(pd.MultiIndex.from_frame(target[FIT_FIELDS]))
    joined = target.assign(correction=values.to_numpy())
    groups = joined.groupby(KEY[:2]).correction
    valid = joined.quiet & groups.transform("count").eq(18)
    return (joined.correction - groups.transform("mean")).where(valid, 0.).to_numpy()


def k1_quiet(rows, days, events):
    return (~rows.date.isin(days) & ~rows.station_id.isin(EXCLUDED_STATIONS)
            & ~pd.MultiIndex.from_frame(rows[KEY[:2]]).isin(events))


def k1_archive(raw, holidays, events):
    raw = raw.assign(dow=raw.date.dt.day_of_week, quiet=k1_quiet(raw, holidays, events))
    forecasts = []
    for cutoff, fold in raw.groupby("origin"):
        fold = fold.reset_index(drop=True)
        adjustment = k1_apply(fold, k1_stats(raw, cutoff))
        forecasts.append(fold.assign(forecast=fold.forecast + adjustment))
    return pd.concat(forecasts, ignore_index=True)
