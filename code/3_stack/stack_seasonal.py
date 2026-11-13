"""Pure functions extracted unchanged from goal_014/model/v4_seasonal_residual.py."""
import numpy as np
import pandas as pd
KEY = ['date', 'station_id', 'hour']
KEYS = ['station_id', 'weekend', 'hour']
CUTOFF = pd.Timestamp('2026-06-30')
EVENT_STATIONS = [3127, 3139, 3226, 3754]
EXCLUDED_STATIONS = [3107, 3108, 3109, *EVENT_STATIONS]
HOLIDAY_FLAGS = ['is_holiday', 'is_bridge_day', 'is_quasi_holiday', 'is_pre_long_offday', 'is_post_long_offday']

def ordinary_rows(frame, holidays, events):
    mask = (~frame.date.isin(holidays) & ~frame.station_id.isin(EXCLUDED_STATIONS)
            & ~pd.MultiIndex.from_frame(frame[KEY[:2]]).isin(events))
    return frame.loc[mask].assign(weekend=lambda x: x.date.dt.day_of_week.ge(5))


def fit_correction(archive, future, origin, holidays, events):
    if future.date.le(origin).any():
        raise ValueError("Forecast dates must follow the fitting cutoff.")
    past = archive.loc[archive.origin.lt(origin) & archive.date.le(origin)].sort_values("origin")
    past = ordinary_rows(past.drop_duplicates(KEY, keep="last"), holidays, events)
    change = np.zeros(len(future))
    if past.date.nunique() < 21:
        return change, int(past.date.nunique())

    raw = (past.y_log - past.forecast).clip(-.45, .45)
    past["hourly_error"] = raw - raw.groupby([past.date, past.station_id]).transform("mean")
    doy = pd.to_datetime({"year": 2000, "month": past.date.dt.month, "day": past.date.dt.day}).dt.day_of_year
    target = ordinary_rows(future, holidays, events)
    for month, rows in target.groupby(target.date.dt.month):
        midpoint = pd.Timestamp(year=2000, month=int(month), day=15).day_of_year
        distance = np.minimum((doy - midpoint).abs(), 366 - (doy - midpoint).abs())
        near = past.assign(weight=(1. - distance / 92.).clip(lower=0.))
        near["weighted"] = near.hourly_error * near.weight
        group = near.groupby(KEYS)[["weight", "weighted"]].sum()
        profile = group.weighted / (group.weight + 28.)
        value = profile.reindex(pd.MultiIndex.from_frame(rows[KEYS])).fillna(0.)
        delta = pd.Series(.35 * value.clip(-.12, .12).to_numpy(), index=rows.index)
        delta -= delta.groupby([rows.date, rows.station_id]).transform("mean")
        change[rows.index] = delta
    return change, int(past.date.nunique())
