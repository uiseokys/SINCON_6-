"""Pure functions extracted unchanged from goal_014/housing/v4_geomdan_scale_probe.py."""
import numpy as np
import pandas as pd
KEY = ['date', 'station_id', 'hour']
STATIONS = (3107, 3108, 3109)
REGIME_WINDOW = 56
MINIMUM_MONTH_DAYS = 12
MINIMUM_MONTH_ERROR = .04
MAXIMUM_LOG_BIAS = .16
FRACTION = .5

def quiet_days(dates, calendar, protected):
    event_days = protected.loc[protected.station_id.isin(STATIONS), 'date']
    return ~dates.isin(calendar[calendar].index) & ~dates.isin(event_days)


def recent_regimes(archive, origin, allowed_days, minimum_origin, stations=None):
    eligible = (archive.date.gt(origin - pd.Timedelta(days=REGIME_WINDOW)) & archive.date.le(origin)
                & archive.origin.lt(origin) & archive.origin.ge(minimum_origin)
                & archive.date.isin(allowed_days))
    if stations is not None:
        eligible &= archive.station_id.isin(stations)
    previous = archive.loc[eligible].sort_values('origin').drop_duplicates(KEY, keep='last').copy()
    previous['residual'] = (previous.y_log - previous.forecast).clip(-.4, .4)
    daily = previous.groupby(['station_id', 'date'], as_index=False).residual.mean()
    signed = {}
    for sid, rows in daily.groupby('station_id'):
        monthly = rows.groupby(rows.date.dt.to_period('M')).residual.agg(['mean', 'size'])
        latest = monthly.tail(2)
        enough = len(latest) == 2 and latest['size'].ge(MINIMUM_MONTH_DAYS).all()
        if enough and (latest['mean'] > MINIMUM_MONTH_ERROR).all():
            sign = 1
        elif enough and (latest['mean'] < -MINIMUM_MONTH_ERROR).all():
            sign = -1
        else:
            sign = 0
        median = float(rows.residual.median())
        effect = sign * FRACTION * np.clip(sign * median, 0., MAXIMUM_LOG_BIAS)
        signed[int(sid)] = {'sign': sign, 'effect': float(effect), 'median_daily_residual': median,
                            'fit_start': str(rows.date.min().date()), 'fit_end': str(rows.date.max().date()),
                            'fit_dates': len(rows), 'by_month': {
                                str(month): {'days': int(row['size']), 'mean_daily_residual': float(row['mean'])}
                                for month, row in monthly.iterrows()}}
    return signed
