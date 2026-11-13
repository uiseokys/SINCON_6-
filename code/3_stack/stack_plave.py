"""Pure functions extracted unchanged from goal_014/bottleneck/build_v4_plave_in_show.py."""
import numpy as np
import pandas as pd
KEY = ['date', 'station_id', 'hour']
CUTOFF = pd.Timestamp('2026-06-30')
STATION = 3127
FIT_CLIP = (-.30, 0.)
SHRINK = .35
DONOR_ORIGINS = {'ye': '2025-06-30', 'muse': '2025-08-31', 'jin': '2025-09-30', 'park_hyo_shin': '2026-02-28'}

def read_history(train, concerts, archive_path):
    performances = pd.read_csv(concerts, parse_dates=['date'], usecols=['event', 'date', 'scheduled_start'])
    events = performances.loc[performances.event.isin(DONOR_ORIGINS)].copy()
    events['origin'] = pd.to_datetime(events.event.map(DONOR_ORIGINS))
    events['start_hour'] = pd.to_datetime(events.scheduled_start, format='%H:%M').dt.hour
    archive = pd.read_csv(archive_path, parse_dates=['date', 'origin'],
                          usecols=['origin', *KEY, 'y_log', 'lunar3122'])
    archive = archive.loc[archive.station_id.eq(STATION)]
    rows = archive.merge(events, on=['date', 'origin'], validate='many_to_one')
    rows['relative_hour'] = rows.hour - rows.start_hour
    rows = rows.loc[rows.relative_hour.isin([0, 1])].merge(train, on=KEY, validate='one_to_one')
    if (len(rows) != 14 or not rows.date.le(CUTOFF).all() or not rows.date.gt(rows.origin).all()
            or not np.allclose(np.log1p(rows.boardings), rows.y_log, atol=1e-10, rtol=0)):
        raise ValueError('Expected 14 verified historical donor rows and exact official training labels.')
    return rows.assign(error=np.log1p(rows.boardings) - rows.lunar3122)


def fit(rows, cutoff, exclude):
    completed = rows.groupby('event').date.max()
    prior = rows.loc[rows.event.isin(completed[completed.lt(cutoff)].index)
                     & rows.origin.lt(cutoff) & rows.event.ne(exclude)]
    profile = prior.groupby(['event', 'relative_hour']).error.median().groupby('relative_hour').median()
    return profile.clip(*FIT_CLIP) * SHRINK, sorted(prior.event.unique())


def backtest(history):
    folds = {}
    all_error, all_update = [], []
    for event, rows in history.groupby('event', sort=False):
        cutoff = rows.origin.iat[0]
        profile, donors = fit(history, cutoff, event)
        if not donors:
            continue
        before = rows.error.to_numpy()
        corrections = rows.relative_hour.map(profile).to_numpy()
        after = before - corrections
        if not np.isfinite(corrections).all():
            raise ValueError('Each validation hour needs prior event support.')
        all_error.extend(before)
        all_update.extend(after)
        folds[event] = {'cutoff': str(cutoff.date()), 'donors': donors, 'rows': len(rows),
                        'baseline': float(np.sqrt(np.mean(before ** 2))),
                        'candidate': float(np.sqrt(np.mean(after ** 2))),
                        'changed_rows': int(np.count_nonzero(corrections))}
    if set(folds) != {'muse', 'jin', 'park_hyo_shin'} or not all(
            value['candidate'] < value['baseline'] for value in folds.values()):
        raise ValueError('Each testable concert series must improve with the frozen rule.')
    return folds, {'rows': len(all_error), 'baseline': float(np.sqrt(np.mean(np.square(all_error)))),
                   'candidate': float(np.sqrt(np.mean(np.square(all_update)))),
                   'sse_improvement': float(np.sum(np.square(all_error)) - np.sum(np.square(all_update)))}
