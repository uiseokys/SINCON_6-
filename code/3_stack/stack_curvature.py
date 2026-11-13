"""Pure functions extracted unchanged from goal_014/model/k3_network_variation.py."""
import numpy as np
import pandas as pd
from stack_seasonal import EXCLUDED_STATIONS, KEY
DAY = KEY[:2]
EVENT_STATIONS = [*EXCLUDED_STATIONS, 3209, 3210]
GROUP = ['station_id', 'weekday']
HOURS = np.arange(6, 24)
N_HOURS = len(HOURS)
FREQUENCIES = np.arange(6, N_HOURS)
HIGH_FREQUENCY = np.sqrt(2 / N_HOURS) * np.cos(np.pi * (np.arange(N_HOURS) + .5)[:, None] * FREQUENCIES / N_HOURS)
PROJECTION = HIGH_FREQUENCY @ HIGH_FREQUENCY.T

def ordinary(frame, days, protected):
    return (~frame.date.isin(days) & ~frame.station_id.isin(EVENT_STATIONS)
            & ~pd.MultiIndex.from_frame(frame[DAY]).isin(protected))


def complete_profiles(rows, value, holidays, events):
    quiet = rows.loc[ordinary(rows, holidays, events)]
    return quiet.pivot(index=DAY, columns='hour', values=value).reindex(columns=HOURS).dropna()


def anchor(observed, future, cutoff, holidays, events):
    if list(future) != [*KEY, 'forecast'] or not future.date.gt(cutoff).all():
        raise ValueError('Only post-cutoff keys and baseline forecasts are valid inference inputs.')
    dates = observed.index.get_level_values('date')
    last = observed.loc[(dates >= cutoff - pd.Timedelta(days=69)) & (dates <= cutoff)].reset_index()
    last['weekday'] = last.date.dt.day_of_week
    weight = np.exp2(-(cutoff - last.date).dt.days / 28.)
    weighted = last[GROUP].join(last[list(HOURS)].mul(weight, axis=0))
    summary = weighted.groupby(GROUP).sum()
    denominator = last.assign(weight=weight).groupby(GROUP).weight.agg(['sum', 'size'])
    reference = summary.div(denominator['sum'].where(denominator['size'].ge(6)), axis=0)

    profile = complete_profiles(future, 'forecast', holidays, events)
    idx = pd.MultiIndex.from_arrays([profile.index.get_level_values('station_id'),
                                     profile.index.get_level_values('date').day_of_week], names=GROUP)
    prior = reference.reindex(idx).to_numpy()
    target = profile.to_numpy()
    shift = .15 * (np.where(np.isfinite(prior), prior - target, 0.) @ PROJECTION)
    amplitude = np.maximum(np.max(abs(shift), axis=1), .065)
    shift *= (.065 / amplitude)[:, None]
    # A common scale preserves the centered log-profile even at zero forecasts.
    limits = np.divide(target, -shift, out=np.full_like(shift, np.inf), where=shift < 0)
    shift *= np.minimum(1., limits.min(axis=1))[:, None]
    adjustment = pd.DataFrame(shift, index=profile.index, columns=profile.columns).stack()
    delta = adjustment.reindex(pd.MultiIndex.from_frame(future[KEY])).fillna(0.).to_numpy()
    if ((future.forecast.to_numpy() + delta < -1e-14).any()
            or np.max(abs(shift.mean(axis=1)), initial=0.) > 1e-14
            or np.any(delta[~ordinary(future, holidays, events)] != 0.)):
        raise ValueError('Expected a nonnegative, zero-mean ordinary-day profile correction.')
    return delta
