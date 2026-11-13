"""Apply the Incheon Grand Park walking-event profile to a verified base CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from apply_firebaseball import CUTOFF, read_official, sha256
from firebaseball import KEY

HERE = Path(__file__).resolve().parent
HOURS = np.arange(12, 19)
STATION = 3226
WEIGHT = .5


def schedule(path):
    walks = pd.read_csv(path, parse_dates=['date'])
    if (not walks.date.is_unique or not walks.event_id.is_unique or not walks.license.eq('KOGL Type 1').all()
            or not walks.station_id.eq(STATION).all() or len(walks) != 3):
        raise ValueError('Expected three separately licensed walking-event records')
    return walks.set_index('event_id')


def hour(clock):
    h, m = map(int, clock.split(':'))
    return h + m / 60


def forecast_increments(history, donor, target, exclusions):
    donor_date, target_date = donor.date, target.date
    if donor_date >= target_date or not history.date.lt(target_date).all():
        raise ValueError('All labels must precede the walking event being forecast')
    panel = history.loc[history.station_id.eq(STATION)].pivot(index='date', columns='hour', values='boardings')
    pre = panel.loc[(panel.index < donor_date) & ~panel.index.isin(exclusions)]
    dates = pre.loc[pre.index.day_of_week == donor_date.day_of_week].sort_index().tail(8)
    if len(dates) != 8 or donor_date not in panel.index:
        raise ValueError('The prior event and eight comparable dates require official observations')
    uplift = (panel.loc[donor_date] - dates.median()).clip(lower=0.)
    # Round the 2.5-hour difference in the schedule to the nearest hourly bin.
    shift = round(hour(donor.end_kst) - hour(target.end_kst))
    return WEIGHT * np.interp(HOURS + shift, uplift.index, uplift, left=0, right=0)


def apply_incheon_walk(history, target, prediction, origin, cal, event_keys, walks):
    """Apply the recorded 2026 race profile in count space and return its exact changed keys."""
    origin = pd.Timestamp(origin)
    prediction = np.asarray(prediction)
    if (history.date.gt(origin).any() or not target.date.gt(origin).all() or 'boardings' in target
            or len(prediction) != len(target) or not np.isfinite(prediction).all() or (prediction < 0).any()):
        raise ValueError('Expected pre-origin labels and aligned nonnegative future forecasts')
    donor, event = (walks.loc[f'incheon_love_walk_{year}'] for year in (2025, 2026))
    if donor.date > origin or event.date <= origin:
        raise ValueError('Expected one historical same-event example and one future event')
    station_dates = event_keys[event_keys.get_level_values('station_id') == STATION].get_level_values('date')
    if event.date in station_dates:
        raise ValueError('The target walking event already has a station-day forecast adjustment')
    excluded = pd.DatetimeIndex(cal.index[cal.is_holiday.eq(1)]).union(walks.date).union(station_dates)
    increments = forecast_increments(history, donor, event, excluded)
    mask = target.date.eq(event.date) & target.station_id.eq(STATION) & target.hour.isin(HOURS)
    test_rows = target.loc[mask, KEY]
    if (not np.array_equal(test_rows.hour, HOURS) or not np.isfinite(increments).all()
            or not (increments > 0).all()):
        raise ValueError('Expected positive changes for every hour in the target event window')
    output = prediction.copy()
    output[mask] += increments
    return output, test_rows.assign(increment=increments)


def run(args):
    if args.base.resolve() == args.output.resolve() or sha256(args.base) != args.base_sha256:
        raise ValueError('Expected a verified base forecast and a distinct output file')
    history, test = read_official(args.official)
    base = pd.read_csv(args.base, float_precision='round_trip')
    if list(base) != ['example_id', 'boardings'] or not base.example_id.equals(test.example_id):
        raise ValueError('Base submission keys must exactly match official test order')
    cal = pd.read_csv(args.calendar, usecols=['date', 'is_holiday'], parse_dates=['date']).set_index('date')
    events = pd.read_csv(args.events, usecols=['date', 'station_id'], parse_dates=['date'])
    walks = schedule(args.walks)
    output, patch = apply_incheon_walk(history, test[KEY], base.boardings.to_numpy(), CUTOFF,
                                       cal, pd.MultiIndex.from_frame(events), walks)
    lines = args.base.read_bytes().splitlines(keepends=True)
    for index in patch.index:
        lines[index + 1] = f'{test.at[index, "example_id"]},{output[index]!s}\n'.encode('ascii')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(b''.join(lines))
    rebuilt = pd.read_csv(args.output, float_precision='round_trip')
    if not np.array_equal(rebuilt.boardings, output) or not rebuilt.example_id.equals(test.example_id):
        raise ValueError('Candidate did not retain verified values and official order')
    audit = {'cutoff': str(CUTOFF.date()), 'rows': len(rebuilt), 'changed_rows': len(patch),
             'changes': json.loads(patch.to_json(orient='records', date_format='iso')),
             'inputs': {str(path): sha256(path) for path in (args.official, args.base, args.calendar, args.events, args.walks)},
             'sha256': sha256(args.output)}
    args.output.with_suffix('.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('official', 'base', 'calendar', 'events', 'output'):
        parser.add_argument(f'--{argument}', type=Path, required=True)
    parser.add_argument('--base-sha256', required=True)
    parser.add_argument('--walks', type=Path, default=HERE / 'incheon_walk.csv')
    run(parser.parse_args())
