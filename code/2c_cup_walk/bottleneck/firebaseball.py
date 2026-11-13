"""Transfer a start-time-aligned profile between documented Landers Field games."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPORTS = HERE.parents[1] / 'night6' / 'resourced' / 'independently_sourced'
KEY = ['date', 'station_id', 'hour']
HOURS = np.arange(6, 24)
STATION = 3127
SHRINKAGE = .75
SPORTS_INPUTS = {
    'kbo_result_178_actual_clock_cc.csv': 'game_date',
    'kbo_160_main_wiki_cc.csv': 'date',
    'ssg_special_two_cc.csv': 'date',
}


def event_schedule(path=HERE / 'firebaseball.csv'):
    games = pd.read_csv(path, parse_dates=['date'])
    if not (games.date.is_unique and games.station_id.eq(STATION).all()
            and games.license.eq('CC BY-NC-SA 2.0 KR').all()
            and games.start_hour.between(6, 23).all()):
        raise ValueError('Unexpected event schedule or license')
    return games.sort_values('date')


def ordinary_day_exclusions(games, cal, event_keys, sports_path):
    sports_dates = []
    for name, field in SPORTS_INPUTS.items():
        table = pd.read_csv(sports_path / name, usecols=[field, 'license'], parse_dates=[field])
        if not table.license.eq('CC BY-NC-SA 2.0 KR').all():
            raise ValueError('Baseball exclusion dates require reviewed source terms')
        sports_dates.append(table[field])
    station_events = event_keys[event_keys.get_level_values('station_id') == STATION]
    other_events = [pd.Series(station_events.get_level_values('date')), *sports_dates]
    if games.date.isin(pd.concat(other_events, ignore_index=True)).any():
        raise ValueError('A Fire Baseball date is already covered by the stadium calendar')
    excluded = [games.date, pd.Series(cal.index[cal.is_holiday]), *other_events]
    return pd.DatetimeIndex(pd.concat(excluded, ignore_index=True).drop_duplicates())


def event_increments(history, target, origin, games, exclusions):
    """Return increments in target order using official labels through origin."""
    origin = pd.Timestamp(origin)
    if history.date.gt(origin).any() or not target.date.gt(origin).all() or 'boardings' in target:
        raise ValueError('Expected pre-origin history and an unlabeled forecast grid')
    panel = (history.loc[history.station_id.eq(STATION)]
             .pivot(index='date', columns='hour', values='boardings').sort_index())
    ordinary = panel.loc[~panel.index.isin(exclusions)]
    donors = []
    for game in games.loc[games.date.le(origin)].itertuples(index=False):
        nearby = ordinary.loc[(ordinary.index < game.date)
                             & (ordinary.index.day_of_week == game.date.day_of_week)].tail(8)
        if len(nearby) >= 4:
            uplift = (panel.loc[game.date] - nearby.median()).clip(lower=0)
            donors.append((game.start_hour, uplift.reindex(HOURS).to_numpy()))

    increments = np.zeros(len(target))
    if not donors:
        return increments
    for game in games.loc[games.date.gt(origin)].itertuples(index=False):
        eligible = target.station_id.eq(STATION) & target.date.eq(game.date)
        hours = target.loc[eligible, 'hour'].to_numpy()
        aligned = [np.interp(hours + start - game.start_hour, HOURS, uplift, left=0, right=0)
                   for start, uplift in donors]
        # Leave the baseline intact before opening and transfer only the post-start profile.
        increments[eligible] = SHRINKAGE * np.mean(aligned, axis=0) * (hours >= game.start_hour)
    return increments


def apply_firebaseball(history, target, prediction, origin, cal, event_keys, sports_path, games):
    """Apply the independently sourced profile to the count-space submission."""
    prediction = np.asarray(prediction)
    if len(prediction) != len(target) or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError('Expected finite nonnegative forecasts aligned with the target grid')
    exclusions = ordinary_day_exclusions(games, cal, event_keys, sports_path)
    additions = event_increments(history, target, origin, games, exclusions)
    patch = target.loc[additions > 0, KEY].assign(increment=additions[additions > 0])
    return prediction + additions, patch


def rmsle(log_actual, log_prediction):
    return float(np.sqrt(np.mean((log_actual - log_prediction) ** 2)))


def main():
    from recon import rows
    from weather_residual import TEAM

    games = event_schedule()
    all_labels = pd.read_csv(TEAM / 'data/train.csv', parse_dates=['date'])
    report = {}
    for fold, history, frame, _, cal, event_keys in rows():
        counts = np.expm1(frame.combined)
        estimates, patch = apply_firebaseball(history, frame[KEY], counts, frame.origin.iloc[0],
                                               cal, event_keys, SPORTS, games)
        corrected = pd.Series(np.log1p(estimates), index=frame.index)
        scoped = frame.station_id.eq(STATION) & frame.date.isin(games.date)
        if fold == 'final':
            if (patch.empty or patch.date.nunique() != 1 or patch.date.iloc[0] != pd.Timestamp('2026-10-31')
                    or patch.duplicated(KEY).any() or not np.isfinite(patch.increment).all()):
                raise ValueError('Unexpected final event adjustment')
            patch.to_csv(HERE / 'firebaseball_patch.csv', index=False)
            report['final'] = {'changed_rows': len(patch), 'added_boardings': float(patch.increment.sum())}
            continue
        report[fold] = {
            'baseline': rmsle(frame.y, frame.combined), 'candidate': rmsle(frame.y, corrected),
            'changed_rows': len(patch),
            'affected_days': [{
                'date': str(day.date()), 'base': rmsle(event.y, event.combined),
                'candidate': rmsle(event.y, corrected.loc[event.index]),
            } for day, event in frame.loc[scoped].groupby('date')],
        }
        if fold == 'summer':
            # Supplement fixed-origin folds with prior-game-only tests, while keeping
            # their ordinary baseline fixed at the original validation origin.
            report['2025_next_game'] = []
            exclusions = ordinary_day_exclusions(games, cal, event_keys, SPORTS)
            for day, event in frame.loc[scoped].groupby('date'):
                origin = day - pd.Timedelta(days=1)
                extra = event_increments(all_labels.loc[all_labels.date.le(origin)], event[KEY],
                                         origin, games, exclusions)
                report['2025_next_game'].append({
                    'date': str(day.date()), 'changed_rows': int((extra > 0).sum()),
                    'base': rmsle(event.y, event.combined),
                    'candidate': rmsle(event.y, np.log1p(np.expm1(event.combined) + extra)),
                })
    (HERE / 'firebaseball_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
