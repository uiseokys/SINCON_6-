"""Rebuild the Fire Baseball uplift from official history and apply it to a verified base CSV."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from firebaseball import KEY, SPORTS_INPUTS, apply_firebaseball, event_schedule

HERE = Path(__file__).resolve().parent
CUTOFF = pd.Timestamp('2026-06-30')


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_official(path):
    with ZipFile(path) as archive:
        train = pd.read_csv(archive.open('train.csv'), usecols=[*KEY, 'boardings'], parse_dates=['date'])
        test = pd.read_csv(archive.open('test.csv'), parse_dates=['date'])
    if (len(train) != 688176 or train.date.max() != CUTOFF or train.duplicated(KEY).any()
            or len(test) != 157194 or list(test) != ['example_id', *KEY] or test.duplicated(KEY).any()
            or not test.example_id.is_unique or not test.date.gt(CUTOFF).all()):
        raise ValueError('Unexpected official competition data')
    return train, test


def run(args):
    source_files = (args.official, args.base, args.calendar, args.events, args.games,
                    *(args.sports / file for file in SPORTS_INPUTS))
    if args.base.resolve() == args.output.resolve():
        raise ValueError('Output must differ from the base prediction file')
    if sha256(args.base) != args.base_sha256:
        raise ValueError('Base forecast does not match the approved SHA-256')
    train, test = read_official(args.official)
    base = pd.read_csv(args.base, float_precision='round_trip')
    if list(base) != ['example_id', 'boardings'] or not base.example_id.equals(test.example_id):
        raise ValueError('Base submission keys must exactly match official test order')
    cal = pd.read_csv(args.calendar, usecols=['date', 'is_holiday'], parse_dates=['date']).set_index('date')
    events = pd.read_csv(args.events, parse_dates=['date'])
    event_keys = pd.MultiIndex.from_frame(events[KEY[:2]])
    values, patch = apply_firebaseball(train, test[KEY], base.boardings.to_numpy(), CUTOFF,
                                      cal, event_keys, args.sports, event_schedule(args.games))
    changed = values != base.boardings.to_numpy()
    if (patch.empty or changed.sum() != len(patch)
            or not test.loc[changed, KEY].equals(patch[KEY]) or not np.isfinite(values).all()):
        raise ValueError('Forecast differences do not match the audited event patch')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('example_id,boardings\n' + ''.join(
        f'{example_id},{value!r}\n' if replace else original
        for example_id, value, replace, original in zip(
            base.example_id, values.tolist(), changed, args.base.read_text().splitlines(keepends=True)[1:],
            strict=True)), encoding='utf-8')
    rebuilt = pd.read_csv(args.output, float_precision='round_trip')
    if (not rebuilt.example_id.equals(test.example_id)
            or not np.array_equal(rebuilt.boardings.to_numpy(), values)
            or not np.array_equal(rebuilt.loc[~changed, 'boardings'], base.loc[~changed, 'boardings'])):
        raise ValueError('Serialized candidate differs from the verified predictions')
    report = {'cutoff': str(CUTOFF.date()), 'rows': len(rebuilt), 'changed_rows': int(changed.sum()),
              'change': json.loads(patch.to_json(orient='records', date_format='iso')),
              'input_sha256': {str(path): sha256(path) for path in source_files},
              'output_sha256': sha256(args.output)}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('official', 'base', 'calendar', 'events', 'sports', 'output'):
        parser.add_argument(f'--{argument}', type=Path, required=True)
    parser.add_argument('--base-sha256', required=True)
    parser.add_argument('--games', type=Path, default=HERE / 'firebaseball.csv')
    print(json.dumps(run(parser.parse_args()), indent=2))
