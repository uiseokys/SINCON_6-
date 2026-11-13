"""Portable wiring for the frozen SINCON stages. Only NumPy and pandas are required."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import hourbias_layer as hb
import stack_curvature as curvature
import stack_geomdan as geomdan
import stack_plave as plave
import stack_seasonal as seasonal
from stack_k1_archive import k1_archive

CUTOFF = pd.Timestamp('2026-06-30')
TOLERANCE = 1e-9
STAGES = [
    'submission_night_v4_main.csv',
    'submission_k1_hourbias.csv',
    'submission_k3_hourbias_holiday_events.csv',
    'submission_k3_geomdan_plave.csv',
    'submission_k3_geomdan_plave_seasonal_refit.csv',
    'submission_set2_l1_on_refit.csv',
    'submission_l1_hourly_curvature.csv',
    'submission_xgb7_w10_on_curvature.csv',
]
FINAL_SHA256 = '596876180bb00eda250041cc6cf8ada4c6aa4702547f7e14a94ffe1340e3b1a6'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_prediction(path, test, round_trip=False):
    frame = pd.read_csv(path, **({'float_precision': 'round_trip'} if round_trip else {}))
    hb.check_submission(frame, test)
    return frame


def patch_csv(source, output, test, values, selected):
    """Match the original overlay scripts: preserve unchanged lines and repr changed floats."""
    lines = Path(source).read_bytes().splitlines(keepends=True)
    assert len(lines) == len(test) + 1
    for i in np.flatnonzero(selected):
        lines[i + 1] = f'{test.example_id.iat[i]},{float(values[i])!r}\n'.encode('ascii')
    Path(output).write_bytes(b''.join(lines))
    written = read_prediction(output, test, round_trip=True)
    assert np.array_equal(written.boardings.to_numpy(), values)


def verify_stage(index, path, root, test):
    reference = Path(root) / 'reference' / STAGES[index]
    # Use the round-trip parser for verification to compare the actual decimal CSV values.
    frame = read_prediction(path, test, round_trip=True)
    ref = read_prediction(reference, test, round_trip=True)
    same_ids = np.array_equal(frame.example_id, ref.example_id)
    gap = float(np.max(np.abs(np.log1p(frame.boardings.to_numpy()) - np.log1p(ref.boardings.to_numpy()))))
    digest, expected = sha256(path), sha256(reference)
    result = dict(stage=index, file=STAGES[index], rows=len(frame), ids_and_order_match=same_ids,
                  max_abs_log1p_diff=gap, byte_identical=digest == expected,
                  sha256=digest, reference_sha256=expected)
    print(f'{index}: {STAGES[index]} | ID·순서={same_ids} | 최대 log1p 차이={gap:.17g} | 바이트 일치={digest == expected}', flush=True)
    assert same_ids and np.isfinite(gap) and gap <= TOLERANCE, result
    return result


def load_inputs(root):
    root = Path(root)
    train = pd.read_csv(root / 'data/train.csv', parse_dates=['date'])
    test = pd.read_csv(root / 'data/test.csv', parse_dates=['date'])
    assert train.date.max() == CUTOFF
    assert not {'boardings', 'alightings', 'y_log'} & set(test)
    assert len(test) == 157194 and not test.duplicated(hb.KEY).any()
    assert test.groupby(hb.DAYKEY).hour.size().eq(18).all()
    history = pd.read_csv(root / 'cache/hourbias_history_oof.csv.gz', parse_dates=['origin', 'date'])
    raw = pd.read_csv(root / 'cache/downstream_oof_nested_main.csv.gz',
                      usecols=['origin', *hb.KEY, 'y_log', 'lunar3122'], parse_dates=['origin', 'date'])
    truth = np.log1p(train.set_index(hb.KEY).boardings)
    for name, rows in [('hourbias_history_oof', history), ('downstream_oof_nested_main', raw)]:
        assert rows.date.le(CUTOFF).all() and rows.date.gt(rows.origin).all(), name
        assert not rows.duplicated(['origin', *hb.KEY]).any(), name
        actual = truth.reindex(pd.MultiIndex.from_frame(rows[hb.KEY])).to_numpy()
        assert np.isfinite(actual).all() and np.allclose(rows.y_log, actual, atol=1e-12, rtol=0), name
    raw = raw.rename(columns={'lunar3122': 'forecast'})
    calendar = pd.read_csv(root / 'cache/calendar_features.csv', parse_dates=['date']).set_index('date')
    calendar = calendar[seasonal.HOLIDAY_FLAGS].ne(0).any(axis=1)
    protected = pd.read_csv(root / 'cache/protected_event_keys.csv', parse_dates=['date'])
    hb_days, hb_events, hb_protected = hb.read_protection(root / 'cache')
    holidays = calendar.index[calendar].union(pd.DatetimeIndex(hb_days))
    events = hb_events.union(pd.MultiIndex.from_frame(protected[hb.DAYKEY]))
    return dict(root=root, train=train, test=test, history=history, raw=raw, calendar=calendar,
                protected=protected, hb_days=hb_days, hb_events=hb_events, hb_protected=hb_protected,
                holidays=holidays, events=events)


def audit_inputs(root, test):
    """Audit every machine-readable runtime input, including date-free prediction arrays."""
    root = Path(root)
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    rows = []
    for entry in manifest['files']:
        role = entry['role']
        if role in ('historical_evidence_snapshot', 'runtime_code_unchanged', 'training_code_unchanged'):
            continue
        path = root / entry['path']
        assert sha256(path) == entry['sha256'], entry['path']
        if path.suffix == '.npy':
            values = np.load(path, allow_pickle=False)
            assert values.shape == (len(test),) and np.isfinite(values).all(), entry['path']
            first, last = str(test.date.min().date()), str(test.date.max().date())
            label = '시험기간 예측, 실측 아님; test.csv 행 순서'
        else:
            df = pd.read_csv(path)
            date_cols = [c for c in ('date', 'origin', 'game_date') if c in df]
            dates = {c: pd.to_datetime(df[c]) for c in date_cols}
            first = '; '.join(f'{c}={d.min().date()}' for c, d in dates.items()) or str(test.date.min().date())
            last = '; '.join(f'{c}={d.max().date()}' for c, d in dates.items()) or str(test.date.max().date())
            if role in ('subway_observations', 'historical_oof_with_labels'):
                assert 'date' in dates and dates['date'].max() <= CUTOFF, entry['path']
                label = '승하차 실측 또는 실측 log 포함: 2026-06-30 이하 확인'
            elif role == 'unlabeled_test_keys':
                assert not {'boardings', 'alightings', 'y_log'} & set(df)
                label = '예측 대상 키, 승하차 실측 없음'
            elif role in ('calendar_or_event_metadata', 'archived_overlay_scope', 'training_exogenous_metadata', 'station_metadata'):
                assert not {'boardings', 'alightings', 'y_log'} & set(df), entry['path']
                label = ('외생 경기·관중·학교·날씨, 지하철 실측 아님' if role == 'training_exogenous_metadata' else '달력·행사 범위·역 메타데이터, 승하차 실측 없음')
                if not dates:
                    first, last = '날짜 없음', '날짜 없음'
            else:
                assert list(df) == ['example_id', 'boardings']
                assert np.array_equal(df.example_id, test.example_id)
                label = '시험기간 예측, 실측 아님; 날짜는 test.csv와 ID 연결'
        row = dict(path=entry['path'], role=role, min_date=first, max_date=last, interpretation=label)
        rows.append(row)
        print(entry['path'], '| 최대 날짜', last, '|', label)
    return rows


def build_geomdan_plave(ctx, baseline_path, k3_path, output):
    test, raw, root = ctx['test'], ctx['raw'], ctx['root']
    # Both the seasonal and weekday archive methods exclude station 3107.
    # Their transformed forecast is therefore exactly raw.forecast on every fitted lake row.
    allowed = ctx['calendar'].index[geomdan.quiet_days(ctx['calendar'].index, ctx['calendar'], ctx['protected'])]
    regime = geomdan.recent_regimes(raw, CUTOFF, allowed, pd.Timestamp('2025-06-28') + pd.Timedelta(days=84), [3107])[3107]
    lake = test.station_id.eq(3107) & test.date.isin(allowed)
    assert regime['sign'] == 1 and lake.sum() == 1872
    donor_history = plave.read_history(ctx['train'], root / 'cache/historical_concerts.csv', root / 'cache/downstream_oof_nested_main.csv.gz')
    folds, pooled = plave.backtest(donor_history)
    profile, donors = plave.fit(donor_history, CUTOFF, 'plave')
    concert = test.station_id.eq(3127) & test.date.isin(pd.to_datetime(['2026-09-12', '2026-09-13'])) & test.hour.isin([19, 20])
    assert tuple(test.loc[concert, 'example_id']) == ('J093667', 'J093668', 'J094945', 'J094946')
    assert not (lake & concert).any()
    base = read_prediction(baseline_path, test, round_trip=True).boardings.to_numpy()
    values = read_prediction(k3_path, test, round_trip=True).boardings.to_numpy().copy()
    mask = (lake | concert).to_numpy()
    assert np.max(np.abs(np.log1p(base[mask]) - np.log1p(values[mask]))) <= 1e-12
    values[lake] = np.expm1(np.log1p(base[lake]) + regime['effect'])
    change = (test.loc[concert, 'hour'] - 19).map(profile).to_numpy()
    values[concert] = np.expm1(np.log1p(base[concert]) + change)
    patch_csv(k3_path, output, test, values, mask)
    return dict(lake=regime, lake_rows=int(lake.sum()), plave_rows=int(concert.sum()),
                plave_profile={str(k): float(v) for k, v in profile.items()}, plave_donors=donors,
                plave_validation=folds, plave_pooled_validation=pooled)


def build_seasonal(ctx, baseline_path, k3_path, local_path, output):
    test = ctx['test']
    k3 = read_prediction(k3_path, test, round_trip=True).boardings.to_numpy()
    local = read_prediction(local_path, test, round_trip=True).boardings.to_numpy()
    # Frozen K2 membership is independent of permissible input-baseline rounding.
    overlay_keys = pd.read_csv(ctx['root'] / 'inputs/k2_overlay_keys.csv')
    k2_mask = test.example_id.isin(overlay_keys.example_id).to_numpy()
    local_mask = np.abs(np.log1p(local) - np.log1p(k3)) > 1e-12
    extra = pd.MultiIndex.from_frame(test.loc[k2_mask | local_mask, hb.DAYKEY].drop_duplicates())
    k1 = k1_archive(ctx['raw'], ctx['hb_days'], ctx['hb_events'])
    future = test.assign(forecast=np.log1p(k3))
    delta, days = seasonal.fit_correction(k1, future, CUTOFF, ctx['holidays'], ctx['events'].union(extra))
    active = delta != 0
    assert active.any() and not np.any(active & (k2_mask | local_mask))
    values = local.copy()
    values[active] = np.expm1(np.maximum(future.forecast.to_numpy()[active] + delta[active], 0.))
    patch_csv(local_path, output, test, values, active)
    return dict(fit_days=days, seasonal_rows=int(active.sum()), max_abs_log_change=float(np.abs(delta).max()))


def build_l1(ctx, baseline_path, k1_path, local_path, seasonal_path, output):
    test = ctx['test'].copy(); test['dow'] = test.date.dt.day_of_week
    main, k1, best, gp = [read_prediction(path, test) for path in (baseline_path, k1_path, seasonal_path, local_path)]
    rows = ctx['history'].copy(); rows['dow'] = rows.date.dt.day_of_week
    events = pd.MultiIndex.from_frame(rows[hb.DAYKEY]).isin(ctx['hb_events'])
    rows['quiet'] = ~rows.date.isin(ctx['hb_days']) & ~events
    stats = hb.fit_stats(hb.select_history(rows))
    frame = test[['example_id', *hb.KEY, 'dow']].copy()
    frame[hb.MODEL] = np.log1p(main.boardings.to_numpy())
    tk = pd.MultiIndex.from_frame(frame[hb.DAYKEY])
    quiet = ~frame.date.isin(ctx['hb_days']) & ~tk.isin(ctx['hb_events']) & ~tk.isin(ctx['hb_protected'])
    delta, good = hb.apply_stats(frame, stats, quiet)
    lm, lk, lb, lg = [np.log1p(f.boardings.to_numpy()) for f in (main, k1, best, gp)]
    touched = np.abs(lg - lk) > 1e-12
    touched_days = pd.MultiIndex.from_frame(frame.loc[touched, hb.DAYKEY].drop_duplicates())
    selected = good & frame.station_id.isin(hb.EXCLUDED_STATIONS).to_numpy() & ~tk.isin(touched_days) & (np.abs(lk - lm) <= 1e-12)
    assert frame.loc[selected].groupby(hb.DAYKEY).hour.size().eq(18).all()
    assert np.abs(pd.Series(delta[selected]).groupby([frame.date.to_numpy()[selected], frame.station_id.to_numpy()[selected]]).mean()).max() < 1e-12
    out = best.copy()
    out.loc[selected, 'boardings'] = np.expm1(lb[selected] + delta[selected])
    hb.check_submission(out, test)
    out.to_csv(output, index=False)
    return dict(selected_rows=int(selected.sum()), by_station=frame.loc[selected].groupby('station_id').date.nunique().to_dict())


def build_curvature(ctx, baseline_path, l1_path, output):
    test, root = ctx['test'], ctx['root']
    # Same union used by l1_hourly_curvature.run, including all three extra event guards.
    events = ctx['events']
    for name in ('events_daily.csv', 'festival_station_days.csv', 'protected_event_keys.csv'):
        guard = pd.read_csv(root / 'cache' / name, usecols=hb.DAYKEY, parse_dates=['date'])
        events = events.union(pd.MultiIndex.from_frame(guard))
    actual = ctx['train'].assign(log_count=np.log1p(ctx['train'].boardings))
    observed = curvature.complete_profiles(actual, 'log_count', ctx['holidays'], events)
    # candidate() protects exactly the frozen K2 station-days as well.
    overlay_keys = pd.read_csv(root / 'inputs/k2_overlay_keys.csv')
    changed = test.example_id.isin(overlay_keys.example_id)
    events = events.union(pd.MultiIndex.from_frame(test.loc[changed, hb.DAYKEY]).unique())
    values = read_prediction(l1_path, test, round_trip=True).boardings.to_numpy()
    future = test[hb.KEY].assign(forecast=np.log1p(values))
    delta = curvature.anchor(observed, future, CUTOFF, ctx['holidays'], events)
    active = delta != 0
    updated = values.copy()
    updated[active] = np.expm1(np.maximum(future.forecast.to_numpy()[active] + delta[active], 0.))
    patch_csv(l1_path, output, test, updated, active)
    return dict(changed_rows=int(active.sum()), max_abs_log_change=float(np.abs(delta).max()),
                observed_max_date=str(observed.index.get_level_values('date').max().date()))


def build_xgb_delta(ctx, curvature_path, output):
    test, root = ctx['test'], ctx['root']
    # Preserve the archived arrays' dtypes and the original NumPy operation order.
    folder = root / 'predictions'
    sub = 0.5 * np.load(folder / 'lgb_test.npy', allow_pickle=False) + 0.5 * np.mean(
        [np.load(folder / f'base_test_s{s}.npy', allow_pickle=False) for s in range(4)], 0)
    difference = np.load(folder / 'xgb_test.npy', allow_pickle=False) - sub
    keys = pd.MultiIndex.from_frame(test[hb.DAYKEY])
    selected = (test.station_id.isin(hb.EXCLUDED_STATIONS).to_numpy() & ~test.date.isin(ctx['hb_days']).to_numpy()
                & ~keys.isin(ctx['hb_events']) & ~keys.isin(ctx['hb_protected']))
    best = read_prediction(curvature_path, test)
    lb = np.log1p(best.boardings.to_numpy())
    out = best.copy()
    out.loc[selected, 'boardings'] = np.expm1(np.maximum(lb[selected] + 0.1 * difference[selected], 0.0))
    hb.check_submission(out, test)
    out.to_csv(output, index=False)
    return dict(weight=0.1, selected_rows=int(selected.sum()), max_abs_log_change=float(np.abs(0.1 * difference[selected]).max()))


def build_k3(ctx, k1, monday, cup_walk):
    """Transfer the original K2 rows, regardless of acceptable baseline rounding."""
    keys = pd.read_csv(ctx['root'] / 'inputs/k2_overlay_keys.csv')
    test = ctx['test']
    assert len(keys) == 2027 and not keys.example_id.duplicated().any()
    output = k1.copy()
    counts = {}
    for kind, donor in [('monday', monday), ('cup_walk', cup_walk)]:
        selected = test.example_id.isin(keys.loc[keys.overlay.eq(kind), 'example_id']).to_numpy()
        output.loc[selected, 'boardings'] = donor.boardings.to_numpy()[selected]
        counts[kind + '_rows'] = int(selected.sum())
    return output, counts
