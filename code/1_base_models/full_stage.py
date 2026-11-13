"""One isolated, fully configured model training stage for SINCON_final.ipynb."""
import argparse
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile


def main(args):
    import numpy as np
    import pandas as pd
    import torch
    work, team = args.payload / 'work', args.payload / 'team'
    sys.path[:0] = [str(work), str(team)]
    from sincon_station_state import KEY, WINDOWS
    torch.set_num_threads(args.threads)
    print('stage', args.stage, 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), flush=True)
    rolling, xgbdir = args.payload / 'rolling', args.payload / 'hourly_xgb'
    from full_train import sync_earlier
    earlier = sync_earlier(args.payload)

    def command(script, *values):
        subprocess.run([sys.executable, '-u', str(work / script), *map(str, values)], cwd=work, check=True)

    common = [team, work / 'external.zip', work / 'sincon_kbo_game_data.csv', work / 'mobility_monthly',
              work / 'sincon_station_addresses.csv', work / 'sincon_station_bus_traffic.csv',
              work / 'sincon_bucheon_football_data.csv', work / 'sincon_festival_station_days.csv',
              work / 'sincon_housing_approvals.csv', work / 'sincon_station_centers.csv']
    candidate_args = Namespace(**dict(zip(['team_dir', 'external_zip', 'scoreboard', 'mobility_directory',
                                'station_addresses', 'bus', 'football', 'events', 'housing', 'station_centers'], common)), threads=args.threads)
    if args.stage == 'native':
        destination = args.output / 'native_retrained'
        command('sincon_clean_native_forecasts.py', team, destination, '--threads', args.threads)
        for name in ('ensemble_parts.csv', 'valid_predictions_ABC.csv.gz'):
            shutil.copyfile(destination / name, team / name)
    elif args.stage == 'stacks':
        from sincon_local_candidate import LocalCandidate
        model = LocalCandidate(candidate_args)
        saved = []
        scores = {}
        for fold, (cutoff, end) in {**WINDOWS, 'final': ('2026-06-30', '2026-10-31')}.items():
            started = perf_counter()
            past = model.data.loc[model.data.date.le(cutoff)]
            held = model.data.loc[model.data.date.gt(cutoff) & model.data.date.le(end)]
            future = held.assign(boardings=np.nan)
            if fold == 'final':
                old = pd.read_csv(team / 'ensemble_parts.csv').set_index('example_id').reindex(future.example_id)
                neural = old.log_nn.to_numpy()
                reference = (old.log_lgb.to_numpy() + neural) / 2
            else:
                old = model.archive.loc[model.archive.fold.eq(fold)].set_index(KEY).reindex(pd.MultiIndex.from_frame(future[KEY]))
                neural, reference = old.nn_log.to_numpy(), old.ens_log.to_numpy()
            estimates = model.predict(past, future, neural, reference, cutoff, with_missing_test=fold != 'final')
            if fold == 'final':
                ids = pd.read_csv(team / 'data/test.csv').example_id
                for column in ('local_stack', 'conditional_stack'):
                    values = pd.Series(np.expm1(np.maximum(estimates[column], 0)), index=future.example_id).reindex(ids)
                    pd.DataFrame({'example_id': ids, 'boardings': values.to_numpy()}).to_csv(work / f'submission_{column}.csv', index=False)
            else:
                actual = np.log1p(held.boardings.to_numpy())
                saved.append(future[KEY].assign(fold=fold, y_log=actual, **estimates))
                scores[fold] = {n: float(np.sqrt(np.mean((actual - np.maximum(v, 0))**2))) for n, v in estimates.items()}
            print(fold, 'seconds', perf_counter() - started, scores.get(fold), flush=True)
            (args.output / 'stacks_scores.json').write_text(json.dumps(scores, indent=2) + '\n')
        pd.concat(saved, ignore_index=True).to_csv(work / 'sincon_local_validation.csv.gz', index=False, float_format='%.9f')
    elif args.stage == 'rolling':
        # Remove only the copied cache files, so the legacy resume check cannot skip training.
        for p in rolling.glob('origin_*.csv.gz'):
            p.unlink()
        command('sincon_rolling_forecasts.py', *common, rolling, '--threads', args.threads,
                '--window', '2025-06-30:2025-10-31', '--window', '2025-07-31:2025-11-30',
                '--window', '2025-08-31:2025-12-31', '--window', '2025-09-30:2026-01-31',
                '--window', '2026-03-31:2026-06-30', '--window', '2026-04-30:2026-06-30')
        command('sincon_summer_replay.py', *common, '--seeds', 30, '--threads', args.threads,
                '--results', args.output / 'strict_summer.json', '--predictions', work / 'sincon_summer_replay_validation.csv.gz')
        sync_earlier(args.payload)
    elif args.stage == 'hourly_xgb':
        spans = {**WINDOWS, 'summer': ('2025-06-30', '2025-10-31'),
                 '2025-07-31': ('2025-07-31', '2025-11-30'), '2025-08-31': ('2025-08-31', '2025-12-31'),
                 '2025-09-30': ('2025-09-30', '2026-01-31'), '2026-03-31': ('2026-03-31', '2026-06-30'),
                 '2026-04-30': ('2026-04-30', '2026-06-30'), 'final': ('2026-06-30', '2026-10-31')}
        validation = []
        for label, (cutoff, end) in spans.items():
            path = xgbdir / f'origin_{cutoff}.csv.gz'
            extra = ['--competition-test'] if label == 'final' else []
            command('sincon_zoo_transfer_season.py', team, path, '--origin', cutoff, '--end', end, *extra)
            if label in WINDOWS:
                validation.append(pd.read_csv(path).assign(fold=label))
        # Other zoo columns are unused by the clean path and remain archived for audit.
        zoo_path = team / 'experiments/valid_predictions_zoo.csv.gz'
        zoo = pd.read_csv(zoo_path).drop(columns='xgb')
        fresh = pd.concat(validation)[[*KEY, 'fold', 'xgb']]
        zoo.merge(fresh, on=[*KEY, 'fold'], validate='one_to_one').to_csv(zoo_path, index=False)
    elif args.stage == 'clean_oof':
        path = args.output / 'early_retrained.csv'
        raw_oof = args.output / 'early_oof.pkl'
        command('sincon_clean_15126_replay.py', team, earlier, xgbdir / 'origin_2026-06-30.csv.gz', path,
                args.output / 'early_retrained.json', '--validate', '--validation-archive', raw_oof,
                '--recent', rolling, '--extra-xgb', xgbdir, '--june-xgb', xgbdir / 'origin_2025-06-30.csv.gz',
                '--april-xgb', xgbdir / 'origin_2026-04-30.csv.gz')
        rows = pd.read_pickle(raw_oof)
        rows[[*KEY, 'label', 'origin', 'combined']].rename(columns={'combined': 'baseline'}).to_csv(
            args.cache / 'sincon_clean_event_oof.csv.gz', index=False, float_format='%.17g')
    elif args.stage == 'event_responses':
        # This baseline is needed only for reporting in the paired feature scenarios.
        baseline = args.output / 'early_retrained.csv'
        if not baseline.exists():
            command('sincon_clean_15126_replay.py', team, earlier, xgbdir / 'origin_2026-06-30.csv.gz', baseline,
                    args.output / 'early_response_only.json')
        response = args.output / 'event_responses'
        response.mkdir()
        command('sincon_munhak_cancellation_lgb.py', team, '--seeds', 3, '--threads', args.threads,
                '--report', response / 'sincon_munhak_cancellation_lgb_results.json')
        command('sincon_songdo_beach_location_lgb.py', team, rolling / 'origin_2025-07-31.csv.gz', baseline,
                '--threads', args.threads, '--report', response / 'sincon_songdo_beach_location_lgb_results.json')
        for seed in range(3):
            command('sincon_munhak_cancelled_nn.py', team, response / f'sincon_munhak_cancelled_nn_seed{seed}.csv',
                    '--seeds', seed, '--threads', args.threads)
            command('sincon_munhak_cancelled_nn.py', team, response / f'sincon_songdo_beach_nn_seed{seed}.csv',
                    '--seeds', seed, '--threads', args.threads, '--songdo-beach')
        for name, tree, prefix in [('sincon_munhak_cancellation_responses.zip', 'sincon_munhak_cancellation_lgb_results.json', 'sincon_munhak_cancelled_nn_seed'),
                                    ('sincon_songdo_beach_location_responses.zip', 'sincon_songdo_beach_location_lgb_results.json', 'sincon_songdo_beach_nn_seed')]:
            with ZipFile(work / name, 'w', ZIP_DEFLATED) as z:
                for n in [tree, *[f'{prefix}{seed}.csv' for seed in range(3)]]:
                    z.write(response / n, n)
    elif args.stage == 'tabicl':
        checkpoint = args.payload / 'models/tabicl-regressor-v2-20260212.ckpt'
        command('sincon_clean_tabicl_forecast.py', team, work / 'external.zip', checkpoint,
                args.cache / 'sincon_clean_tabicl_final.csv', '--folds', 'final', '--threads', args.threads)
        command('sincon_clean_tabicl_forecast.py', team, work / 'external.zip', checkpoint,
                args.cache / 'sincon_clean_tabicl_oof.csv', '--folds', 'A', 'B', 'C', 'summer', '2025-07-31',
                '2025-08-31', '2025-09-30', '2026-03-31', '2026-04-30', '--threads', args.threads)
    elif args.stage == 'vector':
        from sincon_archived_intraday_anomalies import vector_inputs, vector_signal
        source_args = Namespace(work_dir=work, team=team, external=work / 'external.zip')
        data = vector_inputs(source_args)
        cases, warm = [], []
        archive = pd.read_csv(work / 'sincon_local_validation.csv.gz', parse_dates=['date'])
        labels = {**{n: v[0] for n, v in WINDOWS.items()}, 'summer': '2025-06-30',
                  '2025-07-31': '2025-07-31', '2025-08-31': '2025-08-31', '2025-09-30': '2025-09-30',
                  '2026-03-31': '2026-03-31', '2026-04-30': '2026-04-30', 'final': '2026-06-30'}
        for label, cutoff in labels.items():
            if label == 'final':
                test = pd.read_csv(team / 'data/test.csv', parse_dates=['date'])
                reference = pd.read_csv(work / 'submission_conditional_stack.csv')
                source = test.merge(reference, on='example_id', validate='one_to_one')
                series = pd.Series(np.log1p(source.boardings.to_numpy()), index=pd.MultiIndex.from_frame(source[KEY]))
            else:
                if label in WINDOWS:
                    source = archive.loc[archive.fold.eq(label)]
                else:
                    path = work / 'sincon_summer_replay_validation.csv.gz' if label == 'summer' else rolling / f'origin_{cutoff}.csv.gz'
                    source = pd.read_csv(path, parse_dates=['date'])
                if 'conditional_stack' not in source:
                    with ZipFile(work / 'sincon_rolling_conditional_cache.zip') as frozen:
                        precursor = pd.read_csv(frozen.open(f'origin_{cutoff}.csv.gz'), compression='gzip', parse_dates=['date'])
                    if not precursor.origin.eq(cutoff).all():
                        raise ValueError('Frozen conditional precursor origin differs')
                    source = source.merge(precursor[[*KEY, 'conditional_stack']], on=KEY, how='left', validate='one_to_one')
                    if source.conditional_stack.isna().any():
                        raise ValueError('Frozen conditional precursor keys differ')
                series = source.set_index(KEY).conditional_stack
            signal = vector_signal(data, pd.Timestamp(cutoff), label, series)['vector_xgb'].rename('raw_delta').reset_index()
            if label == 'final':
                signal.to_csv(work / 'sincon_vector_shape_final_raw.csv.gz', index=False, float_format='%.17g')
            elif label in WINDOWS:
                signal['reference'] = series.reindex(pd.MultiIndex.from_frame(signal[KEY])).to_numpy()
                signal['shape'] = signal.reference + signal.raw_delta
                cases.append(signal)
            else:
                warm.append(signal.rename(columns={'fold': 'label'}))
            print('vector', label, len(signal), flush=True)
        pd.concat(cases, ignore_index=True).to_csv(work / 'sincon_vector_boost_validation.csv.gz', index=False, float_format='%.17g')
        pd.concat(warm, ignore_index=True).to_csv(args.cache / 'sincon_clean_vector_oof_warm.csv.gz', index=False, float_format='%.17g')
    else:
        raise ValueError(args.stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage')
    for name in ('payload', 'cache', 'output'):
        parser.add_argument(name, type=lambda v: Path(v).resolve())
    parser.add_argument('--threads', type=int, default=4)
    main(parser.parse_args())
