"""Portable notebook runner. Input payload is read-only; every run has its own directory."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROWS = 157_194
KEY = ['date', 'station_id', 'hour']
DEFAULT_STAGES = dict(native=True, stacks=True, rolling=True, hourly_xgb=True,
                      clean_oof=True, event_responses=True, tabicl=True, vector=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locate(local_root=None):
    """Find an unpacked package, including Kaggle's arbitrary Dataset slug."""
    if local_root:
        p = Path(local_root).expanduser().resolve()
        if not (p / 'payload_manifest.json').is_file():
            raise FileNotFoundError(f'패키지 폴더가 아닙니다: {p}')
        return p
    roots = [Path.cwd(), Path('/kaggle/input'), Path('/content')]
    direct = [p for r in roots for p in (r, r / 'final_package') if (p / 'payload_manifest.json').is_file()]
    if direct:
        return direct[0].resolve()
    matches = []
    for r in roots[1:]:
        if r.exists():
            matches.extend(p.parent for p in r.rglob('payload_manifest.json'))
    if len(matches) != 1:
        raise ValueError('final_package를 풀고 LOCAL_ROOT 한 줄에 그 폴더를 지정하세요. 후보: ' + str(matches))
    return matches[0].resolve()


class Run:
    def __init__(self, package, mode='assemble', output_root=None, device='auto', threads=4):
        if mode not in {'assemble', 'full'} or device not in {'auto', 'cpu'}:
            raise ValueError('RUN_MODE는 assemble/full, DEVICE는 auto/cpu입니다.')
        self.package, self.mode = Path(package).resolve(), mode
        default = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/content') if Path('/content').exists() else self.package / 'outputs'
        root = Path(output_root or os.environ.get('SINCON_OUTPUT_DIR', default))
        self.output = (root / ('sincon_' + mode + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))).resolve()
        self.output.mkdir(parents=True)
        self.logs = self.output / 'logs'
        self.logs.mkdir()
        temporary = self.output / 'tmp'
        temporary.mkdir()
        self.env = {**os.environ, 'SINCON_DEVICE': device, 'PYTHONDONTWRITEBYTECODE': '1',
                    'PYTHONNOUSERSITE': '1', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': str(threads),
                    'TMPDIR': str(temporary), 'MPLCONFIGDIR': str(temporary / 'matplotlib')}
        self.env.pop('PYTHONPATH', None)
        if device == 'cpu':
            self.env['CUDA_VISIBLE_DEVICES'] = ''
        self.threads, self.records = int(threads), []
        manifest = json.loads((self.package / 'payload_manifest.json').read_text())['sha256']
        for name, expected in manifest.items():
            p = self.package / 'payload' / name
            if not p.is_file() or sha256(p) != expected:
                raise ValueError(f'입력 해시 불일치: {name}')
        self.payload = self.package / 'payload'
        if mode == 'full':
            self.payload = self.output / 'runtime'
            shutil.copytree(self.package / 'payload', self.payload)
        self.work, self.team = self.payload / 'work', self.payload / 'team'
        self.validate_official()
        cache = self.output / 'cache'
        cache.mkdir()
        with ZipFile(self.work / 'sincon_clean_refined_cache.zip') as z:
            hashes = json.loads(z.read('manifest.json'))
            for name, expected in hashes.items():
                data = z.read(name)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError('캐시 해시 불일치: ' + name)
                (cache / name).write_bytes(data)
        self.cache = cache
        self.xgb = cache / 'origin_2026-06-30.csv.gz'
        with ZipFile(self.work / 'sincon_xgb_forecasts.zip') as z:
            self.xgb.write_bytes(z.read(self.xgb.name))
        self.earlier = self.work / 'sincon_source_reweight_portable.zip'
        self.retrained_vector = False
        self.record('prepare', 'complete', 0., {'mode': mode, 'device': device, 'official_cutoff': '2026-06-30'})

    def validate_official(self):
        train = pd.read_csv(self.team / 'data/train.csv', usecols=[*KEY, 'boardings'])
        test = pd.read_csv(self.team / 'data/test.csv')
        if (len(train) != 688_176 or train.date.max() != '2026-06-30' or train.duplicated(KEY).any()
                or len(test) != ROWS or not test.example_id.is_unique or test.duplicated(KEY).any()
                or test.date.min() != '2026-07-01' or test.date.max() != '2026-10-31'
                or 'boardings' in test or not np.isfinite(train.boardings).all() or train.boardings.lt(0).any()):
            raise ValueError('공식 학습·예측 격자와 맞지 않습니다.')
        self.ids = test.example_id

    def record(self, stage, status, seconds, extra=None):
        self.records.append(dict(stage=stage, status=status, seconds=seconds, **(extra or {})))
        (self.output / 'run.json').write_text(json.dumps({'mode': self.mode, 'records': self.records}, indent=2, ensure_ascii=False) + '\n')

    def execute(self, stage, script, *args):
        path = Path(script)
        if not path.is_absolute():
            path = self.work / path
        log = self.logs / f'{stage}.log'
        start = perf_counter()
        command = [sys.executable, '-u', str(path), *map(str, args)]
        print(f'시작 {stage} | 로그 {log.name}', flush=True)
        with log.open('w') as stream:
            process = subprocess.Popen(command, cwd=self.work, env=self.env, stdout=stream, stderr=subprocess.STDOUT)
            self.record(stage, 'running', 0., {'pid': process.pid, 'started_utc': datetime.now(timezone.utc).isoformat(),
                                              'log': str(log.relative_to(self.output))})
            returncode = process.wait()
        elapsed = perf_counter() - start
        self.record(stage, 'complete' if returncode == 0 else 'failed', elapsed,
                    {'command': [str(path.relative_to(self.payload)) if path.is_relative_to(self.payload) else path.name, *map(str, args)], 'log': str(log.relative_to(self.output))})
        if returncode:
            print('\n'.join(log.read_text(errors='replace').splitlines()[-25:]))
            raise RuntimeError(f'{stage} 실패. {log}를 확인하세요.')
        print(f'완료 {stage}: {elapsed:.2f}초', flush=True)

    def train(self, stages=None):
        if self.mode != 'full':
            print('assemble: 모든 모델 학습을 생략하고 동결 모델 출력을 사용합니다.')
            return
        from full_train import run_full
        selected = DEFAULT_STAGES.copy()
        if stages is not None:
            if set(stages) - set(selected):
                raise ValueError('알 수 없는 학습 단계')
            selected.update(stages)
        run_full(self, selected)

    def assemble(self, corrections=None):
        if self.mode == 'full':
            from full_train import sync_earlier
            self.earlier = sync_earlier(self.payload)
        if self.mode == 'full' and getattr(self, 'full_stages', {}).get('native') and not self.full_stages.get('event_responses'):
            raise ValueError('native를 재학습한 결과로 조립하려면 event_responses도 켜세요. 행사 응답은 원본 예측에 종속됩니다.')
        early, events, model = (self.output / f'{name}.csv' for name in ('early', 'events', 'model'))
        protected = self.output / 'protected.csv'
        oof = self.cache / 'sincon_clean_event_oof.csv.gz'
        self.execute('01_early', 'sincon_clean_15126_replay.py', self.team, self.earlier, self.xgb, early, self.output / 'early.json')
        self.execute('02_events_before_model', 'sincon_clean_event_replay.py', self.team, early, oof, events,
                     self.output / 'events_before_model.json', '--stage', 'before_model', '--protected', protected)
        args = [self.team, events, protected, model, self.output / 'model.json', '--clean-oof', oof,
                '--oof-forecast-column', 'baseline', '--tabicl-daily', self.cache / 'sincon_clean_tabicl_final.csv',
                '--tabicl-oof', self.cache / 'sincon_clean_tabicl_oof.csv', '--vector', self.work / 'sincon_vector_shape_final_raw.csv.gz',
                '--vector-oof-warm', self.cache / 'sincon_clean_vector_oof_warm.csv.gz',
                '--validation-archive', self.output / 'model_oof.csv.gz']
        if self.retrained_vector:
            args.append('--allow-retrained-vector')
        self.execute('03_model_layers', 'sincon_clean_model_layers.py', *args)
        final = self.output / 'submission_clean_refined.csv'
        self.execute('04_events_after_model', 'sincon_clean_event_replay.py', self.team, model, oof, final,
                     self.output / 'events_after_model.json', '--stage', 'after_model', '--early-protected', protected,
                     '--protected', self.output / 'event_keys.csv')
        self.apply_corrections(final, corrections or [])
        self.final = final
        return final

    def apply_corrections(self, final, corrections):
        """Apply Claude-reviewed optional log offsets. Defaults do not add anything."""
        frame = pd.read_csv(final, float_precision='round_trip')
        records = []
        for item in corrections:
            if not item['enabled']:
                records.append({'name': item['name'], 'status': 'disabled'})
                continue
            evidence = self.package / item['evidence']
            facts = json.loads(evidence.read_text())
            required = {'status': 'validated', 'label_source': 'official_train', 'uses_public_score': False}
            if any(facts.get(k) != v for k, v in required.items()) or facts.get('validated_through', '9999') > '2026-06-30':
                raise ValueError('추가 보정에는 6월 30일 이전 공식 검증 근거가 필요합니다.')
            path = self.package / item['path']
            if sha256(path) != facts.get('offset_sha256'):
                raise ValueError('검증된 보정 파일과 SHA-256이 다릅니다.')
            offset = pd.read_csv(path)
            if list(offset) != ['example_id', 'log_delta'] or not offset.example_id.is_unique or not offset.example_id.isin(frame.example_id).all() or not np.isfinite(offset.log_delta).all():
                raise ValueError('추가 보정은 고유한 공식 ID와 유한한 log_delta여야 합니다.')
            delta = frame.example_id.map(offset.set_index('example_id').log_delta).fillna(0.)
            frame['boardings'] = np.expm1(np.maximum(np.log1p(frame.boardings) + delta, 0.))
            records.append({'name': item['name'], 'status': 'applied', 'rows': len(offset), 'evidence_sha256': sha256(evidence)})
        if any(r['status'] == 'applied' for r in records):
            frame.to_csv(final, index=False, float_format='%.17g')
        (self.output / 'corrections.json').write_text(json.dumps(records, indent=2, ensure_ascii=False) + '\n')

    def verify(self):
        result = pd.read_csv(self.final, float_precision='round_trip')
        reference = pd.read_csv(self.package / 'reference/submission_clean_refined.csv', float_precision='round_trip')
        if (list(result) != ['example_id', 'boardings'] or len(result) != ROWS or not result.example_id.equals(self.ids)
                or not np.isfinite(result.boardings).all() or result.boardings.lt(0).any()
                or not reference.example_id.equals(self.ids)):
            raise ValueError('최종 형식 검사 실패')
        diff = np.log1p(result.boardings) - np.log1p(reference.boardings)
        versions = {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'torch', 'xgboost', 'tabicl']}
        report = {'rows': len(result), 'columns': list(result), 'official_id_order': True, 'finite_nonnegative': True,
                  'max_abs_log1p_difference': float(np.abs(diff).max()), 'rms_log1p_difference': float(np.sqrt(np.mean(diff**2))),
                  'max_abs_boardings_difference': float(np.abs(result.boardings - reference.boardings).max()),
                  'reference_sha256': sha256(self.package / 'reference/submission_clean_refined.csv'),
                  'output_sha256': sha256(self.final), 'python': sys.version, 'packages': versions,
                  'mode': self.mode, 'device_requested': self.env['SINCON_DEVICE'],
                  'loaded_numpy': np.__version__, 'loaded_pandas': pd.__version__}
        (self.output / 'verification.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print('제출 CSV:', self.final)
        return report
