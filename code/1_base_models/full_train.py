"""Run the full training graph in order; disabled stages explicitly retain caches."""
from pathlib import Path
import shutil
from zipfile import ZipFile

EARLY_ORIGINS = ('2025-06-30', '2025-07-31', '2025-08-31', '2025-09-30')


def sync_earlier(payload):
    """Keep the original four late-profile fitting origins separate from 2026 validation."""
    directory = payload / 'earlier'
    directory.mkdir(exist_ok=True)
    expected = {f'origin_{origin}.csv.gz' for origin in EARLY_ORIGINS}
    unexpected = {p.name for p in directory.glob('origin_*.csv.gz')} - expected
    if unexpected:
        raise ValueError(f'Late-profile calibration must use only the four 2025 origins: {unexpected}')
    for name in sorted(expected):
        source, target = payload / 'rolling' / name, directory / name
        temporary = target.with_suffix('.tmp')
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    return directory


def run_full(run, stages):
    run.full_stages = stages.copy()
    # Recover every historical precursor first. Rebuilt stages replace these files.
    rolling = run.payload / 'rolling'
    rolling.mkdir()
    import sys
    sys.path.insert(0, str(run.work))
    from sincon_clean_15126_replay import restore_history
    restore_history(run.work / 'sincon_source_reweight_portable.zip', rolling, run.team)
    with ZipFile(run.work / 'sincon_target_season_validation.zip') as z:
        for n in z.namelist():
            (rolling / n).write_bytes(z.read(n))
    run.earlier = sync_earlier(run.payload)
    xgb = run.payload / 'hourly_xgb'
    xgb.mkdir()
    for name in ('sincon_xgb_forecasts.zip', 'sincon_xgb_extra_validation.zip'):
        with ZipFile(run.work / name) as z:
            for n in z.namelist():
                if n.endswith('.csv.gz'):
                    (xgb / n).write_bytes(z.read(n))
    run.xgb = xgb / 'origin_2026-06-30.csv.gz'
    stage_script = Path(__file__).resolve().parent / 'full_stage.py'
    for stage, enabled in stages.items():
        if not enabled:
            run.record('full_' + stage, 'cached_not_retrained', 0.)
            print(f'학습 생략 {stage}: 동결 캐시를 사용합니다.', flush=True)
            continue
        run.execute('full_' + stage, stage_script, stage, run.payload, run.cache, run.output, '--threads', run.threads)
        if stage == 'vector':
            run.retrained_vector = True
    run.record('training_coverage', 'all_requested_stages_complete', 0.,
               {'stages': stages, 'strict_full': all(stages.values()),
                'note': '한 단계라도 꺼져 있으면 모든 모델을 재학습한 실행이 아닙니다.'})
