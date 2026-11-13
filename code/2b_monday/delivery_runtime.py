"""Offline second-stage Monday package. No training, upload, or network calls."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
from datetime import datetime,timezone
import numpy as np
import pandas as pd

ROWS=157194
KIND='sincon_monday_holiday_preregistered_v1'
MANIFEST='monday_delivery_manifest.json'
KEY=['date','station_id','hour']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path,**kwargs):return pd.read_csv(path,float_precision='round_trip',**kwargs)
def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def bounded(root,filename,depth,skip=()):
    root=Path(root);found=[]
    if not root.is_dir() or root.is_symlink():return found
    for directory,children,files in os.walk(root,followlinks=False):
        p=Path(directory);level=len(p.relative_to(root).parts)
        children[:]=sorted(n for n in children if n not in skip and level<depth and not (p/n).is_symlink())
        if filename in files and not (p/filename).is_symlink():found.append(p/filename)
    return found

def verify_inputs(package):
    manifest=json.loads((package/MANIFEST).read_text())
    if manifest.get('package_kind')!=KIND:raise ValueError('다른 패키지 manifest입니다')
    for name,expected in manifest['files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid manifest path')
        if sha(package/relative)!=expected:raise ValueError('동봉 파일 해시 불일치: '+name)
    return manifest

def find_data(package):
    expected=json.loads((package/'official_inputs.json').read_text())
    candidates=[package/'data',package,Path.cwd()/'data']
    for root,depth,skip in [(Path('/content'),3,('drive',)),(Path('/kaggle/input'),4,())]:
        candidates.extend(p.parent for p in bounded(root,'train.csv',depth,skip))
    for directory in dict.fromkeys(candidates):
        if all((directory/name).is_file() for name in expected):
            if all(sha(directory/name)==v['sha256'] for name,v in expected.items()):return directory.resolve()
    raise FileNotFoundError('공식 train.csv·test.csv·metaData.csv를 ROOT/data/에 두세요. /content 깊이3·drive 제외, /kaggle/input 깊이4까지 탐색합니다.')

def validate(frame,grid):
    if list(frame)!=['example_id','boardings'] or len(frame)!=ROWS or not frame.example_id.is_unique:
        raise ValueError('157,194행과 고유 example_id,boardings 열이 필요합니다')
    if not frame.example_id.equals(grid.example_id):raise ValueError('공식 example_id 순서 불일치')
    if not np.isfinite(frame.boardings).all() or frame.boardings.lt(0).any():raise ValueError('유한한 비음수 예측이 필요합니다')

def prepare(package,baseline_override='',output_override=''):
    package=Path(package).resolve();manifest=verify_inputs(package);data=find_data(package)
    grid=read(data/'test.csv',parse_dates=['date'])
    reference=package/'reference/submission_clean_refined.csv'
    baseline=Path(baseline_override).expanduser() if baseline_override else (
        package/'input/submission_clean_refined.csv' if (package/'input/submission_clean_refined.csv').is_file() else reference)
    if not baseline.is_absolute():baseline=package/baseline
    clean=read(baseline);ref=read(reference);validate(clean,grid);validate(ref,grid)
    error=float(abs(np.log1p(clean.boardings)-np.log1p(ref.boardings)).max())
    if error>1e-9:raise ValueError('1단계 CSV 최대 log1p 차이가 1e-9를 넘습니다')
    exact=sha(baseline)==manifest['clean_sha256']
    record=dict(accepted=True,acceptance='sha256_exact' if exact else 'numeric_equivalence',
        supplied_path=str(baseline),input_sha256=sha(baseline),reference_sha256=manifest['clean_sha256'],
        max_abs_log1p_difference=error,tolerance=1e-9,
        numeric_equivalent_input_normalized_to_reference=True,
        reason='Preserve the original six layers including their exact-hash input contract; equivalent inputs share the frozen numeric reference.')
    root=Path(output_override or os.environ.get('SINCON_MONDAY_OUTPUT_DIR','') or (
        '/kaggle/working' if Path('/kaggle/working').is_dir() else '/content' if Path('/content').is_dir()
        else str(Path.cwd().parent/'monday_outputs'))).expanduser().resolve()
    if root.is_relative_to(package) or root.is_relative_to(Path('/kaggle/input')):
        raise ValueError('출력 폴더는 읽기 전용 패키지와 /kaggle/input 밖이어야 합니다')
    output=root/('sincon_monday_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('sincon_monday_final_layers',package/'final_layers.py')
    final=importlib.util.module_from_spec(spec);spec.loader.exec_module(final)
    return dict(package=package,data=data,grid=grid,baseline=reference,output=output,final=final,
        baseline_verification=record,manifest=manifest,
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
            system=platform.system(),machine=platform.machine(),randomness='none; deterministic arithmetic; no seed consumed',
            threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']},gpu_used=False))

def build_final(context):
    output,layers,report=context['final'].build_submission(context['data'],context['package']/'cache',context['baseline'])
    validate(output,context['grid'])
    return output,layers,report

def check_monday_disabled(context):
    final=context['final'];enabled=set(final.LAYER_NAMES)-{'monday'}
    output,_,_=final.build_submission(context['data'],context['package']/'cache',context['baseline'],enabled)
    reference=read(context['package']/'reference/submission_round7_main.csv');validate(reference,context['grid'])
    error=float(abs(np.log1p(output.boardings)-np.log1p(reference.boardings)).max())
    if error>1e-10:raise ValueError('monday 비활성화 주후보 재현 실패')
    context['disabled_verification']=dict(passed=True,max_abs_log1p_difference=error,tolerance=1e-10)
    return context['disabled_verification']

def finish(context,output,layers,layer_report):
    package=context['package'];folder=context['output'];grid=context['grid']
    target=folder/'submission_monday_main.csv'
    output.to_csv(target,index=False,float_format='%.17g',mode='x')
    saved=read(target);validate(saved,grid)
    if not np.array_equal(saved.boardings,output.boardings):raise ValueError('CSV roundtrip changed values')
    reference=read(package/'reference/submission_monday_main.csv');validate(reference,grid)
    error=float(abs(np.log1p(saved.boardings)-np.log1p(reference.boardings)).max())
    if error>1e-10:raise ValueError('후보 재현 최대 log1p 차이가 1e-10을 넘습니다')
    main=read(package/'reference/submission_round7_main.csv')
    diff=np.log1p(saved.boardings)-np.log1p(main.boardings)
    changed=abs(diff)>1e-12
    dates=pd.to_datetime(['2026-08-17','2026-10-05'])
    if not grid.loc[changed,'date'].isin(dates).all():raise ValueError('대상 밖 변화')
    if abs(diff).max()>.05+1e-12:raise ValueError('최대 변화 0.05 초과')
    centered=grid[KEY].assign(delta=diff).groupby(['date','hour']).delta.sum()
    if centered.abs().max()>1e-12:raise ValueError('시간대별 역 합이 0이 아님')
    layers.to_csv(folder/'layer_deltas.csv.gz',index=False,float_format='%.17g',mode='x',compression={'method':'gzip','mtime':0})
    dump(folder/'final_layers.json',layer_report)
    report=dict(success=True,rows=len(saved),official_id_order=True,finite_nonnegative=True,
        candidate_sha256=sha(target),max_abs_log1p_difference=error,reproduction_tolerance=1e-10,
        changed_rows_gt_1e_12=int(changed.sum()),max_abs_log_difference_from_main=float(abs(diff).max()),
        hour_station_sum_max_abs=float(centered.abs().max()),baseline_verification=context['baseline_verification'],
        monday_disabled=context['disabled_verification'],final_layer_order=list(context['final'].LAYER_NAMES),
        environment=context['environment'],output=str(folder))
    dump(folder/'verification.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report
