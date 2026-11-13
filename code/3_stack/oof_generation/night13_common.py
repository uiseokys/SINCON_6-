"""Offline, read-only-source runtime for the preregistered round 13 calculation."""
from __future__ import annotations
import os
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = str(OUT/'tmp/mpl')
os.environ['TMPDIR'] = str(OUT/'tmp')
sys.dont_write_bytecode = True
(OUT/'tmp').mkdir(exist_ok=True)
READS = set()

def fence(event, args):
    def check(value):
        if isinstance(value, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(value)).resolve()
            if not any(path.is_relative_to(p) for p in (OUT, ROOT/'amp_delivery')) and str(path) != '/dev/null':
                raise PermissionError('round13 source write forbidden: '+str(path))
    if event == 'open':
        path, mode, flags = args
        if ((isinstance(mode, str) and any(c in mode for c in 'wax+')) or
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            check(path)
        elif isinstance(path, (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(path)).resolve()
            if p.is_file() and p.is_relative_to(ROOT) and not p.is_relative_to(OUT):
                READS.add(p)
    elif event in {'os.mkdir','os.remove','os.rmdir','os.chmod','os.utime'}:
        check(args[0])
    elif event in {'os.rename','os.link','os.symlink'}:
        check(args[0]); check(args[1])
    elif event.startswith('socket.'):
        raise PermissionError('round13 network forbidden: '+event)

sys.addaudithook(fence)
import hashlib
import importlib.util
import json
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

WORK = ROOT/'final_package/payload/work'
DATA = ROOT/'final_package/payload/team/data'
sys.path[:0] = [str(WORK), str(DATA.parent)]

def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()

def status(message):
    with (OUT/'STATUS.md').open('a') as f:
        f.write('\n## '+now()+'\n\n'+message+'\n')
    print(now(), message, flush=True)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def csv(frame,path):
    frame.to_csv(path,index=False,float_format='%.17g',mode='x',
        compression={'method':'gzip','mtime':0} if str(path).endswith('.gz') else None)

def read(path,**kwargs):
    return pd.read_csv(path,float_precision='round_trip',**kwargs)

def record_reads(path):
    dump(path,{str(p.relative_to(ROOT)):sha(p) for p in sorted(READS) if p.is_file()})
