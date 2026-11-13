"""Execute every notebook code cell offline, saving evidence outside the package.

Uses sequential compile/exec, not a Jupyter kernel. No cell is skipped.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--explicit-root',action='store_true')
    parser.add_argument('--baseline',default='')
    args=parser.parse_args();package=args.package.resolve();output=args.output.resolve()
    if output.is_relative_to(package):raise ValueError('Verification output must be outside the read-only package')
    output.mkdir(parents=True,exist_ok=False)
    paths=[p for p in package.rglob('*') if p.is_file()]
    before={str(p.relative_to(package)):sha(p) for p in paths}
    readonly=all(not(p.stat().st_mode&0o222) for p in [package,*paths])
    roots=[package,Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve(),output]
    sys.dont_write_bytecode=True
    os.environ['SINCON_MONDAY_OUTPUT_DIR']=str(output/'results')
    def check_write(value):
        if isinstance(value,(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(value)).resolve()
            if not p.is_relative_to(output) and str(p)!='/dev/null':raise PermissionError('Write outside test output: '+str(p))
    def audit(event,args):
        if event.startswith('socket.'):raise PermissionError('Network and sockets are forbidden')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
            if writing:check_write(args[0])
            elif str(p)!='/dev/null' and not any(p.is_relative_to(root) for root in roots):raise PermissionError('Read outside isolated package/env: '+str(p))
        elif event in {'os.mkdir','os.remove','os.rmdir','os.chmod','os.utime'}:check_write(args[0])
        elif event in {'os.rename','os.link','os.symlink'}:check_write(args[0]);check_write(args[1])
    sys.addaudithook(audit)
    notebook_path=package/'SINCON_monday.ipynb'
    notebook=json.loads(notebook_path.read_text());executed=copy.deepcopy(notebook)
    namespace={'__name__':'__main__'};records=[];count=0;success=True
    start=perf_counter()
    os.chdir(package)
    for index,cell in enumerate(executed['cells']):
        if cell['cell_type']!='code':continue
        count+=1;stream=io.StringIO();begin=perf_counter();ok=True
        try:
            with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                exec(compile(''.join(cell['source']),f'SINCON_monday.ipynb:cell_{index}','exec'),namespace)
                if 'settings' in cell.get('metadata',{}).get('tags',[]):
                    if args.explicit_root:namespace['ROOT']=str(package)
                    if args.baseline:namespace['BASELINE_CSV']=args.baseline
        except Exception:
            stream.write(traceback.format_exc());success=False;ok=False
        cell['execution_count']=count
        cell['outputs']=[dict(output_type='stream',name='stdout',text=stream.getvalue().splitlines(keepends=True))]
        records.append(dict(cell=index,success=ok,seconds=perf_counter()-begin))
        if not success:break
    after={str(p.relative_to(package)):sha(p) for p in paths}
    report=dict(success=success and before==after,all_code_cells_executed=count==sum(c['cell_type']=='code' for c in notebook['cells']),
        code_cells_executed=count,execution_backend='standard-library sequential compile/exec; no Jupyter kernel',
        package_readonly_permissions=readonly,inputs_unchanged=before==after,
        network_and_socket_blocked=True,original_project_reads_blocked=True,
        notebook_sha256=sha(notebook_path),python=sys.version,system=platform.system(),machine=platform.machine(),
        explicit_root=args.explicit_root,baseline_override=bool(args.baseline),seconds=perf_counter()-start,cells=records)
    if 'result' in namespace:report['result']=namespace['result']
    (output/'execution.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (output/'notebook.executed.ipynb').write_text(json.dumps(executed,ensure_ascii=False,indent=2)+'\n')
    (output/'input_hashes.json').write_text(json.dumps(before,ensure_ascii=False,indent=2)+'\n')
    log='\n'.join(''.join(c['outputs'][0]['text']) for c in executed['cells'] if c.get('outputs'))
    (output/'execution.log').write_text(log)
    print(json.dumps({k:v for k,v in report.items() if k not in ['cells','result']},ensure_ascii=False,indent=2))
    if not report['success']:
        print(log[-5000:]);raise SystemExit(1)

if __name__=='__main__':main()
