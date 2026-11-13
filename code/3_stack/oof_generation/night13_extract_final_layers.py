"""Run the original stage-1 assemble path, observing each model apply call.

Only the copied downstream script gains instrumentation; model arithmetic and
adoption gates are unchanged. The original pipeline and payload are read-only.
"""
from common import *
import argparse
import contextlib
import runpy
import shutil
from time import perf_counter

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT/'stage1')
    args=parser.parse_args(); target=args.output.resolve()
    if not target.is_relative_to(OUT):raise ValueError('Output must be under night13')
    target.mkdir(parents=True,exist_ok=False)
    copied=target/'work'
    shutil.copytree(WORK,copied)
    path=copied/'sincon_clean_model_layers.py'
    text=path.read_text()
    old='    def apply(name, candidate, evidence=None):'
    text=text.replace(old,"    layer_changes = {}\n    initial_final = result.baseline.to_numpy(copy=True)\n\n"+old,1)
    old='        result["baseline"] = candidate.clip(min=0.)'
    text=text.replace(old,'        layer_changes[name] = candidate.clip(min=0.) - previous\n'+old,1)
    old='    changed = result.baseline.ne(np.log1p(base.boardings)).to_numpy()'
    new='''    traced = result.copy()
    traced['initial'] = initial_final
    traced['amp_eligible'] = ordinary(result, protected)
    for name, delta in layer_changes.items():
        traced[name] = delta
    traced.to_csv(args.report.parent / 'final_model_layer_changes.csv.gz', index=False,
                  float_format='%.17g', compression={'method':'gzip','mtime':0})
'''+old
    assert old in text
    text=text.replace(old,new,1)
    path.write_text(text)
    dump(target/'instrumentation.json',dict(source_sha256=sha(WORK/path.name),
        instrumented_sha256=sha(path),purpose='observe initial, actual apply deltas, final ordinary mask; arithmetic unchanged'))
    pipeline=module('round13_pipeline',ROOT/'final_package/pipeline.py')
    runner=pipeline.Run(ROOT/'final_package',mode='assemble',output_root=target/'runs',device='cpu',threads=1)
    runner.work=copied
    sys.path.insert(0,str(copied))
    def execute(stage,script,*arguments):
        p=Path(script)
        if not p.is_absolute():p=copied/p
        started=perf_counter(); log=runner.logs/(stage+'.log')
        status('1단계 assemble '+stage+' 실행')
        old_argv=sys.argv[:]
        try:
            sys.argv=[str(p),*map(str,arguments)]
            with log.open('x') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
                runpy.run_path(str(p),run_name='__main__')
            runner.record(stage,'complete',perf_counter()-started,dict(log=str(log)))
        finally:sys.argv=old_argv
    runner.execute=execute
    final=runner.assemble()
    reference=read(ROOT/'final_package/reference/submission_clean_refined.csv')
    got=read(final)
    assert got.example_id.equals(reference.example_id)
    diff=np.log1p(got.boardings)-np.log1p(reference.boardings)
    traced=read(runner.output/'final_model_layer_changes.csv.gz',parse_dates=['date'])
    names=['station_profiles','tabicl_fixed','dongchun_retail','lunar_eve_and_day',
           'geomdan_morning','tabicl_lead_weight','vector_xgboost']
    closed=traced.initial.to_numpy()+traced[names].sum(axis=1).to_numpy()
    model=read(runner.output/'model.csv')
    event_delta=np.log1p(got.boardings)-np.log1p(model.boardings)
    report=dict(rows=len(got),reference_sha256=sha(ROOT/'final_package/reference/submission_clean_refined.csv'),
        output_sha256=sha(final),max_abs_log1p_difference=float(abs(diff).max()),
        layer_sum_max_error=float(abs(closed-traced.baseline.to_numpy()).max()),
        closure_to_clean_max_error=float(abs(closed+event_delta-np.log1p(reference.boardings)).max()),
        stage1_model_roundtrip_error=float(abs(traced.baseline-np.log1p(model.boardings)).max()),
        protected_model_amplification_changes=int(np.count_nonzero((.5*(traced.station_profiles+traced.vector_xgboost)*traced.amp_eligible)[~traced.amp_eligible])),
        run=str(runner.output.relative_to(ROOT)))
    assert report['max_abs_log1p_difference']<1e-12
    assert report['closure_to_clean_max_error']<1e-12
    traced['after_model_delta']=event_delta.to_numpy()
    traced['closed_clean_log']=closed+event_delta.to_numpy()
    csv(traced,target/'final_model_layer_changes.csv.gz')
    dump(target/'extraction_verification.json',report)
    record_reads(target/'read_manifest.json')
    status('최종 모델층 분해 완료: '+json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
