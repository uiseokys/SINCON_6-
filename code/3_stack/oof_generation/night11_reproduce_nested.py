"""Frozen-prediction forward adoption audit. All writes stay under night11/.

No training, network, score-based tuning, or target-period subway labels.
Preserves the original gates, including their required comparison groups.
At cutoff T only outcomes dated <=T from forecasts issued <T enter gates.
"""
import os
import sys
from pathlib import Path

HOME_OUT = Path(__file__).resolve().parent
ROOT = HOME_OUT.parent
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = str(HOME_OUT / 'mpl')
os.environ['TMPDIR'] = str(HOME_OUT / 'tmp')
sys.dont_write_bytecode = True

def fence(event, args):
    def check(value):
        if isinstance(value, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(value)).resolve()
            if not path.is_relative_to(HOME_OUT) and str(path) != '/dev/null':
                raise PermissionError('night11 write fence: ' + str(path))
    if event == 'open':
        path, mode, flags = args
        if ((isinstance(mode, str) and any(c in mode for c in 'wax+')) or
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            check(path)
    elif event in {'os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.utime'}:
        check(args[0])
    elif event in {'os.rename', 'os.link', 'os.symlink'}:
        check(args[0]); check(args[1])
    elif event in {'socket.connect', 'socket.bind'}:
        raise PermissionError('night11 is offline')

sys.addaudithook(fence)
import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

WORK = ROOT / 'final_package/payload/work'
TEAM = Path('/Users/jeong-uiseok/sincon-2026/sincon-baseline')
RUN = Path(json.loads((ROOT/'final_audit/stage1_run_location.json').read_text())['output'])
CACHE = RUN/'cache'
sys.path[:0] = [str(WORK), str(ROOT/'final_package/payload/team')]
import sincon_clean_model_layers as m

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

final = module('night11_final', ROOT/'night7/final_layers.py')
lunar = module('night11_lunar', ROOT/'night7/audit_lunar.py')
friday = module('night11_friday', ROOT/'sincon_clean_friday_calendar.py')
KEY, OK = m.KEY, m.OOF_KEY
END = pd.Timestamp('2026-06-30')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def status(text):
    now = datetime.now(timezone(timedelta(hours=9))).isoformat()
    with (HOME_OUT/'STATUS.md').open('a') as f:
        f.write('\n## '+now+'\n\n'+text+'\n')
    print(now, text, flush=True)

def csv(d, p):
    d.to_csv(p, index=False, float_format='%.17g',
             compression={'method':'gzip','mtime':0} if str(p).endswith('.gz') else None)

def prior(rows, cutoff):
    return rows.loc[rows.origin.lt(cutoff) & rows.date.le(cutoff)]

def group_scores(rows, columns):
    if rows.empty:
        return {}
    return {name:dict(rows=len(p), first_date=str(p.date.min().date()),
                     last_date=str(p.date.max().date()),
                     **{c:m.rmsle(p.y_log,p[c]) for c in columns})
            for name,p in m.groups_for_scores(rows).items() if len(p)}

def decide(kind, rows, cutoff=None):
    p = rows if cutoff is None else prior(rows, cutoff)
    cols = ['baseline','fixed','guarded'] if kind=='tabicl' else ['baseline','candidate']
    scores = group_scores(p,cols)
    required = {'tabicl': ['disjoint_A_C','summer_established_68','disjoint_warm'],
                'retail': ['disjoint_A_C','disjoint_warm'],
                'vector': [n for n in ['disjoint_A_C','disjoint_warm','summer_established_68'] if n in scores]}[kind]
    enough = bool(required) and all(n in scores for n in required)
    if kind=='tabicl':
        fixed = enough and all(scores[n]['fixed']<scores[n]['baseline'] for n in required)
        guarded = fixed and all(scores[n]['guarded']<scores[n]['fixed'] for n in required)
        chosen = 'guarded' if guarded else 'fixed' if fixed else 'baseline'
    else:
        chosen = 'candidate' if enough and all(scores[n]['candidate']<scores[n]['baseline'] for n in required) else 'baseline'
    return chosen, dict(choice=chosen, sufficient_groups=enough, required=required, comparisons=scores,
                        prior_rows=len(p), latest_label=str(p.date.max().date()) if len(p) else None,
                        latest_origin=str(p.origin.max().date()) if len(p) else None)

def choose(rows, kind, nested, gates, trials=None):
    trials = rows if trials is None else trials
    out = rows.copy()
    for origin,p in rows.groupby('origin',sort=True):
        choice,ev = decide(kind,trials,origin if nested else None)
        gates.append(dict(path='nested' if nested else 'old',layer=kind,origin=str(origin.date()),**ev))
        out.loc[p.index,'baseline'] = p[choice].to_numpy()
    return out

def build_oof(station, tab_trials, history, protected, vector_sources, nested):
    gates = []
    trial = tab_trials.set_index(OK)
    ix = pd.MultiIndex.from_frame(station[OK])
    base = station.copy()
    for col in ['fixed','guarded']:
        base[col] = trial[col].reindex(ix).fillna(pd.Series(base.baseline.to_numpy(),index=ix)).to_numpy()
    selected = choose(base,'tabicl',nested,gates)
    # Production order: fixed first; any guarded increment is delayed until after lunar/morning.
    extra = selected.baseline.to_numpy()-selected.fixed.to_numpy()
    by_origin = {pd.Timestamp(g['origin']):g['choice']!='baseline' for g in gates}
    active = selected.origin.map(by_origin).to_numpy()
    extra = np.where(active, extra, 0.)
    selected['baseline'] = np.where(active,selected.fixed,selected.baseline)
    selected=selected.drop(columns=['fixed','guarded'])
    retail = selected.assign(error=selected.y_log-selected.baseline, observed=selected.y_log,
                             base=selected.baseline,eligible=m.ordinary(selected,protected))
    known = m.retail_select(retail)
    def retail_prediction(part, origin):
        h=prior(known,origin).sort_values(['reliability','origin']).drop_duplicates(KEY,keep='last')
        change,_=m.retail_case(part,m.retail_contrasts(h))
        return (part.baseline+change).clip(lower=0.)
    revised,_=m.oof_layer(retail,retail_prediction)
    retail['candidate'] = revised.baseline.to_numpy()
    oof=choose(retail,'retail',nested,gates)
    phases=m.calendars(WORK)
    def lunar_pred(part,origin):
        change,_=m.lunar_proposal(history.loc[history.date.le(origin)],part[[*KEY,'baseline']],phases,origin,protected)
        return (part.baseline+m.verified_lunar_phases(part,change,phases)).clip(lower=0.)
    def morning_pred(part,origin):
        change,_=m.morning_forecast(history.loc[history.date.le(origin)],part[[*KEY,'baseline']],origin,protected)
        return (part.baseline+change['relative_peak_share']).clip(lower=0.)
    oof,_=m.oof_layer(oof,lunar_pred)
    oof,_=m.oof_layer(oof,morning_pred)
    # oof_layer preserves the label-group row order used in this script.
    assert oof[OK].equals(selected[OK])
    oof['baseline']+=extra
    def vector_pred(part,_):
        eligible=m.ordinary(part,protected)&~part.station_id.isin(m.STABLE_EXCLUSIONS)
        return m.vector_offsets(part.assign(eligible=eligible),vector_sources[part.label.iloc[0]])['20%']
    candidate,_=m.oof_layer(oof,vector_pred)
    oof['candidate']=candidate.baseline.to_numpy()
    oof=choose(oof,'vector',nested,gates)
    return oof[OK+['y_log','baseline']],gates

def validate_downstream(oof, history, output, path):
    oof=oof.copy();oof['adjusted']=oof.baseline
    patch=final.read_csv(ROOT/'night7/cache/timecapsule_oof.csv',parse_dates=['date','origin']).set_index(OK).log_delta
    oof['adjusted']+=patch.reindex(pd.MultiIndex.from_frame(oof[OK]),fill_value=0.).to_numpy()
    for origin,p in oof.groupby('origin',sort=True):
        dates=[d for days in final.CONCERTS.values() for d in days]
        oof.loc[p.index,'adjusted']+=final.concert_delta(p,history,origin,dates,'baseline')
    oof['station025']=oof.adjusted
    stats=[]
    for origin,p in oof.groupby('origin',sort=True):
        b,h=final.station_errors(oof,origin)
        oof.loc[p.index,'station025']=np.maximum(p.adjusted-.25*p.station_id.map(b).fillna(0.),0.)
        a=m.rmsle(p.y_log,p.adjusted); c=m.rmsle(p.y_log,oof.loc[p.index,'station025'])
        stats.append(dict(path=path,label=p.label.iloc[0],origin=str(origin.date()),rows=len(p),prior_rows=len(h),
                          stations=len(b),reference=a,candidate=c,delta=c-a))
    b,h=final.station_errors(oof,END)
    csv(pd.DataFrame(stats),output/f'station_validation_{path}.csv')
    cal=final.read_csv(ROOT/'night7/cache/calendar.csv',parse_dates=['date']).set_index('date')
    regular=cal.index[~cal[final.FLAGS].ne(0).any(axis=1)]
    hist=friday.matrix(history.assign(observed=np.log1p(history.boardings)),'observed')
    empty=pd.MultiIndex.from_arrays([[],[]],names=KEY[:2])
    oof['friday']=oof.station025
    fs=[];fe=[]
    for origin,p in oof.groupby('origin',sort=True):
        days=friday.HISTORICAL[friday.HISTORICAL.isin(p.date)]
        if not len(days): continue
        change,_,ev=friday.forecast(p[KEY].assign(baseline=p.station025),hist,origin,days,regular,empty)
        oof.loc[p.index,'friday']=p.station025.to_numpy()+change.to_numpy()
        a=m.rmsle(p.y_log,p.station025);c=m.rmsle(p.y_log,oof.loc[p.index,'friday'])
        fs.append(dict(path=path,label=p.label.iloc[0],origin=str(origin.date()),rows=len(p),
                       changed_rows=int(change.ne(0).sum()),reference=a,candidate=c,delta=c-a))
        for day in days:
            mask=p.date.eq(day)&change.ne(0)
            a=m.rmsle(p.loc[mask,'y_log'],p.loc[mask,'station025'])
            c=m.rmsle(p.loc[mask,'y_log'],oof.loc[p.index[mask],'friday'])
            fe.append(dict(path=path,origin=str(origin.date()),date=str(day.date()),rows=int(mask.sum()),
                           reference=a,candidate=c,delta=c-a))
    csv(pd.DataFrame(fs),output/f'friday_validation_{path}.csv')
    csv(pd.DataFrame(fe),output/f'friday_exposed_{path}.csv')
    lc=final.read_csv(ROOT/'night7/cache/lunar_calendar.csv',parse_dates=['date']).set_index('date')
    Y=history.loc[history.station_id.eq(3122)].assign(y_log=lambda x:np.log1p(x.boardings)).pivot(index='hour',columns='date',values='y_log')
    oof['lunar3122']=oof.friday
    ls=[];ex=[]
    for origin,p in oof.groupby('origin',sort=True):
        q=p.loc[p.station_id.eq(3122)];P=q.pivot(index='hour',columns='date',values='friday');changed=0
        for episode in lunar.EPISODES:
            for rel in [-1,0]:
                day=episode+pd.Timedelta(days=rel)
                if day not in P.columns: continue
                pred,ev,refs=lunar.transfer(Y,P,lc,day,rel,origin)
                if pred is None: continue
                mask=q.date.eq(day);ix=q.index[mask]
                vals=q.loc[mask,'hour'].map(pred).to_numpy()
                oof.loc[ix,'lunar3122']=vals;changed+=len(ix)
                a=m.rmsle(q.loc[mask,'y_log'],q.loc[mask,'friday']);c=m.rmsle(q.loc[mask,'y_log'],vals)
                ex.append(dict(path=path,origin=str(origin.date()),date=str(day.date()),rel=rel,rows=len(ix),
                               reference=a,candidate=c,delta=c-a))
        a=m.rmsle(p.y_log,p.friday);c=m.rmsle(p.y_log,oof.loc[p.index,'lunar3122'])
        ls.append(dict(path=path,label=p.label.iloc[0],origin=str(origin.date()),rows=len(p),changed_rows=changed,
                       reference=a,candidate=c,delta=c-a))
    csv(pd.DataFrame(ls),output/f'lunar_validation_{path}.csv')
    csv(pd.DataFrame(ex),output/f'lunar_exposed_{path}.csv')
    csv(oof,output/f'downstream_oof_{path}.csv.gz')
    return b,h

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=HOME_OUT/'measurements')
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(HOME_OUT): raise ValueError('Output must be under night11')
    out.mkdir(exist_ok=False,parents=True)
    paths=[TEAM/'data/train.csv',TEAM/'data/test.csv',TEAM/'event_hedges.py',
           TEAM/'submission_night_v4_main.csv',TEAM/'submission_night_v4_events_hedged.csv',
           CACHE/'sincon_clean_event_oof.csv.gz',CACHE/'sincon_clean_tabicl_oof.csv',
           ROOT/'night/cache/vector_all_extra_origins.csv.gz',WORK/'sincon_vector_boost_validation.csv.gz',
           ROOT/'night7/cache/final_stage_oof.csv.gz',ROOT/'night7/final_layers.py',
           ROOT/'night7/cache/timecapsule_oof.csv',WORK/'sincon_clean_model_layers.py',
           WORK/'sincon_local_repro.zip',WORK/'sincon_festival_station_days.csv',RUN/'protected.csv']
    hashes={str(p):sha(p) for p in paths};dump(out/'inputs.json',hashes)
    status('1단계 시작. 보관 예측·공식 정답 해시 고정, CPU 계산 라이브러리별 1스레드, 네트워크·night11 밖 쓰기 차단.')
    train=final.read_csv(TEAM/'data/train.csv',parse_dates=['date'])
    assert len(train)==688176 and train.date.max()==END and not train.duplicated(KEY).any()
    history=train[KEY].assign(observed=np.log1p(train.boardings))
    rows=m.read_oof(CACHE/'sincon_clean_event_oof.csv.gz',history,'baseline')
    explicit=final.read_csv(RUN/'protected.csv',parse_dates=['date'])
    days=pd.MultiIndex.from_frame(explicit[m.DAY].drop_duplicates())
    holidays,known=m.protected_days(WORK);protected=(holidays,known.union(days))
    games,_=m.home_dates(WORK)
    def station_pred(p,origin):
        d,_=m.station_adjustments(history.loc[history.date.le(origin)],p[KEY].assign(combined=p.baseline),origin,protected,games)
        return (p.baseline+d['joint']).clip(lower=0.)
    station,_=m.oof_layer(rows,station_pred)
    alts=m.read_daily(CACHE/'sincon_clean_tabicl_oof.csv',['label','origin',*m.DAY])
    w,ev,trials=m.tabicl_oof(station,alts,protected)
    dump(out/'original_tabicl.json',dict(weight=w,**ev))
    sources=m.vector_archive(WORK)
    raw=final.read_csv(ROOT/'night/cache/vector_all_extra_origins.csv.gz',parse_dates=['date'])
    sources.update({label:p.set_index(KEY).raw_delta for label,p in raw.groupby('label')})
    old,old_gates=build_oof(station,trials,history,protected,sources,False)
    archived=final.read_csv(ROOT/'night7/cache/final_stage_oof.csv.gz',parse_dates=['date','origin']).set_index(OK)
    delta=old.baseline.to_numpy()-archived.baseline.reindex(pd.MultiIndex.from_frame(old[OK])).to_numpy()
    check=dict(rows=len(old),max_abs_log_difference=float(abs(delta).max()),changed_gt_1e_12=int((abs(delta)>1e-12).sum()))
    dump(out/'old_oof_reproduction.json',check)
    assert check['max_abs_log_difference']<1e-12,check
    status('기존 OOF 재현 완료: '+json.dumps(check,ensure_ascii=False))
    nested,nested_gates=build_oof(station,trials,history,protected,sources,True)
    dump(out/'adoption_gates.json',old_gates+nested_gates)
    csv(pd.DataFrame([{k:v for k,v in g.items() if k!='comparisons'} for g in old_gates+nested_gates]),out/'adoption_gates.csv')
    results=[]
    for path,d in [('old',old),('nested',nested)]:
        csv(d,out/f'model_oof_{path}.csv.gz')
        for name,p in m.groups_for_scores(d).items():
            results.append(dict(path=path,label=name,rows=len(p),rmsle=m.rmsle(p.y_log,p.baseline)))
    csv(pd.DataFrame(results),out/'model_scores.csv')
    # Re-run every nested adoption decision after poisoning all labels unavailable at T.
    # Baseline candidate arrays are past-only fits. Rechecking the full chain is separate below.
    poison_checks=[]
    for origin in sorted(station.origin.unique()):
        origin=pd.Timestamp(origin)
        altered=trials.copy();altered.loc[altered.date.gt(origin),'y_log']+=1000.
        a=decide('tabicl',trials,origin);b=decide('tabicl',altered,origin)
        assert a==b
        poison_checks.append(dict(origin=str(origin.date()),tabicl_gate_unchanged=True))
    dump(out/'gate_counterfactual.json',poison_checks)
    old_b,h=validate_downstream(old,train,out,'old')
    new_b,h2=validate_downstream(nested,train,out,'nested')
    fits=pd.DataFrame(dict(station_id=old_b.index,old_centered_error=old_b.values,
                           nested_centered_error=new_b.reindex(old_b.index).values))
    fits['old_log_correction']=-.25*fits.old_centered_error
    fits['nested_log_correction']=-.25*fits.nested_centered_error
    fits['difference']=fits.nested_log_correction-fits.old_log_correction
    fits['relative_1plus_count_change']=np.expm1(fits.difference)
    assert len(fits)==71 and fits.notna().all().all()
    csv(fits,out/'station_final_71.csv')
    corr=float(np.corrcoef(fits.old_log_correction,fits.nested_log_correction)[0,1])
    summary=dict(fit_rows=len(h),unique_keys=len(h.drop_duplicates(KEY)),first=str(h.date.min().date()),
                 last=str(h.date.max().date()),latest_origin=str(h.origin.max().date()),
                 max_abs_log_correction_difference=float(abs(fits.difference).max()),
                 mean_abs_log_correction_difference=float(abs(fits.difference).mean()),correlation=corr)
    dump(out/'station_sensitivity.json',summary)
    # Verify archived station fits and the complete final builder against the current main.
    rebuilt,layers,report=final.build_submission(TEAM/'data',ROOT/'night7/cache')
    main=final.read_csv(TEAM/'submission_night_v4_main.csv')
    maxdiff=float(abs(np.log1p(rebuilt.boardings)-np.log1p(main.boardings)).max())
    assert maxdiff<1e-12
    report_b=pd.Series({int(k):v for k,v in report['layers']['station025']['evidence']['centered_errors'].items()})
    assert abs(old_b-report_b.reindex(old_b.index)).max()<1e-12
    dump(out/'final_main_reproduction.json',dict(max_abs_log_difference=maxdiff,
         original_station_fit_max_difference=float(abs(old_b-report_b.reindex(old_b.index)).max())))
    # Preserve all five other layer algorithms and their order. Only replace the estimator.
    original=final.station_errors
    def nested_station_errors(oof,origin):
        if origin!=END: raise ValueError('Only final fit is substituted')
        return new_b,h2
    final.station_errors=nested_station_errors
    alternative,alternative_layers,alternative_report=final.build_submission(TEAM/'data',ROOT/'night7/cache')
    final.station_errors=original
    effect=np.log1p(alternative.boardings)-np.log1p(main.boardings)
    countdiff=alternative.boardings-main.boardings
    grid=final.read_csv(TEAM/'data/test.csv',parse_dates=['date'])
    csv(grid[['example_id',*KEY]].assign(log_difference=effect,count_difference=countdiff),
        out/'nested_station_row_differences.csv.gz')
    dump(out/'nested_station_diagnostic.report.json',alternative_report)
    summary.update(changed_rows=int((abs(effect)>1e-12).sum()),max_abs_final_log_difference=float(abs(effect).max()),
                   max_abs_final_count_difference=float(abs(countdiff).max()),
                   count_sum_difference=float(countdiff.sum()),final_correlation_with_main=float(np.corrcoef(main.boardings,alternative.boardings)[0,1]))
    dump(out/'station_sensitivity.json',summary)
    dump(out/'source_integrity_end.json',{p:dict(before=v,after=sha(p),unchanged=v==sha(p)) for p,v in hashes.items()})
    assert all(v==sha(p) for p,v in hashes.items())
    status('할 일 1의 1~3 측정 완료. 기존 OOF·최종 주후보 재현 확인. 역별 민감도: '+json.dumps(summary,ensure_ascii=False))
    print(pd.DataFrame(results).pivot(index='label',columns='path',values='rmsle').to_string(),flush=True)

if __name__=='__main__': main()
