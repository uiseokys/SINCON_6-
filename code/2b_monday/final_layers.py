"""Portable accepted final layers, in order. Requires only numpy and pandas.

CLI: python -B final_layers.py --data data --cache cache --output output.csv
The clean baseline defaults to data/../submission_clean_refined.csv.
Use --disable NAME repeatedly to remove layers. No network or model training.
Cache labels are checked against official train.csv before fitting station errors.
Friday arithmetic follows the accepted sincon_clean_friday_calendar.py rule.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ['date','station_id','hour']
OOF_KEY = ['label','origin',*KEY]
CUTOFF = pd.Timestamp('2026-06-30')
CLEAN_SHA = '519c88bd46e991677316175a2d24f802db6248ebb7d3971ecbe75477532f2199'
TRAIN_SHA = '1b3b21e8f1fe22a44c0ff4adba315784a5c21caea25098a15869c4ed370822bc'
TEST_SHA = 'd257fed268db7919fd8fbc45b6c78dfbdb09b6bb40d4ae1d3f2a666c6e69fefa'
LAYER_NAMES = ('timecapsule','plave_existing','station025','friday','monday','plave22','lunar3122')
CONCERTS = {
    'carat_land':['2025-03-20','2025-03-21'], 'ye':['2025-07-26'],
    'muse':['2025-09-27'], 'jin':['2025-10-31','2025-11-01'],
    'park_hyo_shin':['2026-04-04','2026-04-05','2026-04-11']}
LUNAR = pd.to_datetime(['2025-01-29','2025-10-06','2026-02-17'])
LUNAR_TARGET = pd.Timestamp('2026-09-25')
FRIDAYS = pd.to_datetime(['2025-06-06','2025-08-15','2026-05-01'])
FRIDAY_TARGETS = pd.to_datetime(['2026-07-17','2026-10-09'])
EXCLUDED = (3107,3108,3109,3110,3111,3127,3128,3139,3207,3209,3210,3226,3754,3756,3757)
FLAGS = ['is_holiday','is_bridge_day','is_quasi_holiday','is_pre_long_offday','is_post_long_offday','is_long_weekend']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_csv(path, **kwargs):
    return pd.read_csv(path, float_precision='round_trip', **kwargs)

def donor_total(train, origin):
    series=[]
    for dates in CONCERTS.values():
        if pd.Timestamp(max(dates)) <= origin:
            x=train.loc[train.station_id.eq(3127)&train.date.isin(pd.to_datetime(dates))&train.hour.isin([21,22])]
            if len(x)!=2*len(dates):raise ValueError('Missing completed concert observations')
            series.append(float(x.groupby('date').boardings.sum().median()))
    if not series:raise ValueError('No completed concert donor')
    return float(np.median(series))

def concert_delta(frame, train, origin, target_days, column):
    delta=pd.Series(0.,index=frame.index)
    m=frame.station_id.eq(3127)&frame.hour.isin([21,22])&frame.date.isin(pd.to_datetime(target_days))
    if m.any():
        x=frame.loc[m]
        totals=np.expm1(x[column]).groupby(x.date).transform('sum')
        delta.loc[m]=(.35*(np.log1p(donor_total(train,origin))-np.log1p(totals))).clip(0,.50)
    return delta.to_numpy()

def station_errors(oof, origin):
    h=oof.loc[oof.origin.lt(origin)&oof.date.le(origin)&oof.date.gt(origin-pd.Timedelta(days=60))]
    stats=h.assign(error=h.adjusted-h.y_log).groupby('station_id').error.agg(['mean','size'])
    b=stats.loc[stats['size'].ge(360),'mean']
    return b-b.mean(), h

def friday_delta(frame, train, calendar, protected):
    regular=calendar.index[~calendar[FLAGS].ne(0).any(axis=1)]
    def panel(frame,col):
        return frame.loc[~frame.station_id.isin(EXCLUDED)].pivot(index='date',columns=['station_id','hour'],values=col)
    def contrast(p,day):
        refs=p.index.intersection(regular)
        refs=refs[(refs.dayofweek==4)&(abs(refs-day)<=pd.Timedelta(days=42))]
        if len(refs)<3:raise ValueError('Three ordinary Friday controls required')
        return p.loc[day]-p.loc[refs].median(),refs
    hist=panel(train.assign(y_log=np.log1p(train.boardings)),'y_log')
    future=panel(frame,'prediction')
    donors=[]; evidence={'donors':{},'targets':{}}
    for day in FRIDAYS[FRIDAYS+pd.Timedelta(days=2)<=CUTOFF]:
        d,refs=contrast(hist.loc[hist.index<=CUTOFF],day)
        donors.append(d);evidence['donors'][str(day.date())]=refs.strftime('%Y-%m-%d').tolist()
    result=pd.Series(0.,index=frame.index)
    for day in FRIDAY_TARGETS:
        old,refs=contrast(future,day)
        diff=pd.DataFrame(donors).reindex(columns=old.index).median()-old
        selected=frame.date.eq(day)&~frame.station_id.isin(EXCLUDED)&~pd.MultiIndex.from_frame(frame[KEY[:2]]).isin(protected)
        keys=pd.MultiIndex.from_frame(frame.loc[selected,['station_id','hour']])
        raw=diff.reindex(keys).clip(-.25,.25)
        centered=raw-raw.groupby(level='hour').transform('mean')
        limits=centered.abs().groupby(level='hour').transform('max').clip(lower=.25)
        effect=.20*centered*(.25/limits)
        if effect.abs().max()>.05+1e-12 or effect.groupby(level='hour').sum().abs().max()>1e-12:
            raise ValueError('Friday cap or centering violated')
        result.loc[selected]=effect.to_numpy()
        evidence['targets'][str(day.date())]=dict(controls=refs.strftime('%Y-%m-%d').tolist(),changed_rows=int(effect.ne(0).sum()))
    return result.to_numpy(),evidence

MONDAYS = pd.to_datetime(['2025-03-03','2025-05-05','2026-03-02','2026-05-25'])
MONDAY_TARGETS = pd.to_datetime(['2026-08-17','2026-10-05'])

def monday_delta(frame, train, calendar, protected, cutoff, targets, excluded, flags):
    regular=calendar.index[~calendar[flags].ne(0).any(axis=1)]
    def panel(frame,col):
        return frame.loc[~frame.station_id.isin(excluded)].pivot(index='date',columns=['station_id','hour'],values=col)
    def contrast(p,day):
        refs=p.index.intersection(regular)
        refs=refs[(refs.dayofweek==0)&(abs(refs-day)<=pd.Timedelta(days=42))]
        if len(refs)<3:return None,refs
        return p.loc[day]-p.loc[refs].median(),refs
    # Clip observations before any donor statistic is evaluated.
    past=train.loc[train.date.le(cutoff)]
    hist=panel(past.assign(y_log=np.log1p(past.boardings)),'y_log')
    future=panel(frame,'prediction')
    donors=[];evidence={'cutoff':str(cutoff.date()),'donors':{},'targets':{},'skipped':{}}
    for day in MONDAYS[MONDAYS<=cutoff]:
        d,refs=contrast(hist,day)
        if d is None:
            evidence['skipped'][str(day.date())]='fewer than three observed ordinary Monday controls'
            continue
        donors.append(d);evidence['donors'][str(day.date())]=refs.strftime('%Y-%m-%d').tolist()
    result=pd.Series(0.,index=frame.index)
    if not donors:
        evidence['inactive_reason']='no eligible pre-cutoff donor'
        return result.to_numpy(),evidence
    for day in targets:
        if day not in future.index:continue
        old,refs=contrast(future,day)
        if old is None:
            evidence['skipped'][str(day.date())]='fewer than three forecast ordinary Monday controls'
            continue
        diff=pd.DataFrame(donors).reindex(columns=old.index).median()-old
        selected=frame.date.eq(day)&~frame.station_id.isin(excluded)&~pd.MultiIndex.from_frame(frame[['date','station_id']]).isin(protected)
        keys=pd.MultiIndex.from_frame(frame.loc[selected,['station_id','hour']])
        raw=diff.reindex(keys).clip(-.25,.25)
        centered=raw-raw.groupby(level='hour').transform('mean')
        limits=centered.abs().groupby(level='hour').transform('max').clip(lower=.25)
        effect=.20*centered*(.25/limits)
        if not np.isfinite(effect).all() or effect.abs().max()>.05+1e-12 or effect.groupby(level='hour').sum().abs().max()>1e-12:
            raise ValueError('Monday cap or centering violated')
        result.loc[selected]=effect.to_numpy()
        evidence['targets'][str(day.date())]=dict(controls=refs.strftime('%Y-%m-%d').tolist(),changed_rows=int(effect.ne(0).sum()))
    return result.to_numpy(),evidence


def lunar_delta(frame, train, calendar):
    # No observed control is allowed to cross the fitting cutoff.
    hist=train.loc[train.station_id.eq(3122)&train.date.le(CUTOFF)].copy()
    hist['y_log']=np.log1p(hist.boardings)
    Y=hist.pivot(index='hour',columns='date',values='y_log')
    P=frame.loc[frame.station_id.eq(3122)].pivot(index='hour',columns='date',values='prediction')
    sundays=calendar.index[(calendar.index.dayofweek==6)&calendar.is_holiday.eq(0)]
    for episode in LUNAR.append(pd.DatetimeIndex([LUNAR_TARGET])):
        sundays=sundays[abs(sundays-episode)>pd.Timedelta(days=5)]
    def refs(day, columns):
        return sundays.intersection(columns)[abs(sundays.intersection(columns)-day)<=pd.Timedelta(days=35)]
    delta=pd.Series(0.,index=frame.index);evidence=[]
    for rel in [-1,0]:
        samples=[];sources=[]
        for episode in LUNAR:
            day=episode+pd.Timedelta(days=rel)
            control=refs(day,Y.columns)
            if len(control)<3:raise ValueError('Three ordinary Sundays required')
            samples.append(Y[day]-Y[control].median(axis=1))
            sources.append(dict(day=str(day.date()),controls=control.strftime('%Y-%m-%d').tolist()))
        day=LUNAR_TARGET+pd.Timedelta(days=rel)
        control=refs(day,P.columns)
        if len(control)<3:raise ValueError('Three forecast Sunday controls required')
        target=P[control].median(axis=1)+pd.concat(samples,axis=1).mean(axis=1)
        m=frame.station_id.eq(3122)&frame.date.eq(day)
        if m.sum()!=18:raise ValueError('Lunar target requires eighteen official hours')
        delta.loc[m]=frame.loc[m,'hour'].map(target)-frame.loc[m,'prediction']
        evidence.append(dict(day=str(day.date()),sources=sources,forecast_controls=control.strftime('%Y-%m-%d').tolist()))
    return delta.to_numpy(),evidence

def build_submission(data_dir, cache_dir, baseline=None, enabled=None):
    """Return (submission, row-level layer deltas, provenance report). No writes."""
    data_dir,cache_dir=Path(data_dir),Path(cache_dir)
    baseline=Path(baseline) if baseline is not None else data_dir.parent/'submission_clean_refined.csv'
    enabled=set(LAYER_NAMES if enabled is None else enabled)
    if enabled-set(LAYER_NAMES):raise ValueError('Unknown layer')
    for path,expected in [(baseline,CLEAN_SHA),(data_dir/'train.csv',TRAIN_SHA),(data_dir/'test.csv',TEST_SHA)]:
        if sha(path)!=expected:raise ValueError('Input SHA-256 mismatch: '+str(path))
    manifest=json.loads((cache_dir/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if Path(name).name!=name or sha(cache_dir/name)!=expected:raise ValueError('Cache hash mismatch: '+name)
    train=read_csv(data_dir/'train.csv',parse_dates=['date'])
    grid=read_csv(data_dir/'test.csv',parse_dates=['date'])
    clean=read_csv(baseline)
    if len(train)!=688176 or train.date.max()!=CUTOFF or train.duplicated(KEY).any():raise ValueError('Invalid official train')
    if (len(grid)!=157194 or not grid.example_id.is_unique or grid.duplicated(KEY).any()
            or list(clean)!=['example_id','boardings'] or not grid.example_id.equals(clean.example_id)
            or not np.isfinite(clean.boardings).all() or clean.boardings.lt(0).any()):raise ValueError('Invalid submission grid')
    frame=grid.copy();frame['prediction']=np.log1p(clean.boardings)
    layers=grid[['example_id',*KEY]].copy()
    report={'enabled':[x for x in LAYER_NAMES if x in enabled],'clean_sha256':CLEAN_SHA,'cache':manifest,'layers':{}}
    def apply(name,delta,evidence=None):
        delta=np.asarray(delta,float)
        before=frame.prediction.to_numpy(copy=True)
        frame['prediction']=np.maximum(before+delta,0.)
        actual=frame.prediction.to_numpy()-before
        layers[name]=actual
        report['layers'][name]={'changed_rows':int(np.count_nonzero(abs(actual)>1e-12)),
                               'max_abs_log_delta':float(abs(actual).max()),'evidence':evidence}
    rules=json.loads((cache_dir/'rules.json').read_text())
    if 'timecapsule' in enabled:
        delta=np.zeros(len(frame))
        for row in rules['timecapsule']:
            m=frame.station_id.eq(3139)&frame.date.eq('2026-07-11')&frame.hour.eq(row['hour'])
            if m.sum()!=1:raise ValueError('Missing timecapsule target')
            floor=row['ordinary_target']+.75*(row['historical_actual']-row['historical_ordinary'])
            delta[m]=np.maximum(frame.loc[m,'prediction'],np.log1p(floor))-frame.loc[m,'prediction']
        apply('timecapsule',delta,rules['timecapsule'])
    if 'plave_existing' in enabled:
        apply('plave_existing',concert_delta(frame,train,CUTOFF,['2026-09-12','2026-09-13'],'prediction'),
              {'fraction':.35,'cap':.50,'donor_21_22_total':donor_total(train,CUTOFF)})
    if 'station025' in enabled:
        oof=read_csv(cache_dir/'final_stage_oof.csv.gz',parse_dates=['date','origin'])
        if oof.origin.nunique()!=9 or oof.duplicated(OOF_KEY).any() or not oof.date.gt(oof.origin).all() or oof.date.max()>CUTOFF:
            raise ValueError('Invalid chronological OOF')
        truth=np.log1p(train.set_index(KEY).boardings).reindex(pd.MultiIndex.from_frame(oof[KEY])).to_numpy()
        if np.max(abs(truth-oof.y_log))>1e-10:raise ValueError('Cached OOF labels differ from train')
        oof['y_log']=truth;oof['adjusted']=oof.baseline
        if 'timecapsule' in enabled:
            patch=read_csv(cache_dir/'timecapsule_oof.csv',parse_dates=['date','origin']).set_index(OOF_KEY).log_delta
            if not patch.index.is_unique:raise ValueError('Duplicate event patch')
            index=pd.MultiIndex.from_frame(oof[OOF_KEY])
            if not patch.index.isin(index).all():raise ValueError('Missing patch keys')
            oof['adjusted']+=patch.reindex(index,fill_value=0).to_numpy()
        if 'plave_existing' in enabled:
            dates=[d for days in CONCERTS.values() for d in days]
            for origin,p in oof.groupby('origin'):
                oof.loc[p.index,'adjusted']+=concert_delta(p,train,origin,dates,'baseline')
        b,h=station_errors(oof,CUTOFF)
        apply('station025',-.25*frame.station_id.map(b).fillna(0).to_numpy(),
              {'window_days':60,'weight':.25,'fit_rows':len(h),'fit_first':str(h.date.min().date()),
               'fit_last':str(h.date.max().date()),'latest_origin':str(h.origin.max().date()),
               'centered_errors':{str(k):float(v) for k,v in b.items()}})
    if 'friday' in enabled:
        calendar=read_csv(cache_dir/'calendar.csv',parse_dates=['date']).set_index('date')
        dates=FRIDAYS.append(FRIDAY_TARGETS)
        episodes=calendar.loc[dates]
        if not (episodes.is_holiday.eq(1)&episodes.offday_block_len.eq(3)&episodes.offday_block_pos.eq(1)&episodes.is_lunar_holiday.eq(0)).all():
            raise ValueError('Expected non-lunar Friday three-day weekends')
        known=read_csv(cache_dir/'protected_event_keys.csv',parse_dates=['date'])
        if len(known.merge(grid,on=['example_id',*KEY],validate='one_to_one'))!=len(known):raise ValueError('Invalid protected keys')
        protected=pd.MultiIndex.from_frame(known[KEY[:2]].drop_duplicates())
        delta,ev=friday_delta(frame,train,calendar,protected);apply('friday',delta,ev)
    if 'monday' in enabled:
        calendar=read_csv(cache_dir/'calendar.csv',parse_dates=['date']).set_index('date')
        known=read_csv(cache_dir/'protected_event_keys.csv',parse_dates=['date'])
        if len(known.merge(grid,on=['example_id',*KEY],validate='one_to_one'))!=len(known):raise ValueError('Invalid protected keys')
        protected=pd.MultiIndex.from_frame(known[KEY[:2]].drop_duplicates())
        delta,ev=monday_delta(frame,train,calendar,protected,CUTOFF,MONDAY_TARGETS,EXCLUDED,FLAGS)
        apply('monday',delta,ev)
    if 'plave22' in enabled:
        audit=read_csv(cache_dir/'historical_concerts.csv',parse_dates=['date'])
        if audit.date.max()>CUTOFF:raise ValueError('Post-cutoff concert calibration')
        rates=audit.loc[audit.ratio_22.notna()].groupby('event').ratio_22.median().to_numpy()
        q25,q50=np.quantile(rates,[.25,.5]);rate=float(np.sqrt(q25*q50))
        delta=np.zeros(len(frame))
        for date,attendance,ordinary in [('2026-09-12',23120,21.5),('2026-09-13',19494,15.)]:
            m=frame.station_id.eq(3127)&frame.date.eq(date)&frame.hour.eq(22)
            if m.sum()!=1:raise ValueError('Missing PLAVE 22h target')
            delta[m]=np.log1p(ordinary+rate*attendance)-frame.loc[m,'prediction']
        apply('plave22',delta,{'rate':rate,'series_rates':rates.tolist(),'quantiles':[float(q25),float(q50)]})
    if 'lunar3122' in enabled:
        calendar=read_csv(cache_dir/'lunar_calendar.csv',parse_dates=['date']).set_index('date')
        delta,ev=lunar_delta(frame,train,calendar);apply('lunar3122',delta,ev)
    for name in LAYER_NAMES:
        if name not in layers:layers[name]=0.
    values=clean.boardings.to_numpy(copy=True)
    changed=frame.prediction.to_numpy()!=np.log1p(values)
    values[changed]=np.expm1(frame.loc[changed,'prediction'])
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid output values')
    output=clean.assign(boardings=values)
    report.update(rows=len(output),unique_ids=int(output.example_id.nunique()),finite=True,nonnegative=True,
                  changed_from_clean=int(np.count_nonzero(values!=clean.boardings.to_numpy())))
    return output,layers,report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--disable',action='append',choices=LAYER_NAMES,default=[])
    p.add_argument('--compare',type=Path)
    args=p.parse_args()
    outputs=[args.output,args.output.with_suffix('.report.json'),args.output.with_suffix('.layers.csv.gz')]
    if any(x.exists() for x in outputs):raise FileExistsError('Outputs already exist; choose a new filename')
    out,layers,report=build_submission(args.data,args.cache,args.input,set(LAYER_NAMES)-set(args.disable))
    if args.compare:
        reference=read_csv(args.compare)
        if not reference.example_id.equals(out.example_id):raise ValueError('Comparison ID mismatch')
        report['comparison']={'reference_sha256':sha(args.compare),
            'max_abs_log1p_difference':float(abs(np.log1p(out.boardings)-np.log1p(reference.boardings)).max()),
            'rows_log_difference_gt_1e_10':int((abs(np.log1p(out.boardings)-np.log1p(reference.boardings))>1e-10).sum())}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(outputs[0],index=False)
    actual=read_csv(outputs[0])
    if not actual.example_id.equals(out.example_id) or not np.array_equal(actual.boardings,out.boardings):raise ValueError('CSV round trip failed')
    report['sha256']=sha(outputs[0])
    layers.to_csv(outputs[2],index=False,compression={'method':'gzip','mtime':0})
    outputs[1].write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['cache','layers']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
