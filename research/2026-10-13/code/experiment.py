"""Reproducible measurement sensitivity, matched deletion and time-fenced forecasts."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import datetime, json, sys
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from scipy.linalg import solve
from configure import OUT, SOURCE, SCOPES, CONTROLS, METHODS, SEED, sha

WORK = OUT/'work/retail_measurement'
WORK.mkdir(parents=True, exist_ok=True)
SOURCE_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_CODE))
from core import ols

def correlation(x, y, ranks=False):
    x, y = np.asarray(x,float), np.asarray(y,float)
    valid = np.isfinite(x)&np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x)<3 or x.std()<1e-14 or y.std()<1e-14: return np.nan
    if ranks: x, y = rankdata(x), rankdata(y)
    return float(np.corrcoef(x,y)[0,1])

def imbalance(buy, sell, weight=None):
    buy, sell = np.asarray(buy,float), np.asarray(sell,float)
    w = np.ones_like(buy) if weight is None else np.asarray(weight,float)
    total = np.sum(w*(buy+sell),axis=1)
    return np.divide(np.sum(w*(buy-sell),axis=1),total,
                     out=np.full(len(buy),np.nan),where=total>0)

def deletion_weights(d, seed, worst=False):
    d=np.asarray(d,float)
    if worst: removed = np.argmax(d,axis=1)
    else: removed = np.random.default_rng(seed).integers(d.shape[1],size=len(d))
    w=np.ones_like(d);w[np.arange(len(d)),removed]=0
    return w

def block_stats(x, block=4):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if len(x)<12: return {'n':len(x),'mean':float(np.mean(x)) if len(x) else np.nan,
                         'ci_low':np.nan,'ci_high':np.nan,'hac_t':np.nan,'p':np.nan}
    rng=np.random.default_rng(SEED+block)
    starts=rng.integers(0,len(x),size=(4000,int(np.ceil(len(x)/block))))
    indices=(starts[:,:,None]+np.arange(block))%len(x)
    means=x[indices.reshape(4000,-1)[:,:len(x)]].mean(axis=1)
    centered=x-x.mean();lr=float(centered@centered/len(x))
    for k in range(1,min(4,len(x)-1)+1):
        lr+=2*(1-k/5)*float(centered[k:]@centered[:-k]/len(x))
    se=np.sqrt(max(lr,0)/len(x))
    t=x.mean()/se if se>0 else np.nan
    ci=np.quantile(means,[.025,.975])
    return {'n':len(x),'mean':float(x.mean()),'ci_low':float(ci[0]),'ci_high':float(ci[1]),
            'hac_t':float(t),'p':float(2*norm.sf(abs(t)))}

def holm(p):
    p=np.asarray(p,float);out=np.full(len(p),np.nan)
    idx=np.where(np.isfinite(p))[0];order=idx[np.argsort(p[idx])]
    out[order]=np.minimum(1,np.maximum.accumulate((len(order)-np.arange(len(order)))*p[order]))
    return out

def normalized(x, clip=False):
    x=np.asarray(x,float)
    if clip: x=np.clip(x,np.quantile(x,.01,axis=0),np.quantile(x,.99,axis=0))
    sd=x.std(axis=0);sd=np.where(sd>1e-12,sd,1)
    return (x-x.mean(axis=0))/sd

def load_and_enrich():
    manifest=json.loads((SOURCE/'linked_manifest.json').read_text())
    assert manifest['complete']
    calendar=pd.DatetimeIndex(pd.read_csv(SOURCE/'calendar.csv',parse_dates=['date']).date)
    panels=[];files={};original=[]
    for scope in SCOPES:
        path=SOURCE/'analysis'/scope/'weekly_panel.parquet'
        expected=json.loads((path.parent/'run_manifest.json').read_text())['files'][path.name]
        assert sha(path)==expected
        files[str(path)]=expected
        f=pd.read_parquet(path);f['scope']=scope
        panels.append(f)
        original.append(json.loads((path.parent/'run_manifest.json').read_text())['primary'])
    panel=pd.concat(panels,ignore_index=True)
    required=set()
    for day in panel.signal_date.unique():
        j=calendar.get_loc(day);required.update(calendar[j-4:j+1])
    cols=['date','permno','both_signed','direction_disagreements','qmp_buy_volume',
          'qmp_sell_volume','qmp_signed_trades','bjzz_common_buy_volume',
          'bjzz_common_sell_volume','bjzz_common_signed_trades','clean_volume',
          'mean_quote_age_seconds']
    needed_ids=set(panel.permno)
    byday={}
    for i,day in enumerate(sorted(required)):
        path=SOURCE/'linked'/f'{day:%Y-%m-%d}.parquet'
        expected=manifest['daily_files'][path.name]
        assert sha(path)==expected
        files[str(path)]=expected
        f=pd.read_parquet(path,columns=cols)
        assert not f.duplicated('permno').any()
        f=f[f.permno.isin(needed_ids)].set_index('permno')
        byday[day]=f
        if (i+1)%150==0: print(f'Validated daily cache {i+1}/{len(required)}',flush=True)
    (WORK/'input_hashes.json').write_text(json.dumps(files,indent=2))
    enriched=[];audit=[];random_values=[]
    for day,f in panel.groupby('signal_date',sort=True):
        f=f.sort_values('permno').copy()
        j=calendar.get_loc(day);days=calendar[j-4:j+1];ids=f.permno.to_numpy()
        chunks=[byday[d].reindex(ids) for d in days]
        def array(col):return np.column_stack([c[col].to_numpy(float) for c in chunks])
        buy,sell,count=array('qmp_buy_volume'),array('qmp_sell_volume'),array('qmp_signed_trades')
        both,dis=array('both_signed'),array('direction_disagreements')
        assert np.all(dis[np.isfinite(dis)]>=0)
        assert np.all(dis[np.isfinite(dis)&np.isfinite(both)]<=both[np.isfinite(dis)&np.isfinite(both)])
        d=np.divide(dis,both,out=np.full_like(dis,np.nan),where=both>0)
        raw=imbalance(buy,sell)
        eligible=np.isfinite(buy+sell+count).all(axis=1)&(count.sum(axis=1)>=20)
        raw=np.where(eligible,raw,np.nan)
        common=np.isfinite(raw)&f.qmp.notna().to_numpy()
        err=float(np.max(np.abs(raw[common]-f.qmp.to_numpy()[common])))
        assert err<1e-12
        braw=imbalance(array('bjzz_common_buy_volume'),array('bjzz_common_sell_volume'))
        bcommon=np.isfinite(braw)&f.bjzz_common.notna().to_numpy()
        berr=float(np.max(np.abs(braw[bcommon]-f.bjzz_common.to_numpy()[bcommon])))
        assert berr<1e-12
        seed=SEED+int(day.strftime('%Y%m%d'))
        dw=deletion_weights(np.where(np.isfinite(d),d,-1),seed,worst=True)
        rw=deletion_weights(d,seed)
        weighted=imbalance(buy,sell,1-d)
        worst=imbalance(buy,sell,dw);random=imbalance(buy,sell,rw)
        valid_d=np.isfinite(d).all(axis=1)
        for name,values,mask in [('qmp_weighted',weighted,valid_d),
                                  ('qmp_drop_worst',worst,valid_d&(np.sum(dw*count,axis=1)>=20)),
                                  ('qmp_random_drop',random,valid_d&(np.sum(rw*count,axis=1)>=20))]:
            f[name]=np.where(mask&eligible,values,np.nan)
        f['proxy_blend']=(f.qmp+f.bjzz_common)/2
        f['trade_disagreement']=np.divide(dis.sum(axis=1),both.sum(axis=1),out=np.full(len(f),np.nan),where=both.sum(axis=1)>0)
        f['mean_day_disagreement']=np.mean(d,axis=1)
        full=(buy+sell).sum(axis=1)
        f['worst_retained_volume']=np.sum(dw*(buy+sell),axis=1)/full
        f['random_retained_volume']=np.sum(rw*(buy+sell),axis=1)/full
        f['signed_trades']=count.sum(axis=1)
        f['day_disagreement_sd']=np.std(d,axis=1)
        rcols=[]
        for rep in range(20):
            w=deletion_weights(d,seed+10000*(rep+1))
            r=imbalance(buy,sell,w)
            r=np.where(valid_d&eligible&(np.sum(w*count,axis=1)>=20),r,np.nan)
            rcols.append(r)
        matrix=np.column_stack(rcols)
        # Random controls use the same stock support in every replication.
        f['random_rep_complete']=np.isfinite(matrix).all(axis=1)
        f['random_rep_key']=np.arange(len(f))
        np.save(WORK/f'random_{day:%Y%m%d}.npy',matrix)
        formed=f.dropna(subset=CONTROLS+METHODS+['industry'])
        formed=formed[~formed.delisted_at_signal]
        audit.append({'scope':f.scope.iloc[0],'signal_date':day,'members':len(f),
                      'original_formation':int((f[CONTROLS+['qmp','bjzz_common','industry']].notna().all(axis=1)&~f.delisted_at_signal).sum()),
                      'complete_variant_formation':len(formed),'complete_outcomes':int(formed.week1.notna().sum()),
                      'qmp_max_reconstruction_error':err,'bjzz_max_reconstruction_error':berr,
                      'disagreement_missing_stocks':int((~valid_d).sum()),
                      'mean_worst_retained_volume':float(formed.worst_retained_volume.mean()),
                      'mean_random_retained_volume':float(formed.random_retained_volume.mean())})
        enriched.append(f)
    panel=pd.concat(enriched,ignore_index=True)
    panel.to_parquet(WORK/'enriched_panel.parquet',index=False)
    pd.DataFrame(audit).to_csv(OUT/'sample_audit.csv',index=False)
    (WORK/'original_primary.json').write_text(json.dumps(original,indent=2))
    print('Signals reconstructed and enriched',flush=True)
    return panel,files

def empirical(panel):
    records=[];random_records=[];sensitivity=[];groups=[];original_records=[]
    for day,f in panel.groupby('signal_date',sort=True):
        scope=f.scope.iloc[0]
        orig=f.dropna(subset=CONTROLS+['qmp','bjzz_common','industry','week1'])
        orig=orig[~orig.delisted_at_signal]
        original_records.append({'scope':scope,'signal_date':day,'coefficient':ols(orig.qmp,orig.week1,orig[CONTROLS],orig.industry.astype(str))[0]})
        formed=f.dropna(subset=CONTROLS+METHODS+['industry']).copy()
        formed=formed[~formed.delisted_at_signal]
        assert len(formed)>=300
        formed['disagreement_group']=pd.qcut(formed.trade_disagreement.rank(method='first'),3,labels=['low','middle','high'])
        matched=formed.dropna(subset=['week1'])
        y=matched.week1.to_numpy(float)
        z=matched[CONTROLS].to_numpy(float);industry=matched.industry.astype(str)
        for method in METHODS:
            coef=ols(matched[method],y,z,industry)[0]
            records.append({'scope':scope,'signal_date':day,'method':method,'n':len(matched),
                            'rank_ic':correlation(matched[method],y,True),'coefficient':coef,
                            'rank_corr_with_qmp':correlation(matched[method],matched.qmp,True)})
        for group in ['low','middle','high']:
            part=matched[matched.disagreement_group==group]
            for method in ['qmp','bjzz_common','qmp_weighted','qmp_drop_worst']:
                groups.append({'scope':scope,'signal_date':day,'disagreement_group':group,'method':method,'n':len(part),
                               'mean_disagreement':float(part.trade_disagreement.mean()),
                               'rank_ic':correlation(part[method],part.week1,True)})
        rmat=np.load(WORK/f'random_{day:%Y%m%d}.npy')
        same=matched[matched.random_rep_complete]
        for rep in range(20):
            x=rmat[same.random_rep_key.to_numpy(),rep]
            random_records.append({'scope':scope,'signal_date':day,'rep':rep,'n':len(same),
                                   'rank_ic':correlation(x,same.week1,True),
                                   'drop_worst_ic':correlation(same.qmp_drop_worst,same.week1,True),
                                   'qmp_ic':correlation(same.qmp,same.week1,True)})
        for delta in [.01,.025,.05,.1,.2]:
            x=np.sort(formed.qmp.to_numpy(float));n=len(x)
            ordered=int(np.searchsorted(x,x-4*delta,side='left').sum())
            sensitivity.append({'scope':scope,'signal_date':day,'assumed_wrong_signed_volume_fraction':delta,
                                'robust_sign_fraction':float(np.mean(np.abs(x)>2*delta)),
                                'robust_pairwise_order_fraction':ordered/(n*(n-1)/2),'n':n})
    rec=pd.DataFrame(records);rec.to_csv(OUT/'weekly_metrics.csv',index=False)
    pd.DataFrame(groups).to_csv(OUT/'disagreement_group_metrics.csv',index=False)
    pd.DataFrame(random_records).to_csv(OUT/'random_deletion_replications.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'bounded_error_sensitivity.csv',index=False)
    orig=pd.DataFrame(original_records)
    expected=json.loads((WORK/'original_primary.json').read_text())
    audit=[]
    for scope,e in zip(SCOPES,expected):
        actual=orig[orig.scope==scope].coefficient.mean()
        assert abs(actual-e['mean'])<1e-10
        audit.append({'scope':scope,'original_coefficient_mean':e['mean'],'reproduced':actual,'absolute_error':abs(actual-e['mean'])})
    pd.DataFrame(audit).to_csv(OUT/'original_result_reproduction.csv',index=False)
    rows=[]
    for (scope,method),f in rec.groupby(['scope','method']):
        st=block_stats(f.rank_ic)
        rows.append({'scope':scope,'method':method,**st,'ic_ir_unannualized':f.rank_ic.mean()/f.rank_ic.std(ddof=1),
                     'mean_coefficient_bps_per_sd':10000*f.coefficient.mean(),
                     'mean_rank_corr_with_qmp':f.rank_corr_with_qmp.mean()})
    pd.DataFrame(rows).to_csv(OUT/'ic_summary.csv',index=False)
    rnd=pd.DataFrame(random_records).groupby(['scope','signal_date']).agg(random_ic=('rank_ic','mean'),drop_worst_ic=('drop_worst_ic','first'),qmp_ic=('qmp_ic','first')).reset_index()
    rnd.to_csv(OUT/'matched_random_weekly.csv',index=False)
    secondary=[]
    for scope,f in rnd.groupby('scope'):
        secondary.append({'scope':scope,**block_stats(f.drop_worst_ic-f.random_ic)})
    secondary=pd.DataFrame(secondary);secondary['holm_p_3']=holm(secondary.p)
    secondary.to_csv(OUT/'matched_random_contrasts.csv',index=False)
    # Pre-specified period-specific loss-free diagnostics of whether high disagreement predicts rule differences.
    pd.DataFrame(groups).groupby(['scope','disagreement_group','method']).agg(windows=('rank_ic','count'),mean_ic=('rank_ic','mean'),mean_disagreement=('mean_disagreement','mean')).reset_index().to_csv(OUT/'disagreement_group_summary.csv',index=False)
    return rec

def design(formed, method):
    ind=formed.industry.to_numpy(int)
    assert np.all((ind>=0)&(ind<=99))
    dummies=(ind[:,None]==np.arange(1,100)[None,:]).astype(float)
    base=np.column_stack([np.ones(len(formed)),dummies,normalized(formed[CONTROLS],True)])
    return base if method=='controls_only' else np.column_stack([base,normalized(formed[method].to_numpy())])

def forecasts(panel):
    models=['controls_only']+METHODS
    dim={m:107+(m!='controls_only') for m in models}
    stats={m:[np.zeros((dim[m],dim[m])),np.zeros(dim[m])] for m in models}
    pending=[];training=0;last_exit=None;rows=[];predictions=[]
    for day,f in panel.groupby('signal_date',sort=True):
        f=f.dropna(subset=CONTROLS+METHODS+['industry']).copy()
        f=f[~f.delisted_at_signal]
        assert len(f)>=300
        # Feature transformation precedes any outcome filtering.
        xs={m:design(f,m) for m in models}
        mature=[item for item in pending if item['exit_date']<=day]
        pending=[item for item in pending if item['exit_date']>day]
        for item in mature:
            valid=item['valid'];n=int(valid.sum());y=item['y'][valid]
            for m in models:
                x=item['xs'][m][valid]
                stats[m][0]+=x.T@x/n;stats[m][1]+=x.T@y/n
            training+=1;last_exit=item['exit_date'] if last_exit is None else max(last_exit,item['exit_date'])
        assert last_exit is None or last_exit<=day
        valid=f.week1.notna().to_numpy();y=f.week1.to_numpy(float)
        preds={}
        if training>=52:
            for m in models:
                a,b=stats[m];penalty=np.eye(dim[m]);penalty[0,0]=0
                coeff=solve(a+penalty,b,assume_a='pos')
                pred=xs[m]@coeff;preds[m]=pred
                rows.append({'scope':f.scope.iloc[0],'signal_date':day,'method':m,'n':int(valid.sum()),
                             'mature_training_windows':training,'latest_training_exit':last_exit,
                             'mse':float(np.mean((y[valid]-pred[valid])**2)),
                             'prediction_rank_ic':correlation(pred[valid],y[valid],True)})
            # Stock-level loss vectors remain local intermediate licensed-data derivatives.
            tmp=f[['permno','scope','signal_date','week1']].copy()
            for m,pred in preds.items():tmp['pred_'+m]=pred
            predictions.append(tmp)
        pending.append({'exit_date':f.exit_date.iloc[0],'valid':valid,'y':y,'xs':xs})
    result=pd.DataFrame(rows);result.to_csv(OUT/'forecast_weekly.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(WORK/'forecast_predictions.parquet',index=False)
    summary=[]
    for scope,f in result.groupby('scope'):
        pivot=f.pivot(index='signal_date',columns='method',values='mse')
        for method in models:
            summary.append({'scope':scope,'method':method,'windows':len(pivot),'mean_mse':pivot[method].mean(),
                            'mse_gain_vs_controls_percent':100*(1-pivot[method].mean()/pivot.controls_only.mean()),
                            'mse_gain_vs_qmp_percent':100*(1-pivot[method].mean()/pivot.qmp.mean())})
    pd.DataFrame(summary).to_csv(OUT/'forecast_summary.csv',index=False)
    print('Expanding forecasts complete',flush=True)
    return result

def contrasts(metrics,pred):
    rows=[]
    for scope in SCOPES:
        a=metrics[metrics.scope==scope].pivot(index='signal_date',columns='method',values='rank_ic')
        b=pred[pred.scope==scope].pivot(index='signal_date',columns='method',values='mse')
        for method in ['bjzz_common','qmp_weighted','qmp_drop_worst','proxy_blend']:
            for metric,values in [('rank_ic_gain',a[method]-a.qmp),('mse_reduction',b.qmp-b[method])]:
                s=block_stats(values);s8=block_stats(values,8)
                rows.append({'scope':scope,'method':method,'metric':metric,**s,
                             'ci8_low':s8['ci_low'],'ci8_high':s8['ci_high']})
    result=pd.DataFrame(rows);assert len(result)==24
    result['holm_p_24']=holm(result.p);result.to_csv(OUT/'primary_contrasts.csv',index=False)
    return result

def simulation():
    records=[];analytic=[]
    for n in [20,100,500]:
      for weight_type in ['equal','lognormal_sigma1.5']:
       for rep in range(8):
        rng=np.random.default_rng(SEED+n*100+rep+(0 if weight_type=='equal' else 50000))
        mu=rng.uniform(-.25,.25,size=1500)
        signs=np.where(rng.random((1500,n))<(1+mu[:,None])/2,1.,-1.)
        v=np.ones_like(signs) if weight_type=='equal' else rng.lognormal(0,1.5,size=signs.shape)
        w=v/v.sum(axis=1,keepdims=True);true=np.sum(w*signs,axis=1)
        y=.15*normalized(true)+rng.normal(size=len(true));true_ic=correlation(true,y)
        true_rank=correlation(true,y,True);neff=1/np.sum(w*w,axis=1)
        for e in [0,.05,.1,.2,.3,.4]:
         for mechanism in ['independent_symmetric','stock_dependent_symmetric']:
            rates=np.full(len(true),e) if mechanism=='independent_symmetric' else np.clip(e*(1+.8*np.sign(true)),0,.49)
            flips=np.where(rng.random(signs.shape)<rates[:,None],-1.,1.)
            obs=np.sum(w*signs*flips,axis=1)
            a=1-2*rates;expected=a*true;varnoise=4*rates*(1-rates)*np.sum(w*w,axis=1)
            residual=obs-expected
            # Conditional-noise approximation works for both mechanisms using the actual mean function.
            expected_ic=correlation(expected,y)*np.sqrt(np.var(expected)/(np.var(expected)+varnoise.mean()))
            records.append({'trade_count':n,'weights':weight_type,'rep':rep,'mechanism':mechanism,'base_flip_rate':e,
                            'mean_actual_flip_rate':rates.mean(),'mean_effective_trade_count':neff.mean(),
                            'true_pearson_ic':true_ic,'observed_pearson_ic':correlation(obs,y),
                            'predicted_pearson_ic':expected_ic,'true_spearman_ic':true_rank,
                            'observed_spearman_ic':correlation(obs,y,True),
                            'incorrect_linear_spearman_prediction':(1-2*e)*true_rank,
                            'conditional_mean_residual':residual.mean(),
                            'empirical_noise_mse':np.mean(residual**2),'predicted_noise_variance':varnoise.mean()})
        if rep==0:print('Simulation:',n,weight_type,flush=True)
    results=pd.DataFrame(records);results.to_csv(OUT/'simulation_replications.csv',index=False)
    results.groupby(['trade_count','weights','mechanism','base_flip_rate']).mean(numeric_only=True).reset_index().to_csv(OUT/'simulation_summary.csv',index=False)
    return results

def integrity(files):
    changed=[p for p,h in files.items() if sha(p)!=h]
    assert not changed,changed
    return {'original_files_unchanged':True,'validated_input_files':len(files),'input_hashes_file':'input_hashes.json'}

def main():
    protocol=OUT/'protocol.json';receipt=json.loads((OUT/'protocol_receipt.json').read_text())
    assert sha(protocol)==receipt['sha256']
    enriched=WORK/'enriched_panel.parquet'
    if enriched.exists():
        panel=pd.read_parquet(enriched);files=json.loads((WORK/'input_hashes.json').read_text())
        assert all(sha(p)==h for p,h in files.items())
    else:panel,files=load_and_enrich()
    metrics=empirical(panel);pred=forecasts(panel);comparisons=contrasts(metrics,pred);sim=simulation()
    check=integrity(files)
    (OUT/'input_hashes.json').write_text(json.dumps(files,indent=2))
    run={'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete':True,
         'protocol_sha256':sha(protocol),'code_sha256':sha(__file__),'source_code':str(SOURCE_CODE),
         'panel_rows':len(panel),'formation_windows':panel.signal_date.nunique(),
         'simulation_replications':len(sim),'primary_tests':len(comparisons),**check,
         'empirical_scope':'Previously inspected historical data; exploratory chronological forecasts, no untouched new holdout',
         'forecast_uncertainty':'Block resampling of observed fitted losses, conditional on the trained models; no full training refit',
         'not_established':['true signing accuracy','causal impact of measurement error','cost-adjusted trading gain','new-period confirmation']}
    (OUT/'run_manifest.json').write_text(json.dumps(run,indent=2))
    print(json.dumps(run,indent=2),flush=True)

if __name__=='__main__':main()
