"""Outcome fences, non-overlapping schedules, constrained sizing and CIZ settlement."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize,brentq

class MissingHeldReturn(ValueError):pass

def weekly_windows(calendar,start,end):
    calendar=pd.DatetimeIndex(calendar).sort_values();start=pd.Timestamp(start);end=pd.Timestamp(end)
    ends=pd.Series(calendar,index=calendar).groupby(calendar.to_period('W-FRI')).max()
    rows=[];previous_exit=None
    for signal in ends:
        # A truncated last partial week is not a weekly final session.
        if signal.to_period('W-FRI')==calendar[-1].to_period('W-FRI') and calendar[-1].weekday()<4:continue
        if signal<start or signal>end:continue
        i=calendar.get_loc(signal);entry=i+1;exit=entry+5
        if exit>=len(calendar):continue
        if calendar[entry]<start or calendar[exit]>end:continue
        if previous_exit is not None and entry<previous_exit:continue
        rows.append({'signal_date':signal,'entry_date':calendar[entry],'exit_date':calendar[exit],
            'signal_index':i,'entry_index':entry,'exit_index':exit})
        previous_exit=exit
    return pd.DataFrame(rows)

def label(returns,delisted,duration):
    """Cash after the supplied CIZ terminal row; no retirement-return duplication."""
    value=1.
    for r,d,flag in zip(returns,delisted,duration):
        terminal=str(d)=='Y'
        valid_flag=str(flag) in ['D1','D2','D3','D4','DU'] or (terminal and str(flag)=='DD')
        if not np.isfinite(r) or r< -1 or not valid_flag:return np.nan
        value*=1+r
        if terminal:return float(value-1)
    return float(value-1)

def settle(weights,returns,delisted,duration):
    w=np.asarray(weights,float);r=np.asarray(returns,float);held=np.abs(w)>1e-12
    terminal=np.asarray(delisted).astype(str)=='Y';flags=np.asarray(duration).astype(str)
    valid=np.isin(flags,['D1','D2','D3','D4','DU'])|(terminal&(flags=='DD'))
    if ((~np.isfinite(r)|(r< -1)|~valid)&held).any():raise MissingHeldReturn('Missing or non-daily return on held security')
    r=np.where(held,r,0);pnl=float(w@r)
    if pnl<=-1:raise ValueError('Portfolio capital exhausted')
    drifted=w*(1+r)/(1+pnl)
    settled_positions=drifted.copy();settled_positions[~terminal]=0
    drifted[terminal]=0
    return drifted,pnl,settled_positions

def resize(weights,target,rates,entry_rates=None):
    """Post-cost target weights, with cash cost exactly reconciled to traded dollars."""
    w=np.asarray(weights,float);target=np.asarray(target,float);rates=np.asarray(rates,float)
    entry_rates=rates if entry_rates is None else np.asarray(entry_rates,float)
    if not np.isfinite(rates).all() or (rates<0).any():raise ValueError('Invalid known execution cost rates')
    if not np.isfinite(entry_rates).all() or (entry_rates<0).any():raise ValueError('Invalid entry rates')
    def costs(ratio):
        after=ratio*target;same=w*after>0;difference=np.abs(after)-np.abs(w)
        return np.where(same,np.maximum(difference,0)*entry_rates+np.maximum(-difference,0)*rates,
            np.abs(w)*rates+np.abs(after)*entry_rates)
    f=lambda ratio:ratio+costs(ratio).sum()-1
    if f(0)>=0:raise ValueError('Execution cost exhausted capital')
    ratio=brentq(f,0,1,xtol=1e-14);changes=ratio*target-w
    return target.copy(),ratio,changes,costs(ratio)

def constrained_weights(signal,beta,industry,cap=.01):
    ranks=pd.Series(np.asarray(signal,float)).rank().to_numpy();score=ranks-ranks.mean()
    groups=pd.get_dummies(pd.Series(industry),dtype=float).to_numpy();risk=np.column_stack([groups,np.asarray(beta,float)])
    projected=score-risk@np.linalg.lstsq(risk,score,rcond=None)[0]
    if np.abs(projected).sum()<1e-10:raise ValueError('No neutral signal')
    desired=projected*2/np.abs(projected).sum();sign=np.sign(desired)
    if (sign>0).sum()*cap<1 or (sign<0).sum()*cap<1:raise ValueError('Weight cap infeasible for fixed neutral-score signs')
    lower=np.where(sign<0,-cap,0);upper=np.where(sign>0,cap,0)
    a=np.vstack([risk.T,sign]);b=np.r_[np.zeros(risk.shape[1]),2.]
    scale=np.sqrt((a*a).sum(axis=1));scale=np.where(scale>0,scale,1);a=a/scale[:,None];b=b/scale
    def dual(lam):
        w=np.clip(desired-a.T@lam,lower,upper)
        g=.5*(w@w)-desired@w+lam@(a@w-b)
        return -g,b-a@w
    fit=minimize(dual,np.zeros(len(b)),method='L-BFGS-B',jac=True,options={'maxiter':5000,'ftol':1e-15,'gtol':1e-11,'maxls':100})
    w=np.clip(desired-a.T@fit.x,lower,upper)
    if np.max(np.abs(risk.T@w))>1e-7 or abs(abs(w).sum()-2)>1e-7 or abs(w).max()>cap+1e-10:
        raise ValueError('Weight constraints not solved to frozen tolerance')
    return w

def ols(signal,y,controls,industry):
    signal=np.asarray(signal,float);y=np.asarray(y,float);z=np.asarray(controls,float)
    sd=signal.std()
    if sd<1e-12:raise ValueError('Degenerate signal')
    s=(signal-signal.mean())/sd;scale=z.std(axis=0);scale=np.where(scale>1e-12,scale,1)
    x=np.column_stack([pd.get_dummies(pd.Series(industry),dtype=float).to_numpy(),(z-z.mean(axis=0))/scale])
    rs=s-x@np.linalg.lstsq(x,s,rcond=None)[0];ry=y-x@np.linalg.lstsq(x,y,rcond=None)[0]
    if rs@rs<1e-10:raise ValueError('Collinear signal')
    coefficient=float(rs@ry/(rs@rs));fit=np.linalg.lstsq(np.column_stack([x,s]),y,rcond=None)[0][-1]
    if abs(coefficient-fit)>1e-8:raise AssertionError('FWL/OLS mismatch')
    return coefficient,s,rs,ry

def block_means(values,n,draws,block,seed):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    if len(x)<12:raise ValueError('Insufficient weekly observations for 4-week block inference')
    rng=np.random.default_rng(seed);starts=rng.integers(0,len(x),size=(draws,int(np.ceil(n/block))))
    indices=(starts[:,:,None]+np.arange(block)[None,None,:])%len(x)
    return x[indices.reshape(draws,-1)[:,:n]].mean(axis=1)

def inference(values,plan):
    x=np.asarray(values,float);x=x[np.isfinite(x)];p=plan['inference']
    means=block_means(x,len(x),p['draws'],p['block_weeks'],p['seed']);ci=np.quantile(means,[.025,.975])
    z=x-x.mean();lag=p['hac_lag_weeks'];lr=float(z@z/len(z))
    for k in range(1,min(lag,len(z)-1)+1):lr+=2*(1-k/(lag+1))*float(z[k:]@z[:-k]/len(z))
    se=np.sqrt(max(lr,0)/len(z))
    return {'n':len(x),'mean':float(x.mean()),'ci_low':float(ci[0]),'ci_high':float(ci[1]),'hac_se':float(se),'hac_t':float(x.mean()/se) if se>0 else None}
