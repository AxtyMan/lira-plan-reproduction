"""Independent query evaluator and support-seed paired statistics."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from .core import softmax


def check(logits,y):
    z=np.asarray(logits,dtype=float); y=np.asarray(y)
    if z.ndim!=2 or len(z)==0 or not np.isfinite(z).all() or y.shape!=(len(z),) or y.dtype.kind not in 'iu' or np.any((y<0)|(y>=z.shape[1])):
        raise ValueError('invalid logits or label index')
    return z,y


def temperature(logits,y):
    z,y=check(logits,y)
    def loss(log_t):
        a=z/np.exp(log_t); return float(np.mean(logsumexp(a,axis=1)-a[np.arange(len(y)),y]))
    r=minimize_scalar(loss,bounds=(-5,5),method='bounded',options={'xatol':1e-8});
    if not r.success: raise RuntimeError('temperature fitting failed')
    return float(np.exp(r.x))


def metrics(logits,y,*,ids=None,base=None):
    z,y=check(logits,y); c=z.shape[1]; ix=np.arange(len(y)); p=softmax(z); pred=z.argmax(1); ok=pred==y
    cf=np.zeros((c,c),int); np.add.at(cf,(y,pred),1); tp=cf.diagonal()
    rec=np.divide(tp,cf.sum(1),out=np.zeros(c),where=cf.sum(1)>0)
    den=cf.sum(0)+cf.sum(1); f=np.divide(2*tp,den,out=np.zeros(c),where=den>0)
    conf=p.max(1); bins=np.minimum((conf*15).astype(int),14)
    ece=sum(abs((ok[bins==b]-conf[bins==b]).sum()) for b in range(15))/len(y)
    tie=np.asarray(ids,dtype=str) if ids is not None else ix
    order=np.lexsort((tie,-conf)); curve=np.cumsum(~ok[order])/(ix+1)
    out=dict(accuracy=float(ok.mean()),macro_f1=float(f.mean()),recall=rec.tolist(),confusion=cf.tolist(),
             brier=float(np.square(p-np.eye(c)[y]).sum(-1).mean()),nll=float(-np.log(np.clip(p[ix,y],1e-12,1.)).mean()),
             ece=float(ece),aurc=float(curve.mean()),risk_at_coverage={str(s):float(curve[int(np.ceil(s*len(y)))-1]) for s in (.5,.8,.9,1.)})
    if base is not None:
        bz,_=check(base,y)
        if bz.shape!=z.shape: raise ValueError('unpaired visual logits')
        v=bz.argmax(1)==y; h=float(np.mean(v & ~ok)); r=float(np.mean(~v & ok))
        if not np.isclose(ok.mean()-v.mean(),r-h,rtol=0,atol=1e-12): raise AssertionError('paired identity')
        out.update(harm=h,rescue=r,visual_accuracy=float(v.mean()))
    return out


def paired_interval(values,seed=90117,n_boot=10000):
    v=np.asarray(values,dtype=float)
    if v.ndim!=1 or len(v)==0 or not np.isfinite(v).all(): raise ValueError('nonempty finite paired seed differences required')
    if len(v)<2: return {'seed_pairs':len(v),'difference':float(v.mean()),'ci95':None}
    rng=np.random.default_rng(seed); ix=rng.integers(len(v),size=(n_boot,len(v)))
    return {'seed_pairs':len(v),'difference':float(v.mean()),'ci95':np.quantile(v[ix].mean(1),(.025,.975)).tolist()}


def holm(p):
    x=np.asarray(p,dtype=float); ix=x.argsort(); result=np.empty(len(x));
    result[ix]=np.minimum(1,np.maximum.accumulate(x[ix]*np.arange(len(x),0,-1)))
    return result
