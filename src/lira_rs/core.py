"""Equations (1)–(13). Fit accepts no query or external labels."""
from dataclasses import dataclass
import numpy as np

EPS=1e-8

@dataclass(frozen=True)
class Config:
    eta: float=.5
    gamma: float=50.
    tau: float=.07
    lambdas: tuple=(0.,.125,.25,.5,1.)
    fixed: float=.5
    tip_alpha: float=20.
    tip_beta: float=5.5
    gda_fixed_alpha: float=.1

    def __post_init__(self):
        if not (0<=self.eta<1 and self.tau>0 and self.gamma>=0 and 0<=self.fixed<=1
                and self.tip_beta>0 and self.tip_alpha>=0 and self.gda_fixed_alpha>0):
            raise ValueError('invalid geometry or scale')
        if tuple(sorted(set(self.lambdas)))!=tuple(self.lambdas) or not self.lambdas or self.lambdas[0]!=0 or self.lambdas[-1]>1:
            raise ValueError('lambda grid must start at zero and be sorted and unique within [0,1]')

@dataclass
class Groups:
    x: np.ndarray
    y: np.ndarray
    ids: list
    views: list

@dataclass
class State:
    head: str
    arrays: dict
    log: dict


def unit(x):
    x=np.asarray(x,dtype=np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),EPS)


def image_unit(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or len(x)==0 or not np.isfinite(x).all() or np.any(np.linalg.norm(x,axis=1)<EPS):
        raise ValueError('invalid image feature matrix')
    return unit(x)


def softmax(x,axis=-1):
    v=np.exp(x-np.max(x,axis=axis,keepdims=True)); return v/v.sum(axis=axis,keepdims=True)


def grouped(x,y,ids=None):
    x=image_unit(x); y=np.asarray(y)
    ids=list(map(str,range(len(x)) if ids is None else ids))
    if y.shape!=(len(x),) or y.dtype.kind not in 'iu' or len(ids)!=len(x) or any(not v for v in ids):
        raise ValueError('support feature, integer label or ID shape mismatch')
    by={}
    for i,k in enumerate(ids): by.setdefault(k,[]).append(i)
    a,b,n=[],[],[]
    for idx in by.values():
        if len(set(y[idx]))!=1: raise ValueError('one image ID has conflicting labels')
        # Equal weight per original photo. Mean of unit views is not renormalized for p.
        a.append(x[idx].mean(0)); b.append(y[idx[0]]); n.append(len(idx))
    return Groups(np.array(a),np.array(b),list(by),n)


def means(x,y,c):
    count=np.bincount(y,minlength=c)
    if len(count)!=c or np.any(count==0): raise ValueError('every class must have support')
    p=np.zeros((c,x.shape[1])); np.add.at(p,y,x)
    return p/count[:,None]


def text_valid(t,mask):
    return np.asarray(mask,dtype=bool) & np.isfinite(t).all(-1) & (np.linalg.norm(np.where(np.isfinite(t),t,0),axis=-1)>EPS)


def anchors(p,t,mask,cfg,uniform=False):
    # p: [..., C,D]; masks may remove slots, never the core.
    valid=text_valid(t,mask); safe=unit(np.where(valid[...,None],t,0.))
    sim=np.einsum('...cd,cjd->...cj',unit(p),safe)
    sn=np.linalg.norm(sim,axis=-1,keepdims=True)
    weak=(np.linalg.norm(p,axis=-1,keepdims=True)<=EPS)|(sn<=EPS)
    logits=np.where(weak|uniform,0.,cfg.gamma*sim/np.maximum(sn,EPS))
    w=softmax(np.where(valid,logits,-1e10))*valid
    raw=np.einsum('...cj,cjd->...cd',w,safe)
    near=np.linalg.norm(raw,axis=-1)<=EPS
    a=np.where(near[...,None],safe[:,0],unit(raw))
    invalid=~valid[:,0]  # never move a class whose mandatory core is invalid
    a=np.where(invalid[...,None],p,a)
    statuses=np.where(invalid,'invalid_text',np.where(near,'core_text_fallback',np.where(weak[...,0],'uniform_degenerate','ok')))
    return a,sim,statuses


def radius(p,eta):
    gram=p@np.swapaxes(p,-2,-1); pp=(p*p).sum(-1)
    distances=np.maximum(pp[...,None]+pp[...,None,:]-2*gram,0.)
    distances[...,np.arange(p.shape[-2]),np.arange(p.shape[-2])]=np.inf
    return eta*.5*np.sqrt(distances.min(-1))


def directions(p,a,r,bounded):
    u=a-p
    return u*np.minimum(1.,r[...,None]/np.maximum(np.linalg.norm(u,axis=-1,keepdims=True),EPS)) if bounded else u


def geometry(x,y,t,mask,cfg,bounded=True,uniform=False):
    p=means(x,y,len(t)); a,s,status=anchors(p,t,mask,cfg,uniform); r=radius(p,cfg.eta)
    return p,a,r,directions(p,a,r,bounded),s,status


def distance(z,p):
    return np.maximum((z*z).sum(-1)[...,None]+(p*p).sum(-1)-2*np.einsum('...d,...cd->...c',z,p),0.)


PROTOS={'visual','fixed','timo-proto','lira','no-bound','fixed-bounded','full-risk','no-zero','no-igt'}
OTHER={'zs','ensemble','tip','gda-loo','timo-loo','prolip'}
METHODS=tuple(sorted(PROTOS|OTHER))


def fit(x,y,t,config=Config(),*,ids=None,mask=None,method='lira'):
    g=grouped(x,y,ids); x,y=g.x,g.y; t=np.asarray(t,dtype=np.float64)
    if t.ndim!=3 or t.shape[0]<2 or t.shape[1]!=4 or t.shape[2]!=x.shape[1]:
        raise ValueError('expected C×4×D prompt array (core plus 3 slots)')
    mask=np.ones(t.shape[:2],bool) if mask is None else np.asarray(mask,dtype=bool)
    if mask.shape!=t.shape[:2] or not mask[:,0].all(): raise ValueError('invalid prompt mask')
    c=len(t)
    if np.any((y<0)|(y>=c)): raise ValueError('label outside frozen class order')
    counts=np.bincount(y,minlength=c)
    if counts.min()==0 or counts.min()!=counts.max(): raise ValueError('expected balanced C×K distinct IDs')
    k=int(counts[0]); cfg=config
    if method not in METHODS: raise ValueError(f'unknown method {method}')
    if method not in PROTOS:
        from .baselines import fit_other
        return fit_other(g,t,mask,cfg,method,k)
    bounded=method not in {'no-bound','timo-proto','fixed','visual'}; uniform=method=='no-igt'
    p,a,r,u,_,status=geometry(x,y,t,mask,cfg,bounded,uniform)
    log=dict(method=method,K=k,visual_centers=p.tolist(),anchors=a.tolist(),radius=r.tolist(),
             class_status=status.tolist(),lambda_grid=list(cfg.lambdas),support_ids=g.ids,support_view_counts=g.views)
    lam=0.; reason=None
    if method=='visual': reason='visual_only'
    elif method=='fixed': lam=cfg.fixed
    elif k==1: reason='one_shot'
    elif np.all(status=='invalid_text'): reason='invalid_text'
    elif method=='fixed-bounded': lam=cfg.fixed
    else:
        labels=np.eye(c)[y]; fp=np.broadcast_to(p,(len(x),)+p.shape).copy()
        fp[np.arange(len(x)),y]=(k*p[y]-x)/(k-1)
        fr=radius(fp,cfg.eta)
        conditions=['full','drop_1','drop_2','drop_3']; use=conditions if method not in {'timo-proto','full-risk'} else ['full']
        grid=np.asarray(cfg.lambdas); probs=[]; fold_status=[]
        for e,name in enumerate(use):
            fm=mask.copy()
            if name!='full': fm[:,e]=False
            fa,_,fs=anchors(fp,t,fm,cfg,uniform); fold_status.append(fs.tolist())
            fu=directions(fp,fa,fr,bounded)
            mu=fp[:,None]+grid[None,:,None,None]*fu[:,None]
            probs.append(softmax(-distance(x[:,None,:],mu)/cfg.tau))
        probabilities=np.stack(probs,axis=1)
        folds=((probabilities-labels[:,None,None,:])**2).sum(-1)
        risk=np.stack([folds[y==i].mean(0) for i in range(c)]).mean(0)
        worst=(risk-risk[:,[0]]).max(0)
        allowed=(worst<=EPS) if method not in {'timo-proto','no-zero'} else np.ones(len(grid),bool)
        if method=='no-zero':
            allowed[0]=False
            if not np.any(allowed): raise ValueError('no-zero ablation requires a nonzero candidate')
        chosen=int(np.flatnonzero(allowed & (risk[0]<=risk[0,allowed].min()+EPS))[0]); lam=float(grid[chosen])
        if lam==0: reason='risk_rejection' if not allowed[1:].any() else 'zero_minimum'
        log.update(conditions=use,fold_probabilities=probabilities.tolist(),fold_brier=folds.tolist(),
                   fold_text_status=fold_status,risk=risk.tolist(),worst_delta=worst.tolist(),
                   allowed=allowed.tolist(),rejected_count=int((~allowed).sum()),
                   fold_removed_ids=g.ids,fold_removed_views=g.views,
                   fold_remaining_ids=[g.ids[:i]+g.ids[i+1:] for i in range(len(x))])
    delta=lam*u; mu=p+delta
    log.update({'lambda':lam,'fallback_reason':reason,'displacement':np.linalg.norm(delta,axis=1).tolist(),'bounded':bounded})
    return State('distance',dict(mu=mu,tau=np.array(cfg.tau),p=p,delta_bound=lam*r if bounded else np.linalg.norm(delta,axis=1)),log)


def score(x,state):
    z=image_unit(x); v=state.arrays
    if state.head=='distance':
        return -np.maximum((z*z).sum(1)[:,None]+(v['mu']**2).sum(1)-2*z@v['mu'].T,0.)/v['tau']
    if state.head=='linear': return z@v['weight']+v['bias']
    if state.head=='tip': return z@v['weight']+v['alpha']*np.exp(-v['beta']*(1-np.clip(z@v['keys'].T,-1,1)))@v['values']
    if state.head=='projection': return image_unit(np.asarray(x)@v['projection'])@v['weight']
    raise ValueError('unknown scoring head')


def stability(x,p,beta):
    z=image_unit(x); ds=np.maximum((z*z).sum(1)[:,None]+(p*p).sum(1)-2*z@p.T,0.)
    win=ds.argmin(1); ix=np.arange(len(z)); gap=ds-ds[ix,win,None]
    bound=2*np.sqrt(ds)*beta+beta**2; bound+=bound[ix,win,None]
    gap[ix,win]=np.inf
    return gap.min(1),np.all(gap>bound,axis=1)
