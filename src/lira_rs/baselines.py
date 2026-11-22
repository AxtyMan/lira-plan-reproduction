"""Baseline controls and faithful formula replay of TIMO's non-S head.

TGI uses *raw* similarities on text pseudo-observations, not IGT softmax weights.
Source: author models.py (GDA/TIMO), utils.py (image_guide_text/vec_sort).
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .core import EPS, State, anchors, means, text_valid, unit, image_unit, grouped, score

ALPHAS=np.array([10.**i for i in range(-4,5)])


def text_average(t,mask):
    valid=text_valid(t,mask)
    return unit(unit(np.where(valid[...,None],t,0.)).sum(1))


def parts(x,y,t,mask,cfg,timo):
    c=len(t)
    if timo:
        valid=text_valid(t,mask); text=unit(np.where(valid[...,None],t,0.))
        a,raw,_=anchors(means(x,y,c),text,mask,cfg)
        ix=np.argsort(-raw,axis=1,kind='stable')
        pseudo=np.take_along_axis(text,ix[...,None],axis=1)*np.take_along_axis(raw,ix,axis=1)[...,None]
        keep=np.take_along_axis(valid,ix,axis=1)
        # Non-S beta = number of available prompts; masked-out slots are not pseudo-observations.
        x=np.concatenate((pseudo[keep],x)); y=np.concatenate((np.broadcast_to(np.arange(c)[:,None],keep.shape)[keep],y))
        w_text=a.T
    else: w_text=100*text_average(t,mask).T
    mus=means(x,y,c); u=x-mus[y]; d=x.shape[1]
    trace=(u*u).sum()/max(len(x)-1,1)
    if trace<=EPS: w=np.zeros((d,c))  # pinv of zero; GDA Fixed is then a text-only rule
    elif len(x)<d:
        v=u@mus.T
        h=u@u.T; h.flat[::len(x)+1]+=trace
        w=(d/trace)*(mus.T-u.T@cho_solve(cho_factor(h,check_finite=False),v,check_finite=False))
    else:
        h=u.T@u; h.flat[::d+1]+=trace
        w=d*cho_solve(cho_factor(h,check_finite=False),mus.T,check_finite=False)
    b=-np.log(c)-.5*np.sum(mus.T*w,axis=0)
    return w_text,w,b


def alpha_logits(z,parts_,timo):
    tw,w,b=parts_; a=z@tw; p=z@w+b
    return a[...,None,:]*ALPHAS[:,None]+p[...,None,:] if timo else a[...,None,:]+ALPHAS[:,None]*p[...,None,:]


def linear(parts_,alpha,timo,log):
    tw,w,b=parts_
    ww,bb=(alpha*tw+w,b) if timo else (tw+alpha*w,alpha*b)
    return State('linear',{'weight':ww,'bias':bb},{**log,'alpha':float(alpha)})


def allowed_selection(ids,selection_ids,selection_y,c):
    ids=list(ids); selection_ids=list(selection_ids); y=np.asarray(selection_y)
    if len(y)!=len(selection_ids) or len(set(selection_ids))!=len(y) or set(ids)&set(selection_ids):
        raise ValueError('selection IDs must be unique and disjoint from support')
    if y.ndim!=1 or y.dtype.kind not in 'iu' or np.any((y<0)|(y>=c)) or np.any(np.bincount(y,minlength=c)==0):
        raise ValueError('invalid selection labels')


def replay_selection(x,y,t,mask,cfg,z_select,y_select,ids,selection_ids,timo=True):
    """Protocol B / author-formula replay ONLY; never callable from the A fit path."""
    allowed_selection(ids,selection_ids,y_select,len(t))
    g=grouped(x,y,ids); pr=parts(g.x,g.y,t,mask,cfg,timo)
    z=image_unit(z_select); acc=(alpha_logits(z,pr,timo).argmax(-1)==np.asarray(y_select)[:,None]).mean(0)
    j=int(acc.argmax()); alpha=ALPHAS[j] if acc[j]>0 else cfg.gda_fixed_alpha
    return linear(pr,alpha,timo,dict(method='timo-selection' if timo else 'gda-selection',selection_n=len(z),selection_accuracy=acc.tolist(),alpha_grid=ALPHAS.tolist()))


def loo_scores_fast(x,y,t,mask,cfg,timo):
    """Exact low-rank class-local scatter update, followed by a small signed solve.

    Holding out one image changes one class's visual mean, TGI rows, and scatter.
    Recalculate the ridge trace and IGT scale per fold; never reuse full text anchors.
    """
    c=len(t); d=x.shape[1]; n=len(x); vis=means(x,y,c); counts=np.bincount(y,minlength=c)
    valid=text_valid(t,mask); safe=unit(np.where(valid[...,None],t,0.))
    if timo:
        full_a,raw,_=anchors(vis,t,mask,cfg); pseudo=safe*raw[...,None]; counts=counts+valid.sum(1)
    else:
        full_a=100*text_average(t,mask); pseudo=np.zeros_like(safe); valid=np.zeros_like(valid)
    sums=vis*np.bincount(y,minlength=c)[:,None]+pseudo.sum(1); mu=sums/counts[:,None]
    residual=x-mu[y]; pt=(pseudo-mu[:,None])[valid]
    scatter=residual.T@residual+pt.T@pt
    eig,Q=np.linalg.eigh(scatter); eig=np.maximum(eig,0.); trace=float(np.trace(scatter)); N=int(counts.sum())
    Qx=x@Q; Qmu=mu@Q; Qt=safe@Q; Qpseudo=pseudo@Q; vys=np.bincount(y,minlength=c); votes=[]
    for i,cl in enumerate(y):
        remain=(vys[cl]*vis[cl]-x[i])/(vys[cl]-1); num=counts[cl]; sl=valid[cl]
        if timo:
            a,r,_=anchors(remain[None],t[cl:cl+1],mask[cl:cl+1],cfg)
            newp=safe[cl,sl]*r[0,sl,None]; Qnewp=Qt[cl,sl]*r[0,sl,None]
        else: newp=np.empty((0,d)); Qnewp=newp; a=None
        newmu=(vys[cl]*vis[cl]-x[i]+newp.sum(0))/(num-1); Qnewmu=newmu@Q
        pos=np.vstack([Qnewp,np.sqrt(num)*Qmu[cl]]); neg=np.vstack([Qpseudo[cl,sl],Qx[i],np.sqrt(num-1)*Qnewmu]);
        ridge=(trace+(pos*pos).sum()-(neg*neg).sum())/max(N-2,1)
        z=image_unit(x[i:i+1])[0]; tz=z@full_a.T
        if timo: tz[cl]=z@a[0]
        if ridge<=EPS:
            if ridge < -1e-6: raise ArithmeticError('negative post-deletion scatter trace')
            v=np.full(c,-np.log(c))
        else:
            b=Qmu.copy(); b[cl]=Qnewmu; inv=1/(eig+ridge); U=np.vstack([pos,neg]); e=U*inv[None]
            sign=np.r_[np.ones(len(pos)),-np.ones(len(neg))]
            glue=e@U.T; glue.flat[::len(U)+1]+=sign
            w=inv[:,None]*b.T-e.T@np.linalg.solve(glue,e@b.T)
            v=d*((z@Q)@w-.5*np.sum(b.T*w,axis=0))-np.log(c)
        votes.append(tz[None]*ALPHAS[:,None]+v[None] if timo else tz[None]+ALPHAS[:,None]*v[None])
    return np.asarray(votes)


def fit_other(g,t,mask,cfg,method,k):
    x,y=g.x,g.y; c=len(t); log=dict(method=method,K=k,support_ids=g.ids,support_view_counts=g.views)
    w_text=100*(unit(np.where(np.isfinite(t[:,0]),t[:,0],0.)) if method=='zs' else text_average(t,mask)).T
    if method in {'zs','ensemble'}: return State('linear',dict(weight=w_text,bias=np.zeros(c)),log)
    if method=='tip': return State('tip',dict(weight=w_text,alpha=np.array(cfg.tip_alpha),beta=np.array(cfg.tip_beta),keys=image_unit(x),values=np.eye(c)[y]),log)
    if method=='prolip': raise ValueError('ProLIP requires pre-projection features; use fit_projection')
    timo=method=='timo-loo'; a=cfg.gda_fixed_alpha
    if k==1: log.update(method='timo-fixed' if timo else 'gda-fixed',selection='predeclared',fallback_reason='one_shot')
    else:
        fold=loo_scores_fast(x,y,t,mask,cfg,timo).argmax(-1)
        hits=(fold==y[:,None]).sum(0); a=ALPHAS[int(hits.argmax())]
        log.update(selection='support_group_loo',loo_accuracy=(hits/len(x)).tolist(),alpha_grid=ALPHAS.tolist(),loo_predictions=fold.tolist())
    return linear(parts(x,y,t,mask,cfg,timo),a,timo,log)
