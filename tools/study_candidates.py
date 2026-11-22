"""Literal Eq. 1–11 audit and query effects of every preregistered candidate in the primary experiment."""
import csv
import os
from pathlib import Path
import sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import numpy as np
from scipy.special import softmax
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lira_rs import data, core
from lira_rs.cli import config,read_images,read_text


def normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)


def centers(x,y,text,mask,cfg):
    p=np.stack([x[y==c].mean(axis=0) for c in range(len(text))])
    usable=mask & np.isfinite(text).all(-1) & (np.linalg.norm(text,axis=-1)>1e-8)
    t=normalize(np.where(usable[...,None],text,0))
    similarity=np.sum(normalize(p)[:,None,:]*t,axis=-1)
    weighting=cfg.gamma*normalize(similarity)
    weights=softmax(np.where(usable,weighting,-1e12),axis=1)*usable
    weighted=(weights[...,None]*t).sum(1)
    anchor=np.where(np.linalg.norm(weighted,axis=-1,keepdims=True)>1e-8,normalize(weighted),t[:,0])
    delta=np.where(usable[:,0,None],anchor-p,0.)
    pairwise=np.linalg.norm(p[:,None,:]-p[None,:,:],axis=-1)
    np.fill_diagonal(pairwise,np.inf)
    radius=cfg.eta*pairwise.min(axis=1)/2
    return p,delta*np.minimum(1.,radius/np.maximum(1e-8,np.linalg.norm(delta,axis=-1)))[:,None],radius


def literal_risk(x,y,text,mask,cfg):
    x=normalize(np.asarray(x,dtype='float64')); text=np.asarray(text,dtype='float64'); y=np.asarray(y)
    ls=np.array(cfg.lambdas); num_classes=len(text)
    errors=np.zeros((num_classes,4,len(ls))); ids=np.arange(len(y))
    for i in ids:
        keep=ids!=i
        for condition in range(4):
            slots=mask.copy()
            if condition:slots[:,condition]=False
            p,direction,_=centers(x[keep],y[keep],text,slots,cfg)
            mu=p[None]+ls[:,None,None]*direction[None]
            prob=softmax(-((x[i]-mu)**2).sum(-1)/cfg.tau,axis=-1)
            prob[:,y[i]]-=1
            errors[y[i],condition]+=(prob**2).sum(-1)
    return (errors/np.bincount(y)[:,None,None]).mean(axis=0)


def main():
    settings,cfg=config(ROOT/'configs/main.yaml')
    mpath,manifest,cache,idx,image_meta=read_images('resisc45')
    batch=ROOT/'results/replay-primary'; result=[]; grouped=[]; confusion=[]; audit=[]
    qi=np.array([idx[r['id']] for r in manifest['images'] if r['role']=='query'])
    qx=normalize(cache['z'][qi].astype('float64')); qids=cache['ids'][qi]
    # All fixed-lambda candidates are locked before query scoring. No selection is made from these diagnostics.
    truth={r['id']:r['label'] for r in data.load_json(mpath.with_name('query.labels.json'))}
    labels=np.array([truth[i] for i in qids]); classes=manifest['classes']
    for seed in settings['support_seeds']:
        e=data.load_json(batch/f'episodes/resisc45/A-4-{seed}.json'); ids=e['roles']['support']; x=normalize(cache['z'][[idx[i] for i in ids]].astype('float64')); y=np.array(list(ids.values()))
        for cond in ('normal','wrong30'):
            t,_=read_text('resisc45',cond,7701,image_meta); text=t['t'].astype('float64'); masks=t['mask']
            cell=batch/f'cells/resisc45/A/K04/S{seed}/{cond}-7701/lira'; fit=data.load_json(cell/'fit.json')
            risk=literal_risk(cache['z'][[idx[i] for i in ids]],y,text,masks,cfg)
            error=float(np.max(np.abs(risk-fit['risk']))); assert error<2e-12
            worst=(risk-risk[:,[0]]).max(0); eligible=worst<=1e-8
            chosen=np.flatnonzero(eligible & (risk[0]<=risk[0,eligible].min()+1e-8))[0]
            assert cfg.lambdas[chosen]==fit['lambda']; audit.append(error)
            p,u,r=centers(x,y,text,masks,cfg)
            def logits(mu):
                return -((qx*qx).sum(1)[:,None]+(mu*mu).sum(1)-2*qx@mu.T)/cfg.tau
            base=logits(p).astype('float32').argmax(1)
            for j,lam in enumerate(cfg.lambdas):
                mu=p+lam*u; z=logits(mu).astype('float32'); pred=z.argmax(1); ok=pred==labels; vis_ok=base==labels
                if j==chosen:
                    with np.load(cell/'predictions.npz') as stored:
                        np.testing.assert_array_equal(qids,stored['ids']); np.testing.assert_allclose(z,stored['logits'],atol=4e-6,rtol=0)
                margin,stable=core.stability(qx,p,lam*r)
                assert not np.any(stable & (base!=pred))
                harm=vis_ok & ~ok; rescue=~vis_ok & ok
                result.append(dict(seed=seed,condition=cond,lambda_value=lam,selected=j==chosen,admitted=bool(eligible[j]),full_brier_delta=risk[0,j]-risk[0,0],worst_brier_delta=worst[j],accuracy=ok.mean(),accuracy_minus_visual=ok.mean()-vis_ok.mean(),harm=harm.mean(),rescue=rescue.mean(),stable_fraction=stable.mean(),mean_move=np.linalg.norm(lam*u,axis=1).mean()))
                for k in range(4):
                    group=np.digitize(margin,[.02,.08,.2])==k
                    grouped.append(dict(seed=seed,condition=cond,lambda_value=lam,margin_bin=k,count=group.sum(),changed=(pred[group]!=base[group]).sum(),certified=stable[group].sum(),harmed=harm[group].sum(),rescued=rescue[group].sum()))
                if lam==.5:
                    for flag,subset in [('harm',harm),('rescue',rescue)]:
                        transitions,counts=np.unique(np.stack([labels[subset],base[subset],pred[subset]],axis=1),axis=0,return_counts=True)
                        for (truth_i,base_i,pred_i),count in zip(transitions,counts):
                            confusion.append(dict(condition=cond,seed=seed,lambda_value=lam,transition=flag,true=classes[truth_i],visual=classes[base_i],corrected=classes[pred_i],images=count))
        print('checked candidate curves for seed',seed,flush=True)
    for name,rows in [('candidate-outcomes',result),('candidate-margins',grouped),('candidate-transitions',confusion)]:
        with (ROOT/'reports'/f'{name}.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    data.save_json(ROOT/'reports'/'candidate-audit.json',dict(episodes=len(audit),folds=len(audit)*4*45*4,checked_risk_entries=len(audit)*4*5,max_abs_risk_error=max(audit),candidate_rows=len(result),fixed_margin_edges=[.02,.08,.2]))

if __name__=='__main__':main()
