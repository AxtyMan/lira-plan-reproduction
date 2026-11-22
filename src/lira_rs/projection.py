"""ProLIP shared-protocol variant (single deterministic view, no external validation)."""
import numpy as np
import torch
import torch.nn.functional as F
from .baselines import text_average
from .core import State


def fit_projection(pre,y,t,mask,projection,*,seed=101,steps=300,lr=1e-4,reg=.1,device='cpu'):
    if np.asarray(pre).ndim!=2 or len(pre)!=len(y) or steps<1 or lr<=0 or reg<0:
        raise ValueError('invalid projection experiment')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        z=torch.as_tensor(pre,dtype=torch.float32,device=device)
        y=torch.as_tensor(y,dtype=torch.long,device=device)
        initial=torch.as_tensor(projection,dtype=torch.float32,device=device)
        w=torch.as_tensor(100*text_average(t,mask).T,dtype=torch.float32,device=device)
        p=torch.nn.Parameter(initial.clone()); optim=torch.optim.Adam([p],lr=lr,eps=1e-4)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,steps)
        # Full-batch adaptation. Seed determines row order only; single view has no stochastic augmentations.
        rng=np.random.default_rng(seed); ix=torch.as_tensor(rng.permutation(len(z)),device=device)
        for k in range(steps):
            v=F.normalize(z[ix]@p,dim=-1); ce=F.cross_entropy(v@w,y[ix]); penalty=(p-initial).square().sum()
            loss=ce+reg*penalty; optim.zero_grad(); loss.backward(); optim.step(); sched.step()
        return State('projection',dict(projection=p.detach().cpu().numpy(),weight=w.cpu().numpy()),
                     dict(method='prolip',variant='shared_protocol_single_view',optimization_seed=seed,steps=steps,
                          trainable_parameters=p.numel(),lr=lr,regularization=reg,views=1,final_loss=float(loss.detach().cpu())))
