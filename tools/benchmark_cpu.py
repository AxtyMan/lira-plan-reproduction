"""Single-thread-group, all-CPU head timings on one preregistered episode (includes search)."""
from pathlib import Path
from time import perf_counter
import csv
import os
import sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='2'
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lira_rs import core,data,pipeline
from lira_rs.cli import config,read_images,read_text
from lira_rs.projection import fit_projection
from check_encoder import peak_working_set


def main():
    torch.set_num_threads(2)
    cfg,head=config(ROOT/'configs/main.yaml'); ep=data.load_json(ROOT/'results/replay-primary/episodes/resisc45/A-4-101.json')
    _,manifest,cache,idx,im=read_images('resisc45'); text,_=read_text('resisc45','normal',7701,im)
    support=ep['roles']['support']; si=[idx[i] for i in support]; y=list(support.values()); mask=text['mask']; t=text['t']
    qi=[idx[r['id']] for r in manifest['images'] if r['role']=='query']; rows=[]
    for meth in cfg['primary_methods']:
        x=cache['pen' if meth=='prolip' else 'z']; z=x[qi]; xs=x[si]
        def train():
            if meth=='prolip':return fit_projection(xs,y,t,mask,cache['projection'],seed=191,steps=300,device='cpu')
            return core.fit(xs,y,t,head,ids=list(support),mask=mask,method=meth)
        state=train(); core.score(z,state)
        for rep in range(3):
            start=perf_counter();state=train();fit_sec=perf_counter()-start
            start=perf_counter();logits=core.score(z,state);query_sec=perf_counter()-start
            assert logits.shape==(len(qi),len(t)) and np.isfinite(logits).all()
            rows.append(dict(method=meth,dataset='resisc45',K=4,support_seed=101,text='normal',device='cpu',blas_threads=2,warmups=1,repeat=rep+1,support_n=len(y),query_n=len(qi),fit_seconds=fit_sec,query_seconds=query_sec,process_peak_working_set_bytes=peak_working_set()))
        print(meth,'complete',flush=True)
    with (ROOT/'reports'/'efficiency-same-cpu.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
