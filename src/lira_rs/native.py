"""Local native baseline checks using hash-pinned upstream implementation files."""
from collections import defaultdict
import io
from pathlib import Path
import random
from time import perf_counter
from zipfile import ZipFile
import sys
import numpy as np
import torch
from torch.nn.functional import normalize
from PIL import Image
from torchvision import transforms as T
from .data import filehash, load_json, save_json
from .pipeline import put_npz
from .evaluation import metrics


FILES={
 'timo':('TIMO.zip','TIMO-main','8e3cce70c9f55d6dd57ca6372ae6e7afa8183cc57435ccaa5d57590296273b90'),
 'prolip':('ProLIP.zip','ProLIP-master','588a3d7cbcae3402101b84c91adc2128a9e581776478f7cecd6e04ffef47b0f6'),
}
RN_SHA='afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762'


def native(root,name,k,seed,out,batch=64):
    start=perf_counter(); out.mkdir(parents=True,exist_ok=True); repo=root/'data/upstream'; rawpath,top,sha=FILES[name]; a=repo/rawpath
    if filehash(a)!=sha: raise ValueError(f'{name} source archive checksum mismatch')
    native_root=repo/f'local-{name}'
    with ZipFile(a) as z:
        # Avoid ZIP extraction of any path chosen by an archive member.
        paths=['clip/__init__.py','clip/clip.py','clip/model.py','clip/simple_tokenizer.py','clip/bpe_simple_vocab_16e6.txt.gz']
        if name=='timo': paths+=['models.py','utils.py','Prompt_CuPL/eurosat.json']
        for path in paths:
            f=native_root/path; f.parent.mkdir(parents=True,exist_ok=True); f.write_bytes(z.read(f'{top}/{path}'))
    if not torch.cuda.is_available(): raise RuntimeError('native author replay needs CUDA')
    weight=root/'data/weights/RN50.pt'
    if filehash(weight)!=RN_SHA: raise ValueError('RN50 checkpoint SHA-256 mismatch')
    sys.path.insert(0,str(native_root)); import clip
    # Use a Python stream: TorchScript's Windows C++ filename loader cannot open some Unicode paths.
    from clip.clip import _transform
    from clip.model import build_model
    with weight.open('rb') as f: state=torch.jit.load(f,map_location='cpu').state_dict()
    model=build_model(state).cuda(); preprocess=_transform(model.visual.input_resolution)
    model.eval().requires_grad_(False)
    s=load_json(repo/'coop-split-usable.json'); archive=root/'data/downloads/EuroSAT-torchgeo.zip'
    if filehash(archive)!='8ebea626349354c5328b142b96d0430e647051f26efc2dc974c843f25ecf70bd': raise ValueError('native EuroSAT archive changed')
    classnames=[None]*10; by=defaultdict(list)
    for r in s['train']: classnames[r[1]]=r[2]; by[r[1]].append(r)
    random.seed(seed); torch.manual_seed(seed); rows=[r for rr in by.values() for r in random.sample(rr,k)]; y=torch.tensor([r[1] for r in rows],device='cuda')
    crop=T.Compose([T.RandomResizedCrop(224,scale=(.5,1),interpolation=T.InterpolationMode.BICUBIC),T.RandomHorizontalFlip(p=.5),T.ToTensor(),T.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    z=ZipFile(archive)
    def image(r,tf):
        with Image.open(io.BytesIO(z.read('2750/'+r[0]))) as im: return tf(im.convert('RGB'))
    @torch.no_grad()
    def features(rr,tf):
        chunks=[]
        for j in range(0,len(rr),batch):
            images=torch.stack([image(r,tf) for r in rr[j:j+batch]]).cuda(); chunks.append(model.encode_image(images))
        f=torch.cat(chunks); return normalize(f,dim=1) if name=='timo' else f
    @torch.no_grad()
    def txt(texts):
        t=model.encode_text(clip.tokenize(texts,truncate=True).cuda()); return normalize(t[-1] if isinstance(t,tuple) else t,dim=1)
    normal=[f'a centered satellite photo of {c}.' for c in classnames]; query=s['test'];
    if name=='timo':
        import models
        cp=load_json(native_root/'Prompt_CuPL/eurosat.json'); per_class=[txt([p]+cp[c]) for p,c in zip(normal,classnames)]; shortest=min(map(len,per_class)); t=torch.stack([p[:shortest] for p in per_class])
        views=torch.stack([features(rows,crop) for _ in range(10)])
        v0=normalize(views.float().mean(0),dim=1); mu=normalize(torch.stack([v0[y==i].mean(0) for i in range(len(t))]),dim=1)
        igt,raw=models.image_guide_text({'dataset':'eurosat'},t,mu,return_matching=True)
        val=features(s['val'],preprocess); vy=torch.tensor([r[1] for r in s['val']],device='cuda'); qs=features(query,preprocess)
        raw_weight,ix=torch.sort(raw,dim=1,descending=True); t=torch.gather(t,1,ix[...,None].expand_as(t)).float(); pseudo=(t*raw_weight[...,None]).flatten(0,1)
        cy=torch.cat([torch.arange(len(t),device='cuda').repeat_interleave(t.shape[1]),y.repeat(10)])
        v=torch.cat([pseudo,views.flatten(0,1).float()]); alpha,W,b,va=models.GDA(v,cy,igt.T,val,vy,alpha_shift=True)
        logits=(alpha*qs.float()@igt.T.float()+qs.float()@W+b).detach().cpu().numpy()
        more=dict(alpha=alpha,val_accuracy_pct=va,prompts_per_class=t.shape[1],views=10,validation_labels_used=len(vy),upstream_functions='image_guide_text and GDA; non-S TGI tensor construction from TIMO branch')
    else:
        w=txt(normal).float().T; n=300; lr=1e-4; reg=.1; cproj=model.visual.attnpool.c_proj; w0=cproj.weight.detach().float(); bias=cproj.bias.detach().float()
        pr=torch.nn.Parameter(w0.clone()); opt=torch.optim.Adam([pr],lr=lr,eps=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,n)
        # Stream one deterministic view per step, rather than author disk caching of all views.
        for j in range(n):
            f=features(rows,crop).float(); logits=100*normalize(f@pr.T+bias,dim=1)@w
            loss=torch.nn.functional.cross_entropy(logits,y)+reg*(pr-w0).square().sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if j%50==0: print('ProLIP epoch',j,'loss',float(loss.detach()),flush=True)
        q=features(query,preprocess).float(); logits=(100*normalize(q@pr.detach().T+bias,dim=1)@w).cpu().numpy()
        more=dict(steps=n,lr=lr,regularization=reg,views=n,validation_labels_used=0,trainable_parameters=pr.numel(),final_loss=float(loss.detach().cpu()),upstream_functions='ProLIP CLIP pre-projection encoder; author loss, optimizer and scheduler replay')
    z.close()
    put_npz(out/'predictions.npz',ids=[r[0] for r in query],logits=logits)
    qy=np.array([r[1] for r in query]); put_npz(out/'evaluated.npz',ids=[r[0] for r in query],logits=logits,labels=qy)
    report=dict(method=name,category='author_configuration_replay_not_full_author_suite',dataset='eurosat',split='CoOp native without research dedup',backbone='RN50',support_seed=seed,K=k,support_labels=len(rows),query_n=len(query),checkpoint_sha256=RN_SHA,source_sha256=sha,seconds=perf_counter()-start,source_details=more,metrics=metrics(logits,qy))
    save_json(out/'native-report.json',report); print(f"{name} EuroSAT RN50 K={k}: {report['metrics']['accuracy']:.5f}",flush=True)
