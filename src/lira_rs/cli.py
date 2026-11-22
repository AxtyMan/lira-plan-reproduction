"""Commands for reproducible stages, audit, and experimental matrices."""
import argparse
from dataclasses import replace
from pathlib import Path
import traceback
import numpy as np
import yaml
from . import core, data, pipeline

ROOT=Path(__file__).resolve().parents[2]


def config(path):
    p=Path(path); y=yaml.safe_load(p.read_text(encoding='utf-8-sig')); h=y['head'].copy(); h['lambdas']=tuple(h['lambdas']); return y,core.Config(**h)


def extract(args,cfg):
    from .features import Encoder, extract_image, atomic_npz
    m,x=pipeline.resolve_paths(ROOT,args.dataset); man=data.load_json(m); e=Encoder(ROOT/'data/weights/clip',args.device)
    if not args.text_only:
        extract_image(e,m,x,args.batch)
        if args.native_val: extract_image(e,m,ROOT/'data/cache'/f'{args.dataset}-native-val.npz',args.batch,native_validation=True)
    dic=data.load_json(ROOT/f'prompts/{args.dataset}.json'); tl=[(c,cfg['text_seeds'][0]) for c in cfg['conditions']]+[('wrong30',i) for i in cfg['text_seeds'][1:]]
    for condition,seed in tl:
        p,z=data.perturb(dic['classes'],man['classes'],condition,seed); t,mask=e.texts(p); target=text_path(args.dataset,condition,seed)
        atomic_npz(target,t=t,mask=mask,classes=man['classes'],weight_sha=e.weight_sha,prompt_hash=z['prompt_hash'])
        data.save_json(target.with_suffix('.json'),{**z,'prompts':p,'dictionary_version':dic['version'],'dictionary_sha256':data.filehash(ROOT/f'prompts/{args.dataset}.json')})


def text_path(dataset,condition,seed): return ROOT/'data/cache'/f'{dataset}-text-{condition}-{seed}.npz'


def read_images(dataset):
    p,x=pipeline.resolve_paths(ROOT,dataset); m=data.load_json(p); a=pipeline.load_npz(x)
    if set(a['ids'])!=set(r['id'] for r in m['images']) or str(a['split_hash'])!=m['split_hash']:
        raise ValueError('image cache IDs or split hash mismatch')
    am=data.load_json(x.with_suffix('.json')); return p,m,a,{i:j for j,i in enumerate(a['ids'])},am


def read_text(dataset,c,seed,am):
    t=text_path(dataset,c,seed); x=pipeline.load_npz(t); meta=data.load_json(t.with_suffix('.json'))
    dic=data.load_json(ROOT/f'prompts/{dataset}.json'); _,want=data.perturb(dic['classes'],x['classes'].tolist(),c,seed)
    if str(x['weight_sha'])!=am['weight_sha256'] or str(x['prompt_hash'])!=want['prompt_hash'] or str(x['prompt_hash'])!=meta['prompt_hash']:
        raise ValueError('image/text backbone or prompt hash mismatch')
    return x,meta


def configured_plan(a,y,cfg):
    ts=y['text_seeds'][0]; methods=y['primary_methods']; combos=[]
    def add(pr,c,seed,m,variant='main',cc=cfg):
        combos.append((pr,c,seed,m,variant,cc))
    if a.profile=='main':
        for c in ('normal','wrong30'):
            for m in methods: add('A',c,ts,m)
    elif a.profile=='full':
        for c in y['conditions']:
            for m in methods: add('A',c,ts,m)
        for pr in ('B','B-temp'):
            for m in methods: add(pr,'normal',ts,m)
        for c in ('normal','wrong30'):
            for m in y['ablations']: add('A',c,ts,m,'ablation')
        if a.K==4 and a.dataset==y['main_comparison']['dataset']:
            for seed in y['text_seeds'][1:]:
                for m in ('visual','timo-proto','lira'): add('A','wrong30',seed,m)
            sens=y['sensitivity']
            for c in ('normal','wrong30'):
                for param in ('eta','gamma'):
                    for v in sens[param]:
                        if v!=getattr(cfg,param): add('A',c,ts,'lira',f'{param}-{v}',replace(cfg,**{param:v}))
                for param in ('coarse_lambdas','dense_lambdas'):
                    add('A',c,ts,'lira',param,replace(cfg,lambdas=tuple(sens[param])))
    elif a.profile=='ablations':
        for c in ('normal','wrong30'):
            for m in ['visual','lira','timo-proto',*y['ablations']]: add('A',c,ts,m,'main' if m in {'lira','visual','timo-proto'} else 'ablation')
    else:
        for m in a.methods or methods: add(a.protocol,a.condition,a.text_seed,m)
    if a.methods and a.profile!='single': combos=[q for q in combos if q[3] in a.methods or q[3]=='visual']
    if a.conditions: combos=[q for q in combos if q[1] in a.conditions]
    return combos


def matrix(a,y,cfg):
    ds=a.datasets or y['datasets']; shots=a.shots or y['shots']; seeds=a.seeds or y['support_seeds']; base=Path(a.out)
    base.mkdir(parents=True,exist_ok=True); errors=[]; total=0
    data.save_json(base/'frozen-config.json',y)
    for dataset in ds:
        p,m,x,idx,am=read_images(dataset); a.dataset=dataset; texts={}; variants={}
        for k in shots:
            a.K=k; combos=configured_plan(a,y,cfg)
            # Always score the paired visual control, even for an isolated method run.
            combos=list(dict.fromkeys([(pr,c,ts,'visual','main',cfg) for pr,c,ts,_,_,_ in combos]+combos))
            for seed in seeds:
                print(f'{dataset} K={k} support_seed={seed} ({len(combos)} cells)',flush=True)
                eps={}
                for pr,c,ts,method,variant,cc in combos:
                    try:
                        if pr not in eps:
                            ep=base/'episodes'/dataset/f'{pr}-{k}-{seed}.json'; pipeline.frozen_episode(p,k,seed,pr,ep); eps[pr]=data.load_json(ep); pipeline.validate_episode_ids(eps[pr],m)
                        if (c,ts) not in texts: texts[(c,ts)]=read_text(dataset,c,ts,am)
                        t,tm=texts[(c,ts)]; e=eps[pr]; key=base/'cells'/dataset/pr/f'K{k:02}'/f'S{seed}'/f'{c}-{ts}'
                        os=y['optimization_seeds'] if method=='prolip' else [None]
                        for op in os:
                            out=key/(method if variant=='main' else f'{method}--{variant}')
                            if op is not None: out=out/f'optimization-{op}'
                            if a.resume and (out/'metrics.json').exists():
                                prev=data.load_json(out/'fit.json')
                                planned={'config_hash':data.digest(vars(cc)),'episode_hash':data.digest(e),'prompt_hash':str(t['prompt_hash']),'cache_key':str(x['cache_key']),'optimization_seed':op}
                                if all(prev.get(k)==v for k,v in planned.items()) and (op is None or prev['steps']==a.projection_steps):
                                    total+=1; continue
                            st,meta=pipeline.fit_artifact(e,x,t,method,cc,out,idx,opt_seed=op,steps=a.projection_steps,device=a.device)
                            meta.update(condition=c,text_seed=ts,variant_name=variant,prompt_version=tm.get('dictionary_version'),text_perturbation=tm.get('changes')); data.save_json(out/'fit.json',meta)
                            pipeline.score_artifact(m,x,idx,st,meta,out)
                            mr=pipeline.evaluate_artifact(p,out,key/'visual'); total+=1
                            if method=='lira': print(f"  {pr} {c}/{ts} {variant}: Acc {mr['accuracy']:.4f} lambda={meta['lambda']}",flush=True)
                    except Exception as exc:
                        errors.append(dict(dataset=dataset,K=k,seed=seed,protocol=pr,condition=c,text_seed=ts,method=method,variant=variant,error=str(exc),traceback=traceback.format_exc())); data.save_json(base/'failures.json',errors)
                        if not a.keep_going: raise
    data.save_json(base/'run-status.json',dict(completed_fits=total,failed_cells=len(errors),datasets=ds,shots=shots,seeds=seeds,profile=a.profile)); pipeline.aggregate(base)
    if errors: raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(prog='run.py'); p.add_argument('--config',default=ROOT/'configs/main.yaml'); s=p.add_subparsers(dest='command',required=True)
    def cmd(name,ds=True):
        q=s.add_parser(name)
        if ds: q.add_argument('dataset',choices=data.DATASETS)
        return q
    q=cmd('prepare'); q.add_argument('archive',type=Path); q.add_argument('--public-split',type=Path); q.add_argument('--source');
    q=cmd('extract'); q.add_argument('--batch',type=int,default=80); q.add_argument('--device',default='auto'); q.add_argument('--text-only',action='store_true'); q.add_argument('--native-val',action='store_true');
    def episode_args(q): q.add_argument('--K',type=int,default=4); q.add_argument('--seed',type=int,default=101); q.add_argument('--protocol',choices=['A','B','B-temp'],default='A')
    q=cmd('episode'); episode_args(q); q.add_argument('--out',type=Path,required=True)
    q=cmd('fit'); q.add_argument('--episode',type=Path,required=True); q.add_argument('--method',choices=core.METHODS,default='lira'); q.add_argument('--condition',default='normal'); q.add_argument('--text-seed',type=int,default=7701); q.add_argument('--out',type=Path,required=True); q.add_argument('--device',default='cpu')
    q=cmd('score'); q.add_argument('--out',type=Path,required=True)
    q=cmd('evaluate'); q.add_argument('--out',type=Path,required=True); q.add_argument('--visual',type=Path,required=True)
    q=cmd('native',False); q.add_argument('method',choices=['timo','prolip']); q.add_argument('--K',type=int,default=4); q.add_argument('--seed',type=int,default=1); q.add_argument('--out',type=Path,required=True)
    q=cmd('export',False); q.add_argument('cell',type=Path); q.add_argument('csv',type=Path); q.add_argument('--evaluated',action='store_true')
    q=cmd('aggregate',False); q.add_argument('out',type=Path)
    q=cmd('matrix',False); q.add_argument('--out',required=True,type=Path); q.add_argument('--datasets',nargs='+',choices=data.DATASETS); q.add_argument('--shots',nargs='+',type=int); q.add_argument('--seeds',nargs='+',type=int); q.add_argument('--profile',choices=['main','full','single','ablations'],default='main'); q.add_argument('--methods',nargs='+',choices=core.METHODS); q.add_argument('--conditions',nargs='+'); q.add_argument('--condition',default='normal'); q.add_argument('--text-seed',type=int,default=7701); q.add_argument('--protocol',choices=['A','B','B-temp'],default='A'); q.add_argument('--resume',action='store_true'); q.add_argument('--keep-going',action='store_true'); q.add_argument('--device',default='cpu'); q.add_argument('--projection-steps',type=int,default=300)
    a=p.parse_args(); y,cfg=config(a.config)
    if a.command=='prepare':
        dic=data.load_json(ROOT/f'prompts/{a.dataset}.json'); public=a.public_split
        if a.dataset=='eurosat' and not public:
            alt=ROOT/'data/upstream/coop-split-usable.json'; public=alt if alt.exists() else None
        folder=a.dataset+'-coop' if public and a.dataset=='eurosat' else a.dataset
        print(data.prepare(a.dataset,a.archive,dic['classes'],ROOT/'data/manifests'/folder,split_seed=y['split_seed'],public_split=public,source=a.source))
    elif a.command=='extract': extract(a,y)
    elif a.command=='native':
        from .native import native
        native(ROOT,a.method,a.K,a.seed,a.out)
    elif a.command=='export': pipeline.export_rows(a.cell,a.csv,evaluated=a.evaluated)
    elif a.command=='aggregate': print(f'{len(pipeline.aggregate(a.out))} summary rows')
    elif a.command=='matrix':
        from filelock import FileLock
        a.out.mkdir(parents=True,exist_ok=True)
        with FileLock(str(a.out/'.matrix.lock')):
            matrix(a,y,cfg)
    elif a.command=='episode': pipeline.frozen_episode(pipeline.resolve_paths(ROOT,a.dataset)[0],a.K,a.seed,a.protocol,a.out)
    elif a.command=='evaluate': print(pipeline.evaluate_artifact(pipeline.resolve_paths(ROOT,a.dataset)[0],a.out,a.visual))
    elif a.command in {'fit','score'}:
        p,m,x,idx,am=read_images(a.dataset)
        if a.command=='fit':
            e=data.load_json(a.episode); pipeline.validate_episode_ids(e,m); t,tm=read_text(a.dataset,a.condition,a.text_seed,am)
            st,meta=pipeline.fit_artifact(e,x,t,a.method,cfg,a.out,idx,device=a.device)
            meta.update(condition=a.condition,text_seed=a.text_seed,variant_name='main',prompt_version=tm.get('dictionary_version')); data.save_json(a.out/'fit.json',meta)
        else:
            meta=data.load_json(a.out/'fit.json'); st=core.State(meta['head'],pipeline.load_npz(a.out/'state.npz'),meta)
            pipeline.score_artifact(m,x,idx,st,meta,a.out)
