"""File-level boundary: sealed targets never enter fit or score."""
import csv
import gzip
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .core import Config, State, fit, score, stability
from .data import digest, episode, filehash, load_json, save_json
from .evaluation import metrics, temperature, paired_interval, holm


def put_npz(path,**values):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); t=path.with_name(path.name+'.tmp')
    with t.open('wb') as f: np.savez_compressed(f,**values)
    t.replace(path)


def load_npz(path):
    with np.load(path,allow_pickle=False) as p: return {k:p[k] for k in p.files}


def resolve_paths(root,dataset):
    folder=dataset+'-coop' if dataset=='eurosat' and (root/'data/manifests/eurosat-coop/images.json').exists() else dataset
    manifest=root/'data/manifests'/folder/'images.json'
    return manifest,root/'data/cache'/f'{folder}-images.npz'


def frozen_episode(manifest,k,seed,protocol,out):
    m=load_json(manifest); raw=load_json(manifest.with_name('candidate.labels.json'))
    if set(x['id'] for x in raw)!=set(r['id'] for r in m['images'] if r['role']=='candidate'):
        raise ValueError('candidate manifest role mismatch')
    e=episode(raw,k,seed,protocol); labelsets=list(e.values())
    if len(set().union(*labelsets))!=sum(map(len,labelsets)): raise ValueError('non-disjoint label roles')
    save_json(out,{'dataset':m['dataset'],'classes':m['classes'],'split_hash':m['split_hash'],'K':k,'support_seed':seed,'protocol':protocol,'budget':sum(map(len,labelsets)),'roles':e})


def validate_episode_ids(episode,manifest):
    if any(episode[k]!=manifest[k] for k in ('dataset','classes','split_hash')):
        raise ValueError('episode does not identify this manifest')
    pool={r['id'] for r in manifest['images'] if r['role']=='candidate'}
    if any(not set(v)<=pool for v in episode['roles'].values()):
        raise ValueError('selected image outside candidate pool')


def gz_json(path,val):
    t=path.with_name(path.name+'.tmp')
    with gzip.open(t,'wt',encoding='utf-8',compresslevel=3) as f: json.dump(val,f,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    t.replace(path)


def fit_artifact(e,x,t,method,cfg,out,idx,*,opt_seed=191,steps=300,device='cpu'):
    out.mkdir(parents=True,exist_ok=True); start=perf_counter(); roles=e['roles']; protocol=e['protocol']; supports=roles['support']; ids=list(supports)
    keys={'A':{'support'},'B':{'support','select','calibrate'},'B-temp':{'support','calibrate'}}
    labelsets=list(roles.values()); c=len(e['classes']); k=e['K']
    if set(roles)!=keys[protocol] or e['split_hash']!=str(x['split_hash']) or len(set().union(*labelsets))!=sum(map(len,labelsets)):
        raise ValueError('protocol role or split mismatch')
    expected_sizes={'support':k,'select':10,'calibrate':10 if protocol=='B' else 20}
    for key,labels in roles.items():
        y=np.asarray(list(labels.values()))
        if y.ndim!=1 or y.dtype.kind not in 'iu' or np.any((y<0)|(y>=c)) or not np.all(np.bincount(y,minlength=c)==expected_sizes[key]):
            raise ValueError(f'wrong class budget: {key}')
    if t['classes'].tolist()!=e['classes']: raise ValueError('prompt class order mismatch')
    ii=[idx[i] for i in ids]; y=list(supports.values()); text=t['t']; mask=t['mask']
    if method=='prolip':
        from .projection import fit_projection
        state=fit_projection(x['pen'][ii],y,text,mask,x['projection'],seed=opt_seed,steps=steps,device=device)
    elif protocol=='B' and method in {'gda-loo','timo-loo'}:
        from .baselines import replay_selection
        val=roles['select']; jj=[idx[i] for i in val]
        state=replay_selection(x['z'][ii],y,text,mask,cfg,x['z'][jj],list(val.values()),ids,list(val),timo=method=='timo-loo')
    else: state=fit(x['z'][ii],y,text,cfg,ids=ids,mask=mask,method=method)
    sec=perf_counter()-start; T=1.; cal_seconds=0.
    if protocol!='A':
        start=perf_counter(); cal=roles['calibrate']; jj=[idx[i] for i in cal]; f=x['pen'] if state.head=='projection' else x['z']
        T=temperature(score(f[jj],state),list(cal.values())); cal_seconds=perf_counter()-start
    put_npz(out/'state.npz',**state.arrays)
    diag=dict(state.log)
    for field in ('fold_probabilities','fold_brier','fold_text_status','loo_predictions','fold_remaining_ids','visual_centers','anchors'):
        if field in state.log: diag.pop(field)
    gz_json(out/'fit-diagnostics.json.gz',state.log)
    meta={k:v for k,v in e.items() if k!='roles'}
    meta.update(diag,head=state.head,requested_method=method,fit_seconds=sec,calibration_seconds=cal_seconds,temperature=T,
                support_ids=ids,role_sizes={k:len(v) for k,v in roles.items()},prompt_hash=str(t['prompt_hash']),cache_key=str(x['cache_key']),config_hash=digest(vars(cfg)),episode_hash=digest(e))
    save_json(out/'fit.json',meta)
    return state,meta


def score_artifact(m,x,idx,state,meta,out):
    if m['split_hash']!=meta['split_hash'] or str(x['cache_key'])!=meta['cache_key']: raise ValueError('fitted provenance mismatch')
    ids=[r['id'] for r in m['images'] if r['role']=='query']; z=x['z'][[idx[i] for i in ids]]; start=perf_counter()
    mat=x['pen'][[idx[i] for i in ids]] if state.head=='projection' else z
    logits=score(mat,state); elapsed=perf_counter()-start; extra={}
    if state.head=='distance':
        v=state.arrays; margin,cert=stability(z,v['p'],v['delta_bound']); extra.update(margin=margin,stable=cert)
    put_npz(out/'predictions.npz',ids=ids,logits=logits.astype(np.float32),**extra)
    save_json(out/'score.json',{'query_seconds':elapsed,'query_n':len(ids),'scores_sha256':filehash(out/'predictions.npz')})


def evaluate_artifact(manifest,out,visual):
    meta=load_json(out/'fit.json'); score_meta=load_json(out/'score.json'); q=load_npz(out/'predictions.npz'); b=load_npz(visual/'predictions.npz'); bm=load_json(visual/'fit.json')
    if filehash(out/'predictions.npz')!=score_meta['scores_sha256'] or any(meta[k]!=bm[k] for k in ('dataset','split_hash','K','support_seed','protocol','episode_hash','cache_key')) or not np.array_equal(q['ids'],b['ids']):
        raise ValueError('unpaired evaluation inputs or modified scores')
    # The only read of query.labels.json in code after split preparation.
    labels={r['id']:r['label'] for r in load_json(manifest.with_name('query.labels.json'))}
    if set(labels)!=set(q['ids'].tolist()): raise ValueError('missing or extraneous query labels')
    y=np.array([labels[i] for i in q['ids']]); z=q['logits']; mr=metrics(z,y,ids=q['ids'],base=b['logits']);
    opt=metrics(z/meta['temperature'],y,ids=q['ids'],base=b['logits']) if meta['protocol']!='A' else None
    if opt is not None and opt['accuracy']!=mr['accuracy']: raise AssertionError('temperature must keep ranking')
    pred=z.argmax(1); vp=b['logits'].argmax(1); extra={}
    if 'stable' in q:
        stable=q['stable']; margin=q['margin']; changed=pred!=vp
        if np.any(stable & changed): raise AssertionError('certified decision changed')
        extra={'certified_fraction':float(stable.mean()),'margin_groups':[]}
        bins=np.digitize(margin,[.02,.08,.2]);
        for k in range(4):
            mask=bins==k
            extra['margin_groups'].append({'bucket':k,'n':int(mask.sum()),'changed':float(changed[mask].mean()) if mask.any() else None,
                'harm':float(np.mean((vp[mask]==y[mask]) & (pred[mask]!=y[mask]))) if mask.any() else None})
    put_npz(out/'evaluated.npz',**q,labels=y,predicted=pred,visual_predicted=vp,harm=(pred!=y)&(vp==y),rescue=(pred==y)&(vp!=y))
    save_json(out/'metrics.json',dict(meta=meta,score=score_meta,raw=mr,calibrated=opt,diagnostics=extra))
    return mr


def aggregate(run_dir):
    groups={}; paired={}; seen=set(); keys=('dataset','protocol','K','condition','text_seed','requested_method','config_hash','prompt_hash','split_hash')
    preferred={'eurosat-main':0,'main':0,'resisc45-primary':0,'resisc45-other':0,'resisc45-k16':-1}
    root = Path(run_dir)
    paths = [path for path in root.rglob('metrics.json') if 'cells' in path.relative_to(root).parts]
    def run_priority(path):
        batch = path.relative_to(root).parts[0]
        return (-2 if batch.startswith('replay-') else preferred.get(batch, 1), str(path))
    paths.sort(key=run_priority)
    for p in paths:
        m=load_json(p); a=m['meta']; key=tuple(a.get(k) for k in keys); variant=a.get('variant_name','main'); seed=a['support_seed']
        logical=(key,variant,seed,a.get('optimization_seed'),a.get('episode_hash'),a.get('cache_key'))
        if logical in seen: continue
        # Keep scalar sufficient statistics. Fold logs and C-by-C matrices stay on disk.
        scalar_keys=('accuracy','macro_f1','brier','nll','ece','aurc','harm','rescue')
        compact={'raw':{k:m['raw'][k] for k in scalar_keys},
                 'calibrated':{k:m['calibrated'][k] for k in ('brier','nll','ece')} if m['calibrated'] else None,
                 'meta':{'lambda':a.get('lambda'),'fit_seconds':a['fit_seconds']},
                 'score':{'query_seconds':m['score']['query_seconds']}}
        seen.add(logical); groups.setdefault((key,variant),{}).setdefault(seed,[]).append(compact)
    rows=[]
    for (key,variant),by_seed in sorted(groups.items(),key=str):
        row=dict(zip(keys,key)); row.update(variant=variant,seed_count=len(by_seed),fit_count=sum(map(len,by_seed.values())))
        for metric in ('accuracy','macro_f1','brier','nll','ece','aurc','harm','rescue'):
            vals=[np.mean([m['raw'][metric] for m in g]) for g in by_seed.values()]; row[metric]=float(np.mean(vals)); row[metric+'_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        for m in ('brier','nll','ece'):
            vals=[np.mean([v['calibrated'][m] for v in g]) for g in by_seed.values() if all(v['calibrated'] for v in g)]; row['cal_'+m]=float(np.mean(vals)) if vals else None
        l=[x['meta'].get('lambda') for v in by_seed.values() for x in v if x['meta'].get('lambda') is not None]
        row['fallback_rate']=float(np.mean(np.array(l)==0)) if l and key[5] not in {'visual'} else None
        row['fit_seconds_mean']=float(np.mean([x['meta']['fit_seconds'] for v in by_seed.values() for x in v])); row['query_seconds_mean']=float(np.mean([x['score']['query_seconds'] for v in by_seed.values() for x in v])); rows.append(row)
        if variant=='main' and key[1]=='A': paired.setdefault((key[0],key[2],key[3],key[4],key[6],key[8]),{})[key[5]]=by_seed
    comparisons=[]
    for cell,g in sorted(paired.items(),key=str):
        if 'lira' not in g: continue
        for ref in ('timo-proto','visual','timo-loo'):
            if ref not in g: continue
            seeds=sorted(set(g['lira'])&set(g[ref]));
            if not seeds: continue
            for metric in ('accuracy','harm','rescue'):
                ds=[float(np.mean([r['raw'][metric] for r in g['lira'][i]])-np.mean([r['raw'][metric] for r in g[ref][i]])) for i in seeds]
                row=dict(dataset=cell[0],K=cell[1],condition=cell[2],text_seed=cell[3],reference=ref,metric=metric,split_hash=cell[5],seeds=seeds,**paired_interval(ds))
                if len(ds)>1:
                    from itertools import product
                    d=np.asarray(ds); flips=np.array(list(product((-1,1),repeat=len(ds)))); row['p_raw']=float(np.mean(np.abs(flips@d)>=abs(d.sum())-1e-12))
                else: row['p_raw']=None
                comparisons.append(row)
    valid=[r for r in comparisons if r['p_raw'] is not None]; adjusted=holm([r['p_raw'] for r in valid]);
    for r,p in zip(valid,adjusted): r['p_holm_global']=float(p)
    Path(run_dir).mkdir(parents=True,exist_ok=True)
    if rows:
        with (Path(run_dir)/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    save_json(Path(run_dir)/'paired.json',comparisons)
    return rows


def export_rows(cell,out,*,evaluated=False):
    """Export portable per-image CSV (no labels unless explicitly exporting evaluator output)."""
    cell=Path(cell); out=Path(out); a=load_json(cell/'fit.json'); s=load_json(cell/'score.json'); q=load_npz(cell/('evaluated.npz' if evaluated else 'predictions.npz'))
    out.parent.mkdir(parents=True,exist_ok=True)
    lead=['dataset','split_hash','support_seed','K','protocol','requested_method','condition','prompt_version','prompt_hash']; c=a['classes']
    rest=['image_id','predicted_class','lambda','displacement','fallback_reason','fit_seconds','query_seconds']+['logit_'+n for n in c]
    if evaluated: rest+=['query_label','harm','rescue']
    with out.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(lead+rest)
        for i,k in enumerate(q['ids']):
            v=q['logits'][i]; row=[a.get(n) for n in lead]+[k,c[np.argmax(v)],a.get('lambda'),json.dumps(a.get('displacement')),a.get('fallback_reason'),a['fit_seconds'],s['query_seconds']]+v.tolist()
            if evaluated: row+=[c[int(q['labels'][i])],int(q['harm'][i]),int(q['rescue'][i])]
            w.writerow(row)
