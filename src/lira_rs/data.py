"""Image manifest, source validation, role separation, and predeclared prompt perturbations."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from time import perf_counter
from zipfile import ZipFile
import numpy as np
from PIL import Image, ImageOps

DATASETS={
 'eurosat': {'n':27000,'c':10,'archive':'EuroSAT.zip','url':'https://madm.dfki.de/files/sentinel/EuroSAT.zip'},
 'ucmerced': {'n':2100,'c':21,'per_class':100,'archive':'UCMerced_LandUse.zip','md5':'5b7ec56793786b6dc8a908e8854ac0e4','url':'https://hf.co/datasets/torchgeo/ucmerced/resolve/7c5ef3454d9b1cccfa7ccde0c01fc8f00a45909a/UCMerced_LandUse.zip'},
 'resisc45': {'n':31500,'c':45,'per_class':700,'archive':'NWPU-RESISC45.zip','md5':'75206b2e16446591afa88e2628744886','url':'https://hf.co/datasets/torchgeo/resisc45/resolve/a826b44d938a883185f11ebe3d512d38b464312f/NWPU-RESISC45.zip'},
}
EXT={'.jpeg','.jpg','.png','.tif','.tiff'}


def digest(x):
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def filehash(path,algorithm='sha256'):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,algorithm).hexdigest()


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def save_json(path,x):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp'); t.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8'); t.replace(p)


def thumb(img):
    g=np.asarray(ImageOps.grayscale(img).resize((9,8),Image.Resampling.LANCZOS));
    h=int.from_bytes(np.packbits((g[:,1:]>g[:,:-1]).ravel()).tobytes(),'big');
    pix=np.asarray(img.resize((12,12),Image.Resampling.BILINEAR),dtype=np.int16)
    return h,pix


def match_coop(p,labels):
    raw=load_json(p); lookup={}; remaining=[]
    for key,role in [('train','candidate'),('val','sealed_val'),('test','query')]:
        for path,_,_ in raw[key]:
            v=PurePosixPath(path); k=(v.parent.name.lower(),v.name.lower());
            if k in lookup or k[0] not in labels: raise ValueError(f'bad public split row: {k}')
            lookup[k]=role; remaining.append(k)
    return lookup,set(remaining)


def prepare(dataset,archive,prompts,output,*,split_seed=271828,public_split=None,source=None):
    spec=DATASETS[dataset]; archive=Path(archive).resolve(); output=Path(output); started=perf_counter()
    if spec.get('md5') and filehash(archive,'md5')!=spec['md5']:
        raise ValueError(f'{dataset} archive does not match TorchGeo MD5')
    labels={v:i for i,v in enumerate(sorted(prompts))}
    if len(labels)!=spec['c']: raise ValueError('incomplete class dictionary')
    exact={}; excluded=[]; rows=[]; possible={}; thumbs={}; before=np.zeros(len(labels),int)
    public,remaining=match_coop(public_split,labels) if public_split else ({},set())
    # With a public split, a duplicate group is assigned to the highest secrecy role.
    precedence={'candidate':0,'sealed_val':1,'query':2}
    with ZipFile(archive) as z:
        members=sorted(z.namelist()); seen=set()
        for path in members:
            v=PurePosixPath(path); label=v.parent.name.lower()
            if (v.suffix.lower() not in EXT or label not in labels or
                    any(k.startswith('.') for k in v.parts) or '__MACOSX' in v.parts): continue
            if path in seen: raise ValueError(f'duplicate ZIP member: {path}')
            seen.add(path); before[labels[label]]+=1; data=z.read(path); sha=hashlib.sha256(data).hexdigest()
            key=label,v.name.lower(); role=public.get(key)
            if public_split:
                if key not in remaining: raise ValueError(f'image unlisted or repeated in public split: {key}')
                remaining.remove(key)
            reason=None; other=None
            if sha in exact: other=exact[sha]; reason='exact_sha256'
            else:
                with Image.open(io.BytesIO(data)) as im:
                    rgb=im.convert('RGB'); h,p=thumb(rgb)
                keys=[(j,(h>>(13*j))&8191) for j in range(5)]  # distance <=4 implies one of 5 identical bands
                cand=sorted(set(i for k in keys for i in possible.get(k,[])))
                for i in cand:
                    oh,op=thumbs[i]
                    if (h^oh).bit_count()<=4 and np.mean(np.abs(p-op))<=5:
                        other=i; reason='dhash4_rgb_mae5'; break
            if other is not None:
                rep=rows[other]; exact[sha]=other
                if rep['label']!=labels[label]:
                    # Ambiguous labels: remove the entire matched group, not just one copy.
                    rep['ambiguous']=True
                if public_split and precedence[role]>precedence[rep['role']]: rep['role']=role
                excluded.append({'member':path,'representative':rep['member'],'reason':reason,'label_conflict':rep['label']!=labels[label]})
                continue
            i=len(rows); exact[sha]=i; thumbs[i]=(h,p)
            for k in keys: possible.setdefault(k,[]).append(i)
            rows.append(dict(id=sha,member=path,label=labels[label],role=role))
    if remaining or before.sum()!=spec['n'] or (spec.get('per_class') and not np.all(before==spec['per_class'])) or np.any(before==0):
        raise ValueError(f'{dataset}: expected full official class counts, found {before.tolist()}')
    for r in rows:
        if r.get('ambiguous'): excluded.append(dict(member=r['member'],reason='cross_label_duplicate_group'))
    rows=[r for r in rows if not r.get('ambiguous')]; rng=np.random.default_rng(split_seed)
    if not public_split:
        for c in range(len(labels)):
            local=[r for r in rows if r['label']==c]; order=rng.permutation(len(local)); cut=int(.6*len(local))
            for j,i in enumerate(order): local[i]['role']='candidate' if j<cut else 'query'
    sel=[{'id':r['id'],'label':r['label']} for r in rows if r['role']=='candidate']
    q=[{'id':r['id'],'label':r['label']} for r in rows if r['role']=='query']
    val=[{'id':r['id'],'label':r['label']} for r in rows if r['role']=='sealed_val']
    images=[{k:r[k] for k in ('id','member','role')} for r in rows if r['role']!='sealed_val']
    digest_split=digest([(r['id'],r['role'],r['label']) for r in rows]); by_role={k:np.bincount([r['label'] for r in rows if r['role']==k],minlength=len(labels)).tolist() for k in ('candidate','query','sealed_val')}
    save_json(output/'images.json',dict(dataset=dataset,split_hash=digest_split,archive=str(archive),archive_sha256=filehash(archive),source=source or spec['url'],classes=list(labels),images=images))
    save_json(output/'candidate.labels.json',sel); save_json(output/'query.labels.json',q)
    if val:
        save_json(output/'native-val.labels.json',val)
        save_json(output/'native-val.images.json',[{k:r[k] for k in ('id','member','role')} for r in rows if r['role']=='sealed_val'])
    save_json(output/'audit.json',dict(dataset=dataset,split_hash=digest_split,classes=list(labels),scan_count=before.tolist(),by_role=by_role,excluded=excluded,public_split_sha256=filehash(public_split) if public_split else None,split=('CoOp' if public_split else 'stratified_60_40'),split_seed=split_seed,source_groups='unavailable; exact and visual near-duplicates excluded before roles',seconds=perf_counter()-started))
    return output/'images.json'


def episode(candidate,k,seed,protocol):
    if protocol not in {'A','B','B-temp'} or k<1: raise ValueError('invalid label protocol')
    rows={r['id']:int(r['label']) for r in candidate}; groups={}
    if len(rows)!=len(candidate): raise ValueError('duplicate IDs in candidate pool')
    for i,c in rows.items(): groups.setdefault(c,[]).append(i)
    out={'support':{}}
    if protocol=='B': out.update(select={},calibrate={})
    elif protocol=='B-temp': out.update(calibrate={})
    for c,ids in sorted(groups.items()):
        order=np.random.default_rng([seed,c]).permutation(sorted(ids)).tolist()
        if len(order)<k+(0 if protocol=='A' else 20): raise ValueError('pool too small for declared label budget')
        for i in order[:k]: out['support'][i]=c
        if protocol=='B':
            for i in order[k:k+10]: out['select'][i]=c
            for i in order[k+10:k+20]: out['calibrate'][i]=c
        elif protocol=='B-temp':
            for i in order[k:k+20]: out['calibrate'][i]=c
    return out


def perturb(dictionary,classes,condition,seed):
    p=[[dictionary[c]['core'],*dictionary[c]['attrs']] for c in classes]
    if any(len(r)!=4 for r in p): raise ValueError('exactly 3 attributes required')
    positions=np.random.default_rng(seed).permutation(3*len(classes)); changed=[]
    if condition.startswith('wrong') or condition=='ambiguous':
        ratio=.3 if condition=='ambiguous' else int(condition[5:])/100
        if condition!='ambiguous' and ratio not in {.1,.3,.5}: raise ValueError('unregistered wrong fraction')
        for n in positions[:int(ratio*len(positions)+.5)]:
            i,j=divmod(int(n),3); d=dictionary[classes[i]]; other=d['confuser'];
            if other==classes[i] or other not in classes: raise ValueError('invalid preregistered confuser')
            p[i][j+1]=('An overhead view with a mix of colors and spatial patterns.' if condition=='ambiguous' else dictionary[other]['attrs'][j]);
            changed.append([classes[i],j+1,other if condition!='ambiguous' else None])
    elif condition in {'core-only','missing'}:
        for r in p: r[1 if condition=='core-only' else 3:]=[None]*(3 if condition=='core-only' else 1)
    elif condition=='reorder':
        for r in p: r[1:]=[r[3],r[1],r[2]]
    elif condition=='repeat':
        for r in p: r[3]=r[1]
    elif condition=='paraphrase':
        for name,r in zip(classes,p): r[1:]=dictionary[name]['paraphrase']
    elif condition!='normal': raise ValueError(f'unknown condition {condition}')
    return p,dict(condition=condition,seed=seed,replaced=len(changed),denominator=3*len(classes),actual_fraction=len(changed)/(3*len(classes)),changes=changed,prompt_hash=digest(p))
