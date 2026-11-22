"""Re-score archived predictions against the sealed query IDs. No query truth enters any fit."""
import argparse
import csv
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lira_rs import data, pipeline
from lira_rs.cli import config, configured_plan
from import_completed import BATCHES


def measure(z, y, ids, visual):
    assert np.isfinite(z).all()
    p = softmax(z.astype('float64'),axis=1)
    pred = z.argmax(1)
    correct = pred == y
    n, c = z.shape
    matrix = np.bincount(c*y+pred, minlength=c*c).reshape(c,c)
    counts = matrix.sum(0)+matrix.sum(1)
    denom = matrix.sum(1)
    recall = np.divide(matrix.diagonal(),denom,out=np.zeros(c),where=denom>0)
    f1 = np.divide(2*matrix.diagonal(),counts,out=np.zeros(c),where=counts>0)
    confidence = p.max(1)
    bins = np.minimum((confidence*15).astype(int),14)
    order = np.lexsort((ids,-confidence))
    risk = np.cumsum(~correct[order])/np.arange(1,n+1)
    py = p[np.arange(n),y]
    return dict(accuracy=correct.mean(), macro_f1=f1.mean(),
        brier=(1+np.einsum('ij,ij->i',p,p)-2*py).mean(), nll=-np.log(np.clip(py,1e-12,1)).mean(),
        ece=abs(np.bincount(bins,weights=correct-confidence,minlength=15)).sum()/n, aurc=risk.mean(),
        harm=(~correct & visual).mean(), rescue=(correct & ~visual).mean(),visual_accuracy=visual.mean(),
        confusion=matrix,recall=recall,risk_at_coverage={str(f):risk[int(np.ceil(n*f))-1] for f in (.5,.8,.9,1.)})


@lru_cache(3)
def sealed(dataset):
    path,_ = pipeline.resolve_paths(ROOT,dataset)
    m = data.load_json(path)
    ids = np.array([r['id'] for r in m['images'] if r['role']=='query'])
    truth = {r['id']:r['label'] for r in data.load_json(path.with_name('query.labels.json'))}
    assert len(ids)==len(set(ids))==len(truth) and set(ids)==set(truth)
    return ids,np.array([truth[i] for i in ids]),m['split_hash']


@lru_cache(24)
def paired_baseline(path, dataset):
    ids,y,_ = sealed(dataset)
    with np.load(path/'predictions.npz',allow_pickle=False) as q:
        assert np.array_equal(q['ids'],ids)
        return q['logits'].argmax(1)==y


def compare(actual,expected):
    for key,val in actual.items():
        if isinstance(val,dict):
            compare(val,expected[key])
        else:
            np.testing.assert_allclose(val,expected[key],atol=2e-12,rtol=0,err_msg=key)


def expected_keys(datasets, fresh):
    from types import SimpleNamespace
    settings,head = config(ROOT/'configs/main.yaml')
    result=set()
    for d in datasets:
        for k in settings['shots']:
            profile='full' if k==4 and not fresh else 'main'
            args=SimpleNamespace(profile=profile,K=k,dataset=d,conditions=None,methods=None)
            for pr,c,ts,m,v,_ in configured_plan(args,settings,head):
                for s in settings['support_seeds']:
                    for op in (settings['optimization_seeds'] if m=='prolip' else [None]):
                        result.add((d,k,s,pr,c,ts,m,v,op))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh',action='store_true')
    args=parser.parse_args()
    batches=('replay-ucmerced','replay-eurosat','replay-primary','replay-resisc45') if args.fresh else BATCHES
    name='fresh' if args.fresh else 'archive'
    verified=set(); seen_scores={}; per_batch={}; calibrated=0; checks=0
    output=ROOT/'reports'/f'{name}-metrics-audit.csv'
    with output.open('w',encoding='utf-8-sig',newline='') as out:
        w=csv.writer(out); w.writerow(('batch','dataset','K','seed','protocol','condition','text_seed','method','variant','opt_seed','query_n','scores_sha256','accuracy','harm','rescue'))
        for batch in batches:
            base=ROOT/'results'/batch
            status=data.load_json(base/'run-status.json'); assert status['failed_cells']==0
            count=0
            for f in sorted((base/'cells').rglob('metrics.json')):
                obj=data.load_json(f); meta=obj['meta']; d=meta['dataset']; cell=f.parent
                ids,y,split=sealed(d)
                assert meta['split_hash']==split and obj['score']['query_n']==len(ids)
                condition_dir=cell.parent.parent if meta['requested_method']=='prolip' else cell.parent
                visual=paired_baseline(condition_dir/'visual',d)
                scorehash=data.filehash(cell/'predictions.npz'); assert scorehash==obj['score']['scores_sha256']
                ep=(d,meta['K'],meta['support_seed'],meta['protocol'],meta['condition'],meta['text_seed'],meta['requested_method'],meta['variant_name'],meta.get('optimization_seed'))
                key=(d,scorehash,meta['episode_hash'],float(meta['temperature']),meta['protocol'])
                if key not in seen_scores:
                    with np.load(cell/'predictions.npz',allow_pickle=False) as q:
                        assert np.array_equal(q['ids'],ids) and 'labels' not in q.files
                        z=q['logits']; assert z.shape==(len(y),len(meta['classes']))
                        raw=measure(z,y,ids,visual)
                        cal=measure(z/meta['temperature'],y,ids,visual) if meta['protocol']!='A' else None
                    seen_scores[key]=(raw,cal)
                raw,cal=seen_scores[key]; compare(raw,obj['raw']); assert abs(raw['accuracy']-raw['visual_accuracy']-raw['rescue']+raw['harm'])<1e-12
                if cal is None: assert obj['calibrated'] is None
                else:
                    compare(cal,obj['calibrated']); assert cal['accuracy']==raw['accuracy']; calibrated+=1
                w.writerow((batch,*ep,len(ids),scorehash,raw['accuracy'],raw['harm'],raw['rescue']))
                verified.add(ep); count+=1
            assert count==status['completed_fits'],(batch,count,status)
            per_batch[batch]=count;checks+=count
            print(batch,count,'checked',flush=True)
    expected=expected_keys(data.DATASETS,args.fresh)
    assert verified==expected,('missing',list(expected-verified)[:6],'extra',list(verified-expected)[:6])
    data.save_json(ROOT/'reports'/f'{name}-metrics-audit.json',dict(files=checks,distinct_cells=len(verified),expected_cells=len(expected),
        per_batch=per_batch,unique_score_evaluations=len(seen_scores),calibrated_files=calibrated,passed=True))


if __name__=='__main__':main()
