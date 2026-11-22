import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lira_rs import core,pipeline,data
from lira_rs.baselines import parts
from test_protocol import fixture

class StageTests(unittest.TestCase):
    def test_strict_fold_matches_direct_refit(self):
        x,y,t=fixture(); s=core.fit(x,y,t); cfg=core.Config(); z=core.image_unit(x); saved=np.array(s.log['fold_probabilities'])
        for i in range(len(x)):
            for j in range(4):
                mask=np.ones(t.shape[:2],dtype=bool)
                if j: mask[:,j]=False
                p,a,r,u,_,_=core.geometry(np.delete(z,i,axis=0),np.delete(y,i),t,mask,cfg)
                for k,lam in enumerate(cfg.lambdas):
                    baseline=core.softmax(-np.sum((z[i]-(p+lam*u))**2,axis=1)/cfg.tau)
                    np.testing.assert_allclose(saved[i,j,k],baseline,atol=1e-12)

    def test_gda_head_matches_explicit_inverse(self):
        rng=np.random.default_rng(42); x=core.image_unit(rng.normal(size=(8,15))); y=np.repeat([0,1],4)
        t=core.unit(rng.normal(size=(2,4,15))); mask=np.ones((2,4),bool); cfg=core.Config()
        for tm in (False,True):
            v=x; label=y
            if tm:
                sim=np.einsum('cd,cjd->cj',core.unit(core.means(x,y,2)),t); v=np.vstack([(t*sim[...,None]).reshape(-1,15),x]); label=np.r_[np.repeat([0,1],4),y]
            mu=core.means(v,label,2); u=v-mu[label]; cov=np.cov(u.T)
            inv=15*np.linalg.pinv((len(u)-1)*cov+np.trace(cov)*np.eye(15))
            w=inv@mu.T; b=-np.log(2)-np.sum(mu.T*w,axis=0)*.5
            _,ww,bb=parts(x,y,t,mask,cfg,tm); np.testing.assert_allclose(w,ww,rtol=1e-10,atol=1e-10); np.testing.assert_allclose(b,bb,atol=1e-10)

    def test_low_rank_folds_match_full_rebuild(self):
        from lira_rs.baselines import loo_scores_fast,alpha_logits
        rng=np.random.default_rng(42); cfg=core.Config()
        for dim in (10,32):
            x=core.image_unit(rng.normal(size=(12,dim))); y=np.repeat(np.arange(3),4); t=core.unit(rng.normal(size=(3,4,dim))); mask=np.ones(t.shape[:2],bool); mask[0,-1]=False
            for tm in (False,True):
                out=loo_scores_fast(x,y,t,mask,cfg,tm)
                for i in range(len(x)):
                    j=np.arange(len(x))!=i; expected=alpha_logits(x[i:i+1],parts(x[j],y[j],t,mask,cfg,tm),tm)[0]
                    np.testing.assert_allclose(out[i],expected,atol=2e-7,rtol=2e-9)

    def test_file_roles_and_sealed_labels(self):
        x,y,t=fixture(); classes=['a','b','c']; ids=[f's{i}' for i in range(6)]; qids=['q2','q3','q4']; cfg=core.Config()
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); m=d/'images.json'; pool=[{'id':i,'label':int(v)} for i,v in zip(ids,y)];
            man={'dataset':'test-fixture','split_hash':'s','classes':classes,'images':[{'id':i,'role':'candidate'} for i in ids]+[{'id':i,'role':'query'} for i in qids]}
            data.save_json(m,man); data.save_json(d/'candidate.labels.json',pool)
            real_load=data.load_json
            def deny(path):
                if any(term in Path(path).name for term in ('query.labels','val.labels','calibrate')): raise AssertionError(f'forbidden label access: {path}')
                return real_load(path)
            xc={'z':np.vstack((x,x[::2])),'split_hash':np.array('s'),'cache_key':np.array('ck'),'ids':np.array(ids+qids)}
            tc={'classes':np.array(classes),'t':t,'mask':np.ones(t.shape[:2],bool),'prompt_hash':np.array('ph')}
            ix={i:j for j,i in enumerate(ids+qids)}; out=d/'lira'; base=d/'visual'
            with patch.object(pipeline,'load_json',side_effect=deny):
                pipeline.frozen_episode(m,2,101,'A',d/'episode.json'); e=real_load(d/'episode.json');
                pipeline.validate_episode_ids(e,man)
                bad={**e,'roles':{'support':{**e['roles']['support'],'q2':0}}}
                with self.assertRaises(ValueError): pipeline.validate_episode_ids(bad,man)
                for method,p in [('lira',out),('visual',base)]:
                    s,met=pipeline.fit_artifact(e,xc,tc,method,cfg,p,ix); pipeline.score_artifact(man,xc,ix,s,met,p)
            first=pipeline.load_npz(out/'predictions.npz')['logits'].copy()
            for offs in (0,1):
                data.save_json(d/'query.labels.json',[{'id':i,'label':(c+offs)%3} for c,i in enumerate(qids)])
                result=pipeline.evaluate_artifact(m,out,base); np.testing.assert_array_equal(first,pipeline.load_npz(out/'predictions.npz')['logits'])
                if not offs: a=result['accuracy']
                else: self.assertNotEqual(a,result['accuracy'])

if __name__=='__main__': unittest.main()
