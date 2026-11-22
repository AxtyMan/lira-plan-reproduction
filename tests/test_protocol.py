import sys
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lira_rs.core import Config, fit, score, grouped, geometry, softmax, stability
from lira_rs.evaluation import metrics, temperature
from lira_rs.data import episode, perturb


def fixture(k=3):
    z = np.array([[1,.15],[1,-.15],[.1,1],[.2,1],[-1,.1],[-1,-.1]])
    if k==1: z=z[::2]
    y=np.repeat(np.arange(3),len(z)//3)
    t=np.array([[[1,0],[1,.2],[0,1],[1,.1]],[[0,1],[.1,1],[-1,0],[0,1]],
                [[-1,0],[-1,.1],[0,1],[-1,-.1]]])
    return z,y,t


class ProtocolTests(unittest.TestCase):
    def test_one_shot_exact_visual(self):
        x,y,t=fixture(1)
        a=fit(x,y,t); b=fit(x,y,t,method='visual')
        self.assertEqual(a.log['fallback_reason'],'one_shot')
        np.testing.assert_array_equal(score(x,a),score(x,b))

    def test_four_conditions_and_strict_choice(self):
        x,y,t=fixture(); s=fit(x,y,t); p=np.array(s.log['fold_probabilities'])
        self.assertEqual(p.shape,(6,4,5,3))
        risk=((p-np.eye(3)[y,None,None,:])**2).sum(-1).mean(0)
        np.testing.assert_allclose(s.log['risk'],risk,atol=1e-12)
        eligible=np.flatnonzero(np.max(risk-risk[:,[0]],axis=0)<=1e-8)
        j=min(eligible,key=lambda j:(round(float(risk[0,j]),8),j))
        self.assertAlmostEqual(s.log['lambda'],Config().lambdas[j])
        self.assertEqual(s.log['conditions'],['full','drop_1','drop_2','drop_3'])

    def test_image_group_all_views_removed(self):
        x,y,t=fixture(); x2=np.repeat(x,2,axis=0); y2=np.repeat(y,2)
        ids=np.repeat(list('abcdef'),2); g=grouped(x2,y2,ids)
        self.assertEqual(len(g.x),6); self.assertEqual(g.views,[2]*6)
        a=fit(x2,y2,t,ids=ids); b=fit(x,y,t,ids=list('abcdef'))
        np.testing.assert_allclose(a.log['risk'],b.log['risk'],atol=1e-12)
        self.assertEqual(a.log['fold_removed_views'],[2]*6)
        self.assertNotIn('a',a.log['fold_remaining_ids'][0])

    def test_query_permutation_and_label_independence(self):
        x,y,t=fixture(); s=fit(x,y,t); pred=score(x,s); order=np.array([4,3,2,1,0,5]);
        np.testing.assert_allclose(pred[order],score(x[order],s),atol=1e-12)
        self.assertNotEqual(metrics(pred,y)['accuracy'],metrics(pred,(y+1)%3)['accuracy'])
        np.testing.assert_array_equal(pred,score(x,s))
        with self.assertRaises(TypeError): fit(x,y,t,query_labels=y)
        with self.assertRaises(TypeError): fit(x,y,t,val_labels=y)

    def test_cache_name_does_not_determine_labels(self):
        x,y,t=fixture(); a=fit(x,y,t,ids=[f'forest{i:02}' for i in range(len(x))]);
        b=fit(x,y,t,ids=[f'river{i:02}' for i in range(len(x))]);
        np.testing.assert_allclose(score(x,a),score(x,b),atol=1e-12)

    def test_geometry_and_certificate(self):
        x,y,t=fixture(); p,a,r,u,_,_=geometry(x/np.linalg.norm(x,axis=1,keepdims=True),y,t,np.ones(t.shape[:2],bool),Config())
        self.assertTrue(np.all(np.linalg.norm(u,axis=1)<=r+1e-12))
        for c in range(len(p)):
            for k in range(c): self.assertGreaterEqual(np.linalg.norm(p[c]+u[c]-p[k]-u[k])+1e-12,.5*np.linalg.norm(p[c]-p[k]))
        margin,cert=stability(x,p,r); self.assertTrue((margin>=0).all())
        self.assertTrue(np.all(score(x,fit(x,y,t,method='visual')).argmax(1)[cert]==score(x,fit(x,y,t,method='fixed')).argmax(1)[cert]))

    def test_invalid_text_and_duplicate_centers(self):
        x,y,t=fixture(); st=fit(x,y,np.zeros_like(t)); self.assertEqual(st.log['fallback_reason'],'invalid_text')
        self.assertTrue(np.isfinite(score(x,st)).all())
        x[:]=[1,0]; st=fit(x,y,t); self.assertEqual(st.log['displacement'],[0]*3)

    def test_episode_A_never_reads_calibration(self):
        rows=[{'id':f'{c}:{i}','label':c} for c in range(3) for i in range(40)]
        a=episode(rows,1,33,'A'); b=episode(rows,4,33,'B'); c=episode(rows,4,33,'B-temp')
        self.assertEqual(len(a['support']),3); self.assertNotIn('calibrate',a); self.assertNotIn('select',a)
        self.assertTrue(set(a['support'])<=set(b['support']))
        self.assertFalse(set(b['support']) & (set(b['select'])|set(b['calibrate'])))
        self.assertEqual((len(b['select']),len(b['calibrate']),len(c['calibrate'])),(30,30,60))

    def test_temperature_and_harm_identity(self):
        x,y,t=fixture(); base=score(x,fit(x,y,t,method='visual')); a=score(x,fit(x,y,t,method='fixed'))
        T=temperature(a,y); self.assertGreater(T,0)
        np.testing.assert_array_equal(a.argmax(1),(a/T).argmax(1))
        m=metrics(a,y,base=base)
        self.assertAlmostEqual(m['accuracy']-metrics(base,y)['accuracy'],m['rescue']-m['harm'])

    def test_text_replacement_preserves_core(self):
        d={'a':{'core':'A','attrs':['aa','ab','ac'],'confuser':'b'},'b':{'core':'B','attrs':['ba','bb','bc'],'confuser':'a'}}
        p,meta=perturb(d,['a','b'],'wrong30',1)
        self.assertEqual([x[0] for x in p],['A','B']); self.assertEqual(meta['replaced'],2)
        self.assertEqual((p,meta),perturb(d,['a','b'],'wrong30',1))

if __name__=='__main__': unittest.main()
