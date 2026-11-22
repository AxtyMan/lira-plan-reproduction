from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from lira_rs.core import Config, fit
from study_candidates import literal_risk

class FormulaAudit(unittest.TestCase):
    def test_brute_force_matches_vectorized_risk_with_missing_description(self):
        rng=np.random.default_rng(741)
        x=rng.normal(size=(9,7)); y=np.repeat(np.arange(3),3)
        text=rng.normal(size=(3,4,7)); mask=np.ones((3,4),dtype=bool); mask[:,3]=False
        expected=fit(x,y,text,Config(),mask=mask)
        actual=literal_risk(x,y,text,mask,Config())
        np.testing.assert_allclose(actual,expected.log['risk'],atol=2e-12,rtol=0)
