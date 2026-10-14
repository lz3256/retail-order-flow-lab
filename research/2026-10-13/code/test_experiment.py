"""Independent mathematical and temporal checks for substantive research claims."""
import itertools, unittest
import numpy as np
import pandas as pd
from experiment import imbalance, deletion_weights, correlation, holm, design, OUT
from configure import CONTROLS

class ResearchChecks(unittest.TestCase):
    def test_exact_flip_moments_by_enumeration(self):
        s=np.array([1.,-1.,1.]);w=np.array([.1,.3,.6]);e=.2
        values=[];probs=[]
        for bits in itertools.product([0,1],repeat=3):
            k=sum(bits);prob=e**k*(1-e)**(3-k)
            values.append(w@(s*np.where(bits,-1,1)));probs.append(prob)
        mean=np.dot(values,probs);var=np.dot((np.array(values)-mean)**2,probs)
        self.assertAlmostEqual(mean,(1-2*e)*(w@s),places=14)
        self.assertAlmostEqual(var,4*e*(1-e)*(w@w),places=14)

    def test_scaling_cannot_change_rank_ic(self):
        x=np.array([1.,3.,2.,5.,4.]);y=np.array([2.,1.,5.,4.,3.])
        self.assertAlmostEqual(correlation(x,y,True),correlation(.6*x,y,True),places=14)

    def test_weighted_imbalance_and_retention(self):
        buy=np.array([[10.,0.,20.,0.,10.]]);sell=np.array([[0.,10.,0.,20.,0.]])
        self.assertAlmostEqual(imbalance(buy,sell)[0],1/7)
        d=np.array([[.1,.2,.8,.3,.4]])
        w=deletion_weights(d,7,True)
        self.assertEqual(w.sum(),4);self.assertEqual(w[0,2],0)
        self.assertAlmostEqual(imbalance(buy,sell,w)[0],-.2)
        self.assertTrue(np.array_equal(deletion_weights(d,7),deletion_weights(d,7)))

    def test_volume_error_bound_exhaustive(self):
        signs=np.array([1.,-1.,1.,-1.]);weights=np.array([.05,.15,.3,.5])
        x=weights@signs
        for bits in itertools.product([0,1],repeat=4):
            flipped=weights[np.array(bits,dtype=bool)].sum()
            y=weights@(signs*np.where(bits,-1,1))
            self.assertLessEqual(abs(y-x),2*flipped+1e-14)

    def test_holm_known_values(self):
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])

    def test_forecast_design_ignores_outcomes(self):
        rng=np.random.default_rng(42);f=pd.DataFrame(rng.normal(size=(30,7)),columns=CONTROLS)
        f['industry']=np.arange(30)%4;f['qmp']=rng.normal(size=30);f['week1']=rng.normal(size=30)
        x=design(f,'qmp');f['week1']=np.nan
        np.testing.assert_array_equal(x,design(f,'qmp'))
        self.assertEqual(x.shape,(30,108))

    def test_weight_cancellation_covariance_identity(self):
        b=np.array([[1.,3.,5.,2.,4.],[8.,1.,2.,7.,4.]])
        s=np.array([[4.,2.,1.,6.,3.],[2.,5.,3.,1.,4.]])
        p=(b+s)/(b+s).sum(axis=1,keepdims=True);daily=(b-s)/(b+s)
        w=np.array([[.8,.6,.9,.7,.5],[.7,.8,.8,.4,.9]])
        ew=np.sum(p*w,axis=1);eo=np.sum(p*daily,axis=1)
        cov=np.sum(p*w*daily,axis=1)-ew*eo
        np.testing.assert_allclose(imbalance(b,s,w)-imbalance(b,s),cov/ew,atol=1e-14)
        np.testing.assert_allclose(imbalance(b,s,np.array([[.2]*5,[.8]*5])),imbalance(b,s),atol=1e-14)

    def test_completed_forecast_fence(self):
        path=OUT/'forecast_weekly.csv'
        if not path.exists():self.skipTest('Forecast run not finished yet')
        f=pd.read_csv(path,parse_dates=['signal_date','latest_training_exit'])
        self.assertTrue((f.latest_training_exit<=f.signal_date).all())
        self.assertTrue((f.mature_training_windows>=52).all())
        self.assertFalse(f.duplicated(['signal_date','method']).any())

if __name__=='__main__':unittest.main(verbosity=2)
