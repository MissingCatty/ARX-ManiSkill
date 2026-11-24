"""Numerical export checks, without a simulator or invented physical receipts."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .common import write
from .export import histogram_quantiles


class QuantileTests(unittest.TestCase):
    def test_empirical_quantile_intervals_include_exact_order_statistics(self):
        rng=np.random.default_rng(20261030)
        values=rng.normal(size=(1051,15))
        values[:,14]=1.  # Constant gripper/state dimension must stay exact.
        with tempfile.TemporaryDirectory() as tmp:
            roots=[]
            for i,part in enumerate(np.array_split(values,3)):
                root=Path(tmp)/str(i);roots.append(root)
                write(root/'labels.json',{'items':[{'state':r[:8].tolist(),'action':r[8:].tolist()} for r in part]})
            result=histogram_quantiles(roots,values.min(0),values.max(0),len(values))
            ordered=np.sort(values,axis=0)
            for name,p in (('q01',.01),('q99',.99)):
                exact=ordered[int(np.ceil(p*len(values)))-1]
                bounds=np.array(result[name+'_interval'])
                self.assertTrue((exact>=bounds[:,0]-1e-12).all())
                self.assertTrue((exact<=bounds[:,1]+1e-12).all())
                self.assertTrue((np.abs(exact-result[name])<=np.array(result['maximum_absolute_value_error'])+1e-12).all())
                self.assertEqual(result[name][-1],1.)
            with self.assertRaisesRegex(ValueError,'frame count changed'):
                histogram_quantiles(roots,values.min(0),values.max(0),len(values)+1)


if __name__=='__main__':unittest.main()
