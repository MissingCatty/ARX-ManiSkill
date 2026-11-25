"""Surface-quality checks use area-uniform samples rather than vertex density."""
import unittest
import numpy as np
from .evaluation import uniform_surface


class SurfaceSampling(unittest.TestCase):
    def test_sampling_tracks_triangle_area_and_is_reproducible(self):
        vertices=np.asarray([[0,0,0],[1,0,0],[0,1,0],[10,0,0],[12,0,0],[10,2,0]])
        faces=np.asarray([[0,1,2],[3,4,5]])
        a=uniform_surface(vertices,faces,count=20000,seed=123)
        b=uniform_surface(vertices,faces,count=20000,seed=123)
        np.testing.assert_array_equal(a,b)
        self.assertLess(abs((a[:,0]>5).mean()-.8),.02)
        self.assertTrue(np.all(a[:,2]==0))
        small=a[a[:,0]<5];large=a[a[:,0]>5]
        self.assertTrue(np.all(small[:,:2]>=0) and np.all(small[:,0]+small[:,1]<=1+1e-12))
        self.assertTrue(np.all(large[:,0]>=10) and np.all(large[:,0]-10+large[:,1]<=2+1e-12))

    def test_degenerate_truth_fails(self):
        with self.assertRaises(ValueError):uniform_surface(np.zeros((3,3)),[[0,1,2]],count=10)


if __name__=='__main__':unittest.main()
