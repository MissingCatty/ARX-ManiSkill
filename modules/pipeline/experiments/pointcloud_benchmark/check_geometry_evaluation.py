"""Check units, distance definitions and rigid/similarity frame recovery."""
import numpy as np
from scipy.spatial.transform import Rotation
from geometry_evaluation import fit_similarity,cube_rotations,surface_metrics,unproject

rng=np.random.default_rng(23);a=rng.normal(size=(256,3))
r=Rotation.from_euler('xyz',[17,-11,23],degrees=True).as_matrix();s=.073;t=np.array([.2,-.1,.45])
b=a@r.T*s+t
got_s,got_r,got_t=fit_similarity(a,b,(.01,1.))
assert np.allclose(got_s,s,atol=1e-12) and np.allclose(got_r,r,atol=1e-12) and np.allclose(got_t,t,atol=1e-12)
assert len(cube_rotations())==24 and all(np.isclose(np.linalg.det(x),1) for x in cube_rotations())
m=surface_metrics(b,b,b)
assert m['chamfer_l1_mm']==0 and m['fscore_2mm']==1 and m['coverage_5mm']==1
z=np.ones((3,3),dtype=np.float32)*.4;k=np.array([[2,0,1],[0,2,1],[0,0,1]])
p=unproject(z,np.ones((3,3),dtype=bool),k)
assert np.allclose(p[0],[-.2,-.2,.4]) and np.allclose(p[4],[0,0,.4])
print('Geometry unit/frame/distance checks passed')
