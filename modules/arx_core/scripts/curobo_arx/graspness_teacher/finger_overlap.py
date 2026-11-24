"""Proposal-quality evidence from unchanged, recorded ARX finger geometry.

Positive target-vertex containment can reject a poor proposal. A negative result
is never a collision-free certificate: flat authored pieces and edge-only
intersections are not resolved here. Original planning/physical gates remain.
"""
from pathlib import Path
import numpy as np
from scripts.curobo_arx.data_factory.common import read, sha, verify_files
from scripts.curobo_arx.data_factory.geometry import rigid


def proposal_variant(registration, request):
    policy=request['pool'].get('paired_policy')
    if policy is not None:
        if policy not in ('baseline','variant'):raise ValueError('Unknown proposal comparison policy')
        return policy=='variant'
    return registration.get('production') is True


def penetration(points, bounds, equations):
    points=np.asarray(points,dtype=float)
    bounds=np.asarray(bounds,dtype=float);equations=np.asarray(equations,dtype=float)
    if (points.ndim!=2 or points.shape[1]!=3 or bounds.shape!=(2,3)
            or equations.ndim!=2 or equations.shape[1]!=4
            or not len(equations) or not np.isfinite(points).all()
            or not np.isfinite(bounds).all() or not np.isfinite(equations).all()
            or (bounds[0]>bounds[1]).any()
            or not np.allclose(np.linalg.norm(equations[:,:3],axis=1),1.,atol=1e-6)):
        raise ValueError('Invalid authored-piece query')
    selected=points[((points>=bounds[0])&(points<=bounds[1])).all(1)]
    if not len(selected):return 0.
    return max(0.,float(-(selected@equations[:,:3].T+equations[:,3]).max(1).min()))


class FingerOverlap:
    def __init__(self, specification):
        path=Path(specification['path'])
        if sha(path)!=specification['sha256']:raise RuntimeError('Registered finger proposal model changed')
        model=read(path)
        verify_files(model['source_files'])
        if model['version']!='authored_arx_finger_containment_v1' or not model['parts']:
            raise ValueError('Missing registered authored finger pieces')
        self.parts=model['parts'];self.unresolved=model['unresolved_flat_parts']
        self.threshold=float(specification['minimum_detected_penetration_m'])
        if self.threshold!=.0001:raise ValueError('Unregistered proposal-quality threshold')

    def inspect(self, goal, snapshot, target_vertices):
        goal=rigid(goal);eef=rigid(snapshot['link_world']['right_eef_link'])
        obj=rigid(snapshot['objects']['test_object']['world'])
        world=np.asarray(target_vertices)@obj[:3,:3].T+obj[:3,3]
        converted={};maximum=0.;link_depth={}
        for part in self.parts:
            link=part['link']
            if link not in converted:
                eef_from_link=np.linalg.inv(eef)@rigid(snapshot['link_world'][link])
                world_from_link=goal@eef_from_link
                converted[link]=(world-world_from_link[:3,3])@world_from_link[:3,:3]
            depth=penetration(converted[link],part['bounds'],part['equations'])
            maximum=max(maximum,depth);link_depth[link]=max(link_depth.get(link,0.),depth)
        return {'reject_proposal':maximum>self.threshold,'maximum_detected_penetration_m':maximum,
                'per_link_m':link_depth,'unresolved_flat_parts':len(self.unresolved),
                'collision_free_certificate':False,'robot_model_or_physics_modified':False,
                'scope':'Additional proposal-quality heuristic; positive vertex containment only'}


def guarded_teacher(base):
    """Apply only to the registered variant's proposal stage, never execution."""
    from scripts.curobo_arx.data_factory.teacher import Rejection
    class Teacher(base):
        proposal_request=None

        def qualify_pool(self, request):
            active=proposal_variant(self.registration,request)
            self.proposal_request=request if active else None
            try:return super().qualify_pool(request)
            finally:self.proposal_request=None

        def nominal_closed(self, grasp, *args, **kwargs):
            request=self.proposal_request
            if request is not None:
                if not hasattr(self,'finger_overlap'):
                    self.finger_overlap=FingerOverlap(self.registration['authored_finger_proposal_filter'])
                target=self.visual_target(request)
                if target is None:raise RuntimeError('Proposal geometry has no verified actual target')
                evidence=self.finger_overlap.inspect(grasp,request['snapshot'],target.mesh.vertices)
                if evidence['reject_proposal']:raise Rejection('authored_open_finger_target_overlap',evidence)
            return super().nominal_closed(grasp,*args,**kwargs)
    return Teacher
