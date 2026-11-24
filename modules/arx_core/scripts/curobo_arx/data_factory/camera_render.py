"""A registered renderer-only near clip correction; no camera calibration edits."""
import numpy as np
from .common import digest


def correct_wrist_near_clip(stage,path,clipping):
    from pxr import UsdGeom,Gf
    camera=UsdGeom.Camera(stage.GetPrimAtPath(path))
    if not camera:raise ValueError('Expected existing calibrated ARX wrist camera')
    if abs(UsdGeom.GetStageMetersPerUnit(stage)-1.)>1e-9:raise ValueError('Camera clip correction requires metre stage units')
    values=np.asarray(clipping,dtype=float)
    old=np.asarray(camera.GetClippingRangeAttr().Get(),dtype=float)
    if values.shape!=(2,) or not np.isfinite(values).all() or not .0001<=values[0]<old[0]-1e-8 or values[1]!=old[1]:
        raise ValueError('Only a positive smaller near clip with unchanged far clip may be registered')
    def other_attributes():
        return {a.GetName():str(a.Get()) for a in camera.GetPrim().GetAttributes() if a.GetName()!='clippingRange'}
    before=other_attributes()
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(*values))
    after=other_attributes()
    if before!=after:raise RuntimeError('Renderer correction changed another camera property')
    return {'path':path,'before_m':old.tolist(),'after_m':list(camera.GetClippingRangeAttr().Get()),
        'all_other_camera_attributes_unchanged':True,'other_attributes_sha256':digest(before),
        'scope':'In-memory new-teacher renderer near clip only; no source USD, calibration, transform, physics or benchmark change'}
