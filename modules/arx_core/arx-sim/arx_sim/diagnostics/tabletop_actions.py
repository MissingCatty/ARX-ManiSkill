"""Live ARX tabletop loading and unannotated active-viewport recording."""
from pathlib import Path
import json,time
import numpy as np

def tick(bridge):
    rec=getattr(bridge,'video_recording',None)
    if not rec:return
    from omni.kit.viewport.utility import get_active_viewport,capture_viewport_to_file
    view=get_active_viewport()
    if str(view.camera_path)!=rec['camera_path']:
        rec['error']='Active camera changed during capture';return
    number=len(rec['frames']);path=Path(rec['directory'])/f'{number:06d}.png'
    rec['captures'].append(capture_viewport_to_file(view,str(path)))
    rec['frames'].append({'file':path.name,'wall_time':time.monotonic(),'sim_time':bridge.context.world.current_time})

def handle(bridge,request):
    from pxr import UsdGeom,Gf
    from isaacsim.core.experimental.prims import RigidPrim
    action=request['action'];c=bridge.context
    if action=='grasp_viewport_state':
        from omni.kit.viewport.utility import get_active_viewport
        view=get_active_viewport()
        if view is None:raise RuntimeError('No active viewport')
        saved=request.get('restore')
        if saved:
            path=saved['camera_path']
            if not path.startswith('/World/SceneCameras/'):raise ValueError('Expected an existing scene camera')
            prim=c.stage.GetPrimAtPath(path)
            if not prim or not prim.IsA(UsdGeom.Camera):raise ValueError('Scene camera is absent')
            matrix=np.asarray(saved['local_matrix'],dtype=float)
            if matrix.shape!=(4,4) or not np.isfinite(matrix).all():raise ValueError('Invalid camera matrix')
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(matrix.tolist()))
            for name,value in saved['attributes'].items():
                if name not in ('focalLength','horizontalAperture','verticalAperture','horizontalApertureOffset','verticalApertureOffset','clippingRange','projection'):raise ValueError('Unsupported camera attribute')
                prim.GetAttribute(name).Set(Gf.Vec2f(*value) if name=='clippingRange' else value)
            view.camera_path=path
        prim=c.stage.GetPrimAtPath(str(view.camera_path))
        attrs={name:prim.GetAttribute(name).Get() for name in ('focalLength','horizontalAperture','verticalAperture','horizontalApertureOffset','verticalApertureOffset','clippingRange','projection')}
        attrs['clippingRange']=list(attrs['clippingRange'])
        return {'camera_path':str(view.camera_path),'local_matrix':np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()).tolist(),
                'world_matrix':np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).tolist(),'attributes':attrs}
    if action=='grasp_asset_info':
        return {name:{'path':obj.paths[0],'mass_kg':obj.get_masses().numpy().tolist(),
                      'world':c.controller._world_pose(obj).tolist()} for name,obj in bridge.tabletop_objects.items()}
    if action=='grasp_asset_load':
        if bridge.active:raise RuntimeError('Trajectory active')
        name=Path(request['asset_path']).stem
        for i,(key,obj) in enumerate(bridge.tabletop_objects.items()):
            obj.set_world_poses(positions=np.array([[10+i*.4,10,.4]]),orientations=np.array([[1.,0.,0.,0.]]))
            obj.set_velocities(linear_velocities=np.zeros((1,3)),angular_velocities=np.zeros((1,3)))
        obj=bridge.tabletop_objects[name]
        obj.set_world_poses(positions=np.array(request['position'])[None],orientations=np.array([request.get('orientation',[1.,0.,0.,0.])]))
        obj.set_velocities(linear_velocities=np.zeros((1,3)),angular_velocities=np.zeros((1,3)))
        bridge.objects['test_object']=obj;bridge.test_object_size=request['size_m']
        return {'path':obj.paths[0]}
    if action=='grasp_asset_bind':
        return bridge.snapshot()['objects']['test_object']
    if action=='grasp_video_start':
        from omni.kit.viewport.utility import get_active_viewport
        if getattr(bridge,'video_recording',None):raise RuntimeError('Recording already active')
        directory=Path(request['directory']);directory.mkdir(parents=True,exist_ok=False)
        view=get_active_viewport();cam=c.stage.GetPrimAtPath(str(view.camera_path))
        bridge.video_recording={'directory':str(directory),'camera_path':str(view.camera_path),'camera_matrix':np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(cam)).tolist(),'frames':[],'captures':[]}
        return {'camera_path':str(view.camera_path)}
    if action=='grasp_video_stop':
        rec=getattr(bridge,'video_recording',None)
        if rec is None:return {'recording':False}
        bridge.video_recording=None
        info={k:v for k,v in rec.items() if k!='captures'}
        (Path(rec['directory'])/'frames.json').write_text(json.dumps(info,indent=2))
        # Keep capture futures alive until the following request / simulation frames complete.
        bridge.last_video_captures=rec['captures']
        return {'frames':len(rec['frames']),'camera_path':rec['camera_path'],'error':rec.get('error')}
    raise ValueError(action)
