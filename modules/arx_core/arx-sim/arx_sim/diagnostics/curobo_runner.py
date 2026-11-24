"""Run the normal ARX scene with a file mailbox for measured cuRobo validation.

Requests and replies stay under ARX_CUROBO_VALIDATION_DIR; no hardware API is used.
All USD, PhysX and articulation access happens on the simulation main thread.
"""
import json
import os
import time
from pathlib import Path

import numpy as np


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value))
    temp.replace(path)


class ValidationBridge:
    trajectory_trace_version = 'arx_policy_trace_s02_v1'
    def __init__(self, context):
        from isaacsim.core.experimental.prims import RigidPrim
        from omni.physx import get_physx_simulation_interface
        self.context = context
        self.root = Path(os.environ['ARX_CUROBO_VALIDATION_DIR'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.links = {name: RigidPrim(context.config['simulation']['articulation_root']+'/'+name)
                      for name in context.robot.link_names}
        self.objects = {p.GetName(): RigidPrim(str(p.GetPath()))
                        for p in context.stage.Traverse()
                        if p.GetName() in ('red_cube','blue_cylinder','green_sphere')}
        self.tabletop_objects={p.GetName().removeprefix('asset_'):RigidPrim(str(p.GetPath())) for p in context.stage.Traverse() if str(p.GetParent().GetPath())=='/World/Task/assets'}
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(self.contact)
        self.contacts = []
        self.active = None
        self.records = []
        restore=self.root/'restore.json'
        if restore.exists():
            saved=json.loads(restore.read_text());q=np.array(saved['joint_positions'])
            limits=np.stack([v.numpy()[0] for v in context.robot.get_dof_limits()],axis=-1)
            q=np.clip(q,limits[:,0],limits[:,1])
            context.robot.set_dof_positions(q)
            context.robot.set_dof_position_targets(q)
            context.robot.set_dof_velocities(np.zeros_like(q))
            context.robot.set_dof_velocity_targets(np.zeros_like(q))
            for name,entry in saved.get('objects',{}).items():
                if name in self.objects:
                    self.objects[name].set_world_poses(positions=np.array(entry['world'])[:3,3][None],orientations=np.array([[1.,0.,0.,0.]]))
            restore.unlink()
        write_json(self.root/'ready.json', self.snapshot())

    def contact(self, headers, data):
        from pxr import PhysicsSchemaTools
        if self.active is None:
            return
        for h in headers:
            a, b = str(PhysicsSchemaTools.intToSdfPath(h.actor0)), str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            right = lambda p: '/right_link' in p or '/right_eef_link' in p
            if right(a) or right(b):
                self.contacts.append({'a': a, 'b': b, 'type': int(h.type),
                                      'sim_time': self.context.world.current_time,
                                      'num_contacts': int(h.num_contact_data)})

    def snapshot(self):
        c = self.context
        poses = {n: c.controller._world_pose(p).tolist() for n,p in self.links.items()}
        return {'joint_names': c.robot.dof_names,
                'joint_positions': c.controller._positions().tolist(),
                'joint_limits': np.stack([v.numpy()[0] for v in c.robot.get_dof_limits()],axis=-1).tolist(),
                'link_world': poses, 'right_arm': c.config['controllers']['right_arm']['joints'],
                'camera_errors':dict(c.controller._camera_errors),
                'camera_times':{n:p.frame_time for n,p in c.cameras.items()},
                'camera_world': {n:c.controller._world_pose(p).tolist() for n,p in c.camera_prims.items()},
                'objects': {n:{'path':p.paths[0], 'world':c.controller._world_pose(p).tolist()}
                            for n,p in self.objects.items()},
                'sim_time': c.world.current_time}

    def request(self, request):
        from pxr import UsdGeom, UsdPhysics, Gf
        c = self.context
        action = request['action']
        if action == 'snapshot':
            return self.snapshot()
        if action == 'place_object':
            if self.active or c.controller.active is not None:
                raise ValueError('Controller is busy')
            name=request['name']
            if name not in self.objects:raise ValueError('Expected an existing task object')
            position=np.asarray(request['position'],dtype=float)
            if position.shape!=(3,) or not np.isfinite(position).all():raise ValueError('Invalid position')
            obj=self.objects[name]
            obj.set_world_poses(positions=position[None],orientations=np.array([[1.,0.,0.,0.]]))
            obj.set_velocities(linear_velocities=np.zeros((1,3)),angular_velocities=np.zeros((1,3)))
            return self.snapshot()['objects'][name]
        if action == 'export_scene':
            from isaacsim.core.experimental.prims import XformPrim
            cache=UsdGeom.XformCache()
            base=np.asarray(self.snapshot()['link_world']['body4'])
            world_to_base=np.linalg.inv(base)
            meshes=[]
            for prim in c.stage.Traverse():
                path=str(prim.GetPath())
                if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                    continue
                if '/right_link' in path or '/right_eef_link' in path:
                    continue
                matrix=np.asarray(cache.GetLocalToWorldTransform(prim)).T
                ancestor=prim
                while ancestor and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                    ancestor=ancestor.GetParent()
                if ancestor and ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                    authored=np.asarray(cache.GetLocalToWorldTransform(ancestor)).T
                    live=c.controller._world_pose(XformPrim(str(ancestor.GetPath())))
                    matrix=live@np.linalg.inv(authored)@matrix
                if prim.IsA(UsdGeom.Mesh):
                    mesh=UsdGeom.Mesh(prim)
                    points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
                    counts=mesh.GetFaceVertexCountsAttr().Get(); indices=mesh.GetFaceVertexIndicesAttr().Get()
                    faces=[]; offset=0
                    for count in counts:
                        ids=indices[offset:offset+count];offset+=count
                        faces.extend([[ids[0],ids[k],ids[k+1]] for k in range(1,count-1)])
                else:
                    bounds=UsdGeom.BBoxCache(0,[UsdGeom.Tokens.default_]).ComputeLocalBound(prim).ComputeAlignedRange()
                    lo,hi=np.array(bounds.GetMin()),np.array(bounds.GetMax())
                    points=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
                    faces=[[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]
                if not len(points):continue
                vertices=(np.c_[points,np.ones(len(points))]@(world_to_base@matrix).T)[:,:3]
                nearest=np.maximum(vertices.min(0),0)+np.minimum(vertices.max(0),0)
                if np.linalg.norm(nearest)>1.4:continue
                meshes.append({'path':path,'vertices':vertices.tolist(),'faces':faces})
            write_json(self.root/'scene_meshes.json',{'base_world':base.tolist(),'meshes':meshes})
            return {'mesh_count':len(meshes)}
        if action == 'obstacle':
            if self.active:
                raise ValueError('Cannot change obstacle during execution')
            path='/World/CuroboValidationObstacle'
            if c.stage.GetPrimAtPath(path):
                c.stage.RemovePrim(path)
            if request.get('remove'):
                return {'removed': True}
            center=np.asarray(request['center'],dtype=float)
            dims=np.asarray(request['dims'],dtype=float)
            if center.shape != (3,) or dims.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(dims).all() or (dims <= 0).any():
                raise ValueError('Expected finite center[3] and positive dims[3]')
            cube=UsdGeom.Cube.Define(c.stage,path)
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*center))
            cube.AddScaleOp().Set(Gf.Vec3d(*dims))
            cube.CreateDisplayColorAttr([Gf.Vec3f(1.0,0.3,0.05)])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            return {'path':path,'center':center.tolist(),'dims':dims.tolist()}
        if action == 'trajectory':
            if self.active or c.controller.active is not None:
                raise ValueError('Controller is busy')
            names=request['joint_names']
            if names != c.config['controllers']['right_arm']['joints']:
                raise ValueError('Only exact ARX right arm joint order is supported')
            q=np.asarray(request['position'],dtype=float)
            times=np.asarray(request['time_from_start'],dtype=float)
            if q.ndim != 2 or q.shape[1] != len(names) or len(q)<2 or times.shape != (len(q),):
                raise ValueError('Invalid trajectory dimensions')
            if not np.isfinite(q).all() or not np.isfinite(times).all() or times[0]!=0 or (np.diff(times)<=0).any() or times[-1]>120:
                raise ValueError('Invalid values or trajectory times')
            idx=c.controller.arm_indices
            limits=np.stack([v.numpy()[0,idx] for v in c.robot.get_dof_limits()],axis=-1)
            if (q<limits[:,0]-1e-5).any() or (q>limits[:,1]+1e-5).any():
                raise ValueError('Trajectory violates joint limits')
            if np.max(np.abs(q[0]-c.controller._positions()[idx]))>0.03:
                raise ValueError('Trajectory start differs from measured arm position')
            tolerance=float(request.get('position_tolerance_rad',0.015))
            if not np.isfinite(tolerance) or not 0<tolerance<=0.015:
                raise ValueError('Invalid endpoint tolerance')
            gripper=request.get('gripper_target')
            if gripper is not None and (not np.isfinite(gripper) or not -3.2<=gripper<=0.1):
                raise ValueError('Invalid ARX gripper target')
            trace = request.get('trace')
            if trace is not None:
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts/curobo_arx'))
                from lilo_vla.trace import VERSION, stamp
                if trace.get('version') != VERSION or not all(trace.get(k) for k in ('request_id','observation_id','command_id')):
                    raise ValueError('Invalid S02 trajectory trace identity')
                trace = {**trace, 'simulator_received':stamp(), 'execution_start_sim_time':c.world.current_time,
                         'measured_start':self.snapshot()}
            self.active={'q':q,'times':times,'start':c.world.current_time,'request':request,'wall_start':time.monotonic()}
            if trace is not None:self.active['trace'] = trace
            if gripper is not None:
                c.controller._set_gripper_api_target(gripper,'right')
                if trace is not None:trace['gripper_setpoint_applied'] = stamp()
            self.contacts=[]; self.records=[]
            return None
        if action.startswith('grasp_'):
            import importlib
            from arx_sim.diagnostics import grasp_actions
            return importlib.reload(grasp_actions).handle(self,request)
        raise ValueError('Unknown diagnostic action')

    def before_step(self,poll_requests=True):
        request_path=self.root/'request.json'
        if poll_requests and request_path.exists() and self.active is None:
            request=json.loads(request_path.read_text()); request_path.unlink()
            try:
                result=self.request(request)
                if result is not None:
                    write_json(self.root/'response.json',{'id':request['id'],'ok':True,'result':result})
            except Exception as exc:
                import traceback
                traceback.print_exc()
                write_json(self.root/'response.json',{'id':request['id'],'ok':False,'error':str(exc)})
        if self.active:
            a=self.active; elapsed=self.context.world.current_time-a['start']
            if elapsed-a.get('last_target_time',-1.0)<1/30-1e-6:
                return
            a['last_target_time']=elapsed
            target=np.array([np.interp(elapsed,a['times'],a['q'][:,i]) for i in range(a['q'].shape[1])])
            self.context.robot.set_dof_position_targets(target,dof_indices=self.context.controller.arm_indices)
            if 'trace' in a and 'first_arm_setpoint_applied' not in a['trace']:
                from lilo_vla.trace import stamp
                a['trace']['first_arm_setpoint_applied'] = stamp()
                a['trace']['first_arm_setpoint_sim_time'] = self.context.world.current_time

    def after_step(self):
        from arx_sim.diagnostics.research_setup import tick as research_tick
        research_tick(self)
        if getattr(self,"video_recording",None):
            from arx_sim.diagnostics.tabletop_actions import tick
            try:tick(self)
            except Exception as exc:self.video_recording["error"]=str(exc)
        if not self.active:
            return
        c=self.context; a=self.active; elapsed=c.world.current_time-a['start']
        if self.records and elapsed-self.records[-1]['time'] < 1/30-1e-6:
            return
        actual=c.controller._positions()[c.controller.arm_indices]
        target=np.array([np.interp(elapsed,a['times'],a['q'][:,i]) for i in range(a['q'].shape[1])])
        self.records.append({'time':elapsed,'position':actual.tolist(),'target':target.tolist(),
                             'ee_world':c.controller._world_pose(c.ee_prim).tolist(),
                             'gripper':c.controller._positions()[c.controller.gripper_indices].tolist(),
                             'objects':{n:c.controller._world_pose(p).tolist() for n,p in self.objects.items()}})
        tolerance=float(a['request'].get('position_tolerance_rad',0.015))
        settled=elapsed>=a['times'][-1] and np.max(np.abs(actual-a['q'][-1]))<tolerance
        timed_out=elapsed>a['times'][-1]+4
        if settled or timed_out:
            result={'settled':bool(settled),'records':self.records,'contacts':self.contacts,'final':self.snapshot(),
                    'wall_duration_s':time.monotonic()-a['wall_start'],'sim_duration_s':elapsed}
            if 'trace' in a:
                from lilo_vla.trace import stamp
                result['trace'] = {**a['trace'], 'execution_finished':stamp(),
                                   'execution_end_sim_time':c.world.current_time}
            write_json(self.root/'response.json',{'id':a['request']['id'],'ok':True,'result':result})
            self.active=None


def main():
    from arx_sim.runtime import simulation_loop, stage_loader
    original_load=stage_loader.load_composed_stage
    def load(app,config):
        from pxr import PhysxSchema, UsdPhysics
        stage=original_load(app,config)
        catalog_path=Path(os.environ.get('ARX_TABLETOP_CATALOG') or '/workspace/COOLAS-ARX/runs/arx_grasp/tabletop_catalog.json')
        if catalog_path.exists():
            from pxr import UsdGeom,Gf
            for i,asset in enumerate(json.loads(catalog_path.read_text())):
                prim=stage.DefinePrim('/World/Task/assets/asset_'+asset['name'],'Xform')
                prim.GetReferences().AddReference(asset['container_path'])
                UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(10+i*.4,10,asset['size_m'][2]/2+.01))
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith('/World/Robot') and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        return stage
    stage_loader.load_composed_stage=load
    original_loop=simulation_loop.run_simulation_loop
    def loop(context,stop_event,**kwargs):
        bridge=ValidationBridge(context)
        # Update targets at the fixed physics rate even when RTX renders less
        # often. Keep the outer hook to service requests when physics is paused.
        context.world.add_physics_callback('arx_validation_control',lambda dt: bridge.before_step(False))
        step=context.world.step
        def wrapped_step(*args,**kw):
            bridge.before_step()
            result=step(*args,**kw)
            bridge.after_step()
            return result
        context.world.step=wrapped_step
        return original_loop(context,stop_event,**kwargs)
    simulation_loop.run_simulation_loop=loop
    from arx_sim.inprocess_app import main as run
    return run()


if __name__=='__main__':
    raise SystemExit(main())
