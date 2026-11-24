"""ARX side-task bridge: moving physical obstacle and receding-horizon commands.

Uses the original scene loader, controllers, physics and recording implementation.
No main-research source is edited. All Isaac accesses stay on its main thread.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

from arx_sim.diagnostics import curobo_runner as original

BaseBridge = original.ValidationBridge


class DynamicBridge(BaseBridge):
    def __init__(self, context):
        from arx_sim.diagnostics.collection_status import identity
        self.collection_identity = identity()
        self.moving = None
        self.streaming = False
        self.command = None
        self.command_id = None
        self.dynamic_contacts = []
        self.trace = None
        self.last_stream_record = -1.
        super().__init__(context)

    def snapshot(self):
        result = super().snapshot()
        c = self.context
        result['joint_velocities'] = c.robot.get_dof_velocities().numpy()[0].tolist()
        result['wall_time'] = time.time()
        if self.moving:
            result['moving_obstacle'] = {
                'path': self.moving['path'], 'dims': self.moving['dims'].tolist(),
                'center': self.obstacle_center().tolist(),
                'velocity': self.moving['velocity'].tolist(),
                'start_sim_time': self.moving['start'],
                'measured_world': c.controller._world_pose(self.moving['prim']).tolist(),
                'renderer_world': self.obstacle_renderer_world(),
                'renderer_scaled_world': self.obstacle_renderer_matrix(),
            }
        result['streaming'] = self.streaming
        result['command_id'] = self.command_id
        # A legacy dynamic_reload cannot establish the startup input identity.
        # Such sessions remain explicitly incompatible until a clean restart.
        if hasattr(self, 'collection_identity'):
            from arx_sim.diagnostics.collection_status import snapshot
            result['collection_status'] = snapshot(self)
        return result

    def obstacle_center(self):
        m = self.moving
        elapsed = max(0., self.context.world.current_time - m['start'])
        return m['center'] + elapsed * m['velocity']

    def obstacle_renderer_world(self):
        from isaacsim.core.experimental.utils.backend import use_backend
        with use_backend('usdrt'):
            return self.context.controller._world_pose(self.moving['prim']).tolist()

    def obstacle_renderer_matrix(self):
        import usdrt
        matrix = self.moving['prim']._get_fabric_hierarchy().get_world_xform(
            usdrt.Sdf.Path(self.moving['path']))
        return np.asarray(matrix).T.tolist()

    def sync_obstacle_renderer(self):
        # Tensor-teleported kinematic bodies need a render-only Fabric update.
        # Set the declared scale explicitly: get_world_poses discards scale and
        # RigidPrim.set_world_poses(usdrt) may otherwise draw a unit-size cube.
        import usdrt
        matrix = usdrt.Gf.Matrix4d().SetScale(usdrt.Gf.Vec3d(*self.moving['dims']))
        matrix.SetTranslateOnly(usdrt.Gf.Vec3d(*self.obstacle_center()))
        self.moving['prim']._get_fabric_hierarchy().set_world_xform(
            usdrt.Sdf.Path(self.moving['path']), matrix)

    def contact(self, headers, data):
        super().contact(headers, data)
        if not self.streaming and not self.moving:
            return
        from pxr import PhysicsSchemaTools
        for h in headers:
            a, b = (str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.actor0, h.actor1))
            right = any(x in a+b for x in ('/right_link', '/right_eef_link'))
            obstacle = '/World/DynamicPickObstacle' in a+b
            if right or obstacle:
                self.dynamic_contacts.append({'a': a, 'b': b, 'type': int(h.type),
                    'num_contacts': int(h.num_contact_data), 'sim_time': self.context.world.current_time})

    def request(self, request):
        action = request['action']
        if action.startswith('grasp_lilo_'):
            from arx_sim.diagnostics import lilo_observation
            return lilo_observation.handle(self, request)
        if action == 'static_view_capture':
            from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
            view = get_active_viewport()
            self.static_capture_future = capture_viewport_to_file(view, request['path'])
            return {'path':request['path'], 'camera_path':str(view.camera_path)}
        if action == 'static_pole_create':
            from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf
            root = '/World/StaticPickPole'
            if self.active or self.streaming or self.moving or self.context.stage.GetPrimAtPath(root).IsValid():
                raise RuntimeError('Pole requires a fresh idle simulator')
            center = np.asarray(request['center'], dtype=float)
            height, radius = float(request['height']), float(request['radius'])
            if center.shape != (3,) or not np.isfinite(center).all() or not .01 <= radius <= .025 or not .8 <= height <= 1.:
                raise ValueError('Invalid registered pole geometry')
            UsdGeom.Xform.Define(self.context.stage, root)
            parts = [('shaft', center, radius, height), ('base', np.array([*center[:2], .0125]), .06, .025)]
            for name, xyz, r, h in parts:
                shape = UsdGeom.Cylinder.Define(self.context.stage, root+'/'+name)
                shape.CreateRadiusAttr(r); shape.CreateHeightAttr(h); shape.CreateAxisAttr('Z')
                shape.CreateExtentAttr([Gf.Vec3f(-r,-r,-h/2),Gf.Vec3f(r,r,h/2)])
                shape.AddTranslateOp().Set(Gf.Vec3d(*xyz))
                shape.CreateDisplayColorAttr([Gf.Vec3f(.95,.55,.08)])
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
                PhysxSchema.PhysxContactReportAPI.Apply(shape.GetPrim()).CreateThresholdAttr(0.)
            return {'path':root,'center':center.tolist(),'radius':radius,'height':height,'base_radius':.06,'base_height':.025,'static':True}
        fixture_roots = [r for r in ('/World/StaticPickRack','/World/StaticPickPole') if self.context.stage.GetPrimAtPath(r).IsValid()]
        if action == 'export_scene' and fixture_roots:
            # Correct primitive export for authorized new fixtures only.
            from pxr import UsdGeom
            super().request(request)
            path = self.root / 'scene_meshes.json'
            data = json.loads(path.read_text())
            data['meshes'] = [m for m in data['meshes'] if not any(m['path'].startswith(r+'/') for r in fixture_roots)]
            inverse = np.linalg.inv(np.asarray(data['base_world']))
            cache = UsdGeom.XformCache()
            for root in fixture_roots:
                for prim in self.context.stage.GetPrimAtPath(root).GetChildren():
                    if prim.IsA(UsdGeom.Cube):
                        h = UsdGeom.Cube(prim).GetSizeAttr().Get()/2.
                        points = np.array([[x,y,z,1.] for x in (-h,h) for y in (-h,h) for z in (-h,h)])
                        faces = [[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]
                    else:
                        cylinder = UsdGeom.Cylinder(prim)
                        n = 128; r = cylinder.GetRadiusAttr().Get()/np.cos(np.pi/n); h = cylinder.GetHeightAttr().Get()/2.
                        angles = np.arange(n)*2*np.pi/n
                        points = np.array([[r*np.cos(a),r*np.sin(a),z,1.] for z in (-h,h) for a in angles]+[[0.,0.,-h,1.],[0.,0.,h,1.]])
                        faces = []
                        for i in range(n):
                            j = (i+1)%n
                            faces += [[i,j,n+j],[i,n+j,n+i],[2*n,j,i],[2*n+1,n+i,n+j]]
                    transform = inverse @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
                    vertices = (points @ transform.T)[:,:3]
                    data['meshes'].append({'path':str(prim.GetPath()),'vertices':vertices.tolist(),'faces':faces})
            original.write_json(path, data)
            return {'mesh_count':len(data['meshes']), 'fixture_export':'exact cubes; 128-sided circumscribed cylinders'}
        if action == 'static_rack_create':
            from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf
            if self.active or self.streaming or self.moving:
                raise RuntimeError('Static rack requires a fresh idle side simulator')
            root = '/World/StaticPickRack'
            if self.context.stage.GetPrimAtPath(root).IsValid():
                raise RuntimeError('Preserve registered rack; restart before a new trial')
            parts = request['parts']
            for part in parts:
                center, dims = np.asarray(part['center']), np.asarray(part['dims'])
                if center.shape != (3,) or dims.shape != (3,) or not np.isfinite([center,dims]).all() or (dims <= 0).any():
                    raise ValueError('Invalid rack geometry')
                if not part['name'].isidentifier():
                    raise ValueError('Invalid part name')
            UsdGeom.Xform.Define(self.context.stage, root)
            for part in parts:
                cube = UsdGeom.Cube.Define(self.context.stage, root+'/'+part['name'])
                cube.CreateSizeAttr(1.)
                cube.AddTranslateOp().Set(Gf.Vec3d(*part['center']))
                cube.AddScaleOp().Set(Gf.Vec3d(*part['dims']))
                cube.CreateDisplayColorAttr([Gf.Vec3f(.95,.55,.08)])
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                PhysxSchema.PhysxContactReportAPI.Apply(cube.GetPrim()).CreateThresholdAttr(0.)
            return {'path':root,'parts':parts,'static':True}
        if action == 'dynamic_reload':
            if self.active or self.streaming:
                raise RuntimeError('Reload requires idle control')
            import importlib.util
            original.ValidationBridge = BaseBridge
            spec = importlib.util.spec_from_file_location('arx_dynamic_bridge_reload', __file__)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.__class__ = module.DynamicBridge
            original.ValidationBridge = module.DynamicBridge
            return {'reloaded': True}
        if action == 'dynamic_setup':
            if self.streaming or (self.moving and np.linalg.norm(self.moving['velocity'])>0):
                raise RuntimeError('Stop dynamic session before resetting')
            from arx_sim.diagnostics.research_setup import begin
            return begin(self, request)
        if action == 'dynamic_obstacle':
            from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf
            from isaacsim.core.experimental.prims import RigidPrim
            if self.active or self.streaming:
                raise RuntimeError('Create obstacle while controllers are idle')
            center, dims, velocity = [np.asarray(request[k], dtype=float) for k in ('center','dims','velocity')]
            if any(a.shape != (3,) or not np.isfinite(a).all() for a in (center,dims,velocity)):
                raise ValueError('Finite 3-vectors required')
            if (dims <= 0).any() or np.linalg.norm(velocity) > .2:
                raise ValueError('Invalid obstacle dimensions/speed')
            if self.moving:
                # Keep the rigid-body topology fixed: deleting a RigidPrim
                # invalidates all Isaac tensor views, including the ARX robot.
                if not np.array_equal(dims,self.moving['dims']):
                    raise ValueError('Existing obstacle dimensions are fixed')
                self.moving.update(center=center,velocity=velocity,
                    start=self.context.world.current_time+float(request.get('delay',0)))
                self.moving['prim'].set_world_poses(positions=center[None],orientations=np.array([[1.,0.,0.,0.]]))
                self.dynamic_contacts=[]
                return self.snapshot()['moving_obstacle']
            path = '/World/DynamicPickObstacle'
            cube = UsdGeom.Cube.Define(self.context.stage, path)
            cube.CreateSizeAttr(1.)
            cube.AddTranslateOp().Set(Gf.Vec3d(*center))
            cube.AddScaleOp().Set(Gf.Vec3d(*dims))
            cube.CreateDisplayColorAttr([Gf.Vec3f(1., .55, .02)])
            prim = cube.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            rb = UsdPhysics.RigidBodyAPI.Apply(prim)
            rb.CreateKinematicEnabledAttr(True)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            self.moving = {'path':path, 'center':center, 'dims':dims, 'velocity':velocity,
                'start':self.context.world.current_time + float(request.get('delay',0)),
                'prim':RigidPrim(path)}
            self.dynamic_contacts = []
            return self.snapshot()['moving_obstacle']
        if action == 'dynamic_obstacle_stop':
            if self.moving:
                self.moving['prim'].set_world_poses(positions=self.obstacle_center()[None], orientations=np.array([[1.,0.,0.,0.]]))
                self.moving['center'] = self.obstacle_center()
                self.moving['velocity'][:] = 0.
                self.moving['start'] = self.context.world.current_time
            return {'stopped': True}
        if action == 'dynamic_obstacle_remove':
            if self.streaming or self.active:
                raise RuntimeError('Remove requires idle control')
            if self.moving:
                # Park only this new test fixture between trials. No original
                # room/robot collider is removed or modified.
                self.moving.update(center=np.array([3.,2.17,1.1]),velocity=np.zeros(3),
                    start=self.context.world.current_time)
                self.moving['prim'].set_world_poses(positions=self.moving['center'][None],orientations=np.array([[1.,0.,0.,0.]]))
            return {'parked':True,'rigid_body_removed':False}
        if action == 'dynamic_stream_start':
            if self.active or self.streaming:
                raise RuntimeError('Controller busy')
            self.streaming = True
            self.command = None
            self.command_id = None
            (self.root/'stream_command.json').unlink(missing_ok=True)
            if request.get('trace_path'):
                path = Path(request['trace_path']); path.parent.mkdir(parents=True, exist_ok=True)
                self.trace = path.open('x')
            return self.snapshot()
        if action == 'dynamic_stream_stop':
            self.streaming = False
            self.command = None
            if self.trace:
                self.trace.close(); self.trace = None
            return {'snapshot':self.snapshot(),'contacts':self.dynamic_contacts}
        if action == 'dynamic_contacts':
            return {'contacts':self.dynamic_contacts}
        if action == 'trajectory' and self.streaming:
            raise RuntimeError('Stop streaming before original trajectory execution')
        # These legacy hot-reload actions replace bridge.__class__; never call them here.
        if action in ('grasp_reset_setup','grasp_enable_physics_control'):
            raise RuntimeError('Use dynamic_setup; preserve dynamic bridge class')
        return super().request(request)

    def before_step(self, poll_requests=True):
        if self.moving:
            self.moving['prim'].set_world_poses(positions=self.obstacle_center()[None],
                orientations=np.array([[1.,0.,0.,0.]]))
        if self.streaming:
            command_path = self.root/'stream_command.json'
            if command_path.exists():
                command = {}
                try:
                    command = json.loads(command_path.read_text())
                    if command['id'] != self.command_id:
                        q = np.asarray(command['position'], dtype=float)
                        ts = np.asarray(command['time_from_start'], dtype=float)
                        c = self.context; idx = c.controller.arm_indices
                        limits = np.stack([v.numpy()[0,idx] for v in c.robot.get_dof_limits()],axis=-1)
                        if command['joint_names'] != c.config['controllers']['right_arm']['joints']:
                            raise ValueError('Wrong ARX joint order')
                        if q.ndim != 2 or q.shape[1] != len(idx) or len(q)<2 or ts.shape != (len(q),):
                            raise ValueError('Bad stream dimensions')
                        if not np.isfinite(q).all() or not np.isfinite(ts).all() or ts[0]!=0 or (np.diff(ts)<=0).any():
                            raise ValueError('Bad stream values/times')
                        if (q<limits[:,0]-1e-6).any() or (q>limits[:,1]+1e-6).any():
                            raise ValueError('Stream exceeds original USD limits')
                        if np.max(np.abs(q[0]-c.controller._positions()[idx]))>.08:
                            raise ValueError('Stream discontinuity from measured state')
                        if not 0 < ts[-1] <= .6:
                            raise ValueError('Stream horizon must be <= .6 seconds')
                        self.command = {'q':q,'times':ts,'start':float(command['sim_time']), 'received':c.world.current_time}
                        self.command_id = command['id']
                        original.write_json(self.root/'stream_ack.json',{'id':command['id'],'ok':True,'sim_time':c.world.current_time})
                except Exception as exc:
                    original.write_json(self.root/'stream_ack.json',{'id':command.get('id'), 'ok':False,'error':str(exc)})
                    self.command = None
            if self.command:
                a = self.command; elapsed = self.context.world.current_time-a['start']
                target = np.array([np.interp(elapsed,a['times'],a['q'][:,i]) for i in range(a['q'].shape[1])])
                self.context.robot.set_dof_position_targets(target,dof_indices=self.context.controller.arm_indices)
        super().before_step(poll_requests)

    def after_step(self):
        if self.moving:
            self.sync_obstacle_renderer()
        super().after_step()
        snap = self.snapshot()
        original.write_json(self.root/'stream_state.json',snap)
        if self.streaming or self.moving:
            original.write_json(self.root/'stream_events.json',{'count':len(self.dynamic_contacts),
                'recent':self.dynamic_contacts[-30:]})
        if self.streaming and self.trace:
            self.trace.write(json.dumps(snap)+'\n'); self.trace.flush()


def main():
    original.ValidationBridge = DynamicBridge
    return original.main()


if __name__ == '__main__':
    raise SystemExit(main())
