"""Isaac main-thread ARX environments; one World step advances all clones.

Imports Isaac only inside runtime constructors. Historical control/trajectory
code is reused; no mailbox or global test_object is shared between environments.
"""
import copy
import time
from pathlib import Path
import numpy as np
from .common import ROOT, read, write
from .geometry import Roles, translated


def set_initial_asset_translation(prim,position):
    """Normal placement, retaining existing authored rotation/scale operations."""
    from pxr import Gf,UsdGeom
    transform=UsdGeom.Xformable(prim)
    op=next((v for v in transform.GetOrderedXformOps() if v.GetOpName()=='xformOp:translate' and not v.IsInverseOp()),None)
    if op is None:op=transform.AddTranslateOp()
    cls={UsdGeom.XformOp.PrecisionDouble:Gf.Vec3d,UsdGeom.XformOp.PrecisionFloat:Gf.Vec3f,
        UsdGeom.XformOp.PrecisionHalf:Gf.Vec3h}[op.GetPrecision()]
    op.Set(cls(*position))


class ScopedStage:
    def __init__(self, stage, root, include_globals=False):
        self.stage, self.root, self.include_globals = stage, root, include_globals

    def Traverse(self):
        from pxr import Usd
        local = list(Usd.PrimRange(self.stage.GetPrimAtPath(self.root)))
        if self.include_globals:
            local += [p for p in self.stage.Traverse() if str(p.GetPath()).startswith(('/World/PhysicsMaterials/','/OmniverseKit_')) or str(p.GetPath()) in ('/physicsScene','/World')]
        return local

    def __getattr__(self, name):
        return getattr(self.stage, name)


def bridge_class():
    from arx_sim.diagnostics.curobo_runner import ValidationBridge

    class EnvironmentBridge(ValidationBridge):
        def __init__(self, context, root, origin, mailbox, assets):
            from isaacsim.core.experimental.prims import RigidPrim
            self.context, self.environment_root = context, root
            self.roles = Roles(root, origin)
            self.root = Path(mailbox)
            self.root.mkdir(parents=True, exist_ok=False)
            self.links = {name: RigidPrim(context.config['simulation']['articulation_root']+'/'+name) for name in context.robot.link_names}
            self.objects = {name: RigidPrim(root+'/Task/'+name) for name in ('red_cube','blue_cylinder','green_sphere')}
            self.tabletop_objects = {asset['object_id']: RigidPrim(root+'/Task/assets/asset_'+asset['object_id']) for asset in assets if asset['available']}
            self.contacts, self.records, self.active = [], [], None
            self.fault, self.raw_snapshot = None, False
            self.initial = read(ROOT/'configs/lilo_vla/fixtures/tabletop_rt_v1/initial_setup.json')

        def snapshot(self):
            result = super().snapshot()
            result['joint_velocities'] = self.context.robot.get_dof_velocities().numpy()[0].tolist()
            result['environment_root'] = self.environment_root
            if not self.raw_snapshot:
                for name, value in result['link_world'].items():
                    result['link_world'][name] = translated(value, self.roles.origin, True).tolist()
                for name, value in result['camera_world'].items():
                    result['camera_world'][name] = translated(value, self.roles.origin, True).tolist()
                for value in result['objects'].values():
                    value['raw_path'] = value['path']
                    value['path'] = self.roles.path(value['path'])
                    value['world'] = translated(value['world'], self.roles.origin, True).tolist()
            return result

        def contact(self, headers, data):
            from pxr import PhysicsSchemaTools
            for header in headers:
                a, b = (str(PhysicsSchemaTools.intToSdfPath(v)) for v in (header.actor0,header.actor1))
                if not (self.roles.owns(a) or self.roles.owns(b)):
                    continue
                if not (self.roles.owns(a) and self.roles.owns(b)):
                    self.fault = {'error':'cross_environment_contact', 'a':a,'b':b}
                    continue
                if self.active and any('/right_link' in p or '/right_eef_link' in p for p in (a,b)):
                    self.contacts.append(self.roles.event({'a':a,'b':b,'num_contacts':int(header.num_contact_data),
                        'type':int(header.type),'sim_time':self.context.world.current_time}))

        def reset_episode(self, asset):
            if self.active or self.context.controller.active:
                raise RuntimeError('Cannot reset an active ARX environment')
            c = self.context
            if self.initial['joint_names'] != list(c.robot.dof_names):
                raise ValueError('Historical ARX DOF order changed')
            bounds = np.stack([v.numpy()[0] for v in c.robot.get_dof_limits()],axis=-1)
            q = np.clip(np.asarray(self.initial['joint_positions']),bounds[:,0],bounds[:,1])
            c.robot.set_dof_positions(q)
            c.robot.set_dof_position_targets(q)
            c.robot.set_dof_velocities(np.zeros_like(q))
            c.robot.set_dof_velocity_targets(np.zeros_like(q))
            initial = {}
            def place(name, obj, xyz, quat):
                obj.set_world_poses(positions=(np.asarray(xyz)+self.roles.origin)[None],orientations=np.asarray(quat)[None])
                obj.set_velocities(linear_velocities=np.zeros((1,3)),angular_velocities=np.zeros((1,3)))
                initial[name] = {'path':obj.paths[0], 'position':list(map(float,xyz)), 'quaternion_wxyz':list(map(float,quat))}
            for name, xyz in [('blue_cylinder',[4.33,1.79,.76]),('green_sphere',[4.46,1.96,.76]),('red_cube',[4.22,1.75,.73])]:
                place(name,self.objects[name],xyz,[1.,0.,0.,0.])
            for i, (name,obj) in enumerate(self.tabletop_objects.items()):
                place('asset:'+name,obj,[10+i*.4,10,.4],[1.,0.,0.,0.])
            selected = self.tabletop_objects[asset['object_id']]
            place('asset:'+asset['object_id'],selected,asset['placement']['position'],asset['placement']['orientation'])
            self.objects['test_object'] = selected
            self.test_object_size = asset['size_m']
            return {'joint_names':list(c.robot.dof_names),'joint_positions':q.tolist(),'objects':initial,
                    'zero_velocities':True,'reset_sim_time':c.world.current_time}

        def export_scene(self):
            if getattr(self,'scene_transport',None)=='arx_geometry_files_live_pose_v1':
                from scripts.curobo_arx.sdg_stage1.scene_transport import export_scene
                return export_scene(self)
            c = self.context
            stage = c.stage
            try:
                c.stage = ScopedStage(stage,self.environment_root)
                self.raw_snapshot = True
                super().request({'action':'export_scene'})
            finally:
                c.stage = stage
                self.raw_snapshot = False
            data = read(self.root/'scene_meshes.json')
            for mesh in data['meshes']:
                mesh['raw_path'] = mesh['path']
                mesh['path'] = self.roles.path(mesh['path'])
            data['base_world'] = translated(data['base_world'],self.roles.origin,True).tolist()
            # Keep the original raw export; the caller persists the canonical
            # version once. Rewriting hundreds of MB here stalls all environments.
            return data

        def command(self, request):
            if self.fault:
                raise RuntimeError(str(self.fault))
            if super().request({'action':'trajectory',**request}) is not None:
                raise RuntimeError('Expected nonblocking trajectory dispatch')

        def result(self):
            if self.active:
                return None
            path = self.root/'response.json'
            if not path.is_file():
                return None
            data = read(path)
            path.unlink()
            if not data['ok']:
                raise RuntimeError(data)
            result = data['result']
            for row in result['records']:
                row['raw_ee_world'] = row['ee_world']
                row['ee_world'] = translated(row['ee_world'],self.roles.origin,True).tolist()
                row['objects'] = {n:translated(v,self.roles.origin,True).tolist() for n,v in row['objects'].items()}
            return result

        def observation(self, env_id, attempt_id, object_id, goal, command=None, stage=None, camera_role='right_wrist'):
            from pxr import UsdGeom
            import sys
            sys.path.insert(0,str(ROOT/'scripts/curobo_arx/lilo_vla'))
            from observations import encode_state
            c = self.context
            camera = c.cameras[camera_role]
            rgb, depth = camera.get_rgba(),camera.get_depth()
            if rgb is None or depth is None or not np.any(rgb) or camera.frame_time is None:
                raise RuntimeError('Calibrated RGB-D camera unavailable')
            snapshot = self.snapshot()
            if abs(snapshot['sim_time']-camera.frame_time)>1e-4:
                raise RuntimeError('Camera/ARX state not from the same rendered tick')
            view = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(c.stage.GetPrimAtPath(camera.prim_path))).T
            optical = translated(view @ np.diag([1,-1,-1,1]),self.roles.origin,True)
            row = {'version':'arx_data_factory_v1','env_id':env_id,'attempt_id':attempt_id,'object_id':object_id,
                   'environment_root':self.environment_root,'env_origin':self.roles.origin.tolist(),
                   'snapshot':snapshot,'frame_time':camera.frame_time,'capture_sim_time':snapshot['sim_time'],
                   'camera_role':camera_role,'camera_owner_root':self.environment_root,
                   'camera_source_path':c.camera_prims[camera_role].paths[0],
                   'camera_world_optical':optical.tolist(),'intrinsics':camera.get_intrinsics_matrix().tolist(),
                   'camera_clipping_range_m':list(UsdGeom.Camera(c.stage.GetPrimAtPath(camera.prim_path)).GetClippingRangeAttr().Get()),
                   'stage':stage,'teacher_command':command,'wall_time':time.time()}
            if getattr(camera, 'capture_metadata', None):
                row['capture'] = copy.deepcopy(camera.capture_metadata)
            if goal is not None:
                row['state'] = encode_state(snapshot,'grasp_goal_world_error_v1',goal).tolist()
            return row,rgb,depth

    return EnvironmentBridge


def create_group(registration, renderer, count, out):
    """Construct one native multi-environment World with unchanged ARX settings."""
    import sys
    def progress(phase):
        write(Path(out)/'initialization.json',{'phase':phase,'wall':time.time(),'env_count':count})
        print('ARX_ENV_INIT',phase,flush=True)
    progress('starting_simulation_app')
    for path in ('shared','arx-ops','arx-sim'):
        sys.path.insert(0,str(ROOT/path))
    from isaacsim import SimulationApp
    config = copy.deepcopy(registration['render_configs'][renderer])
    sim = config['simulation']
    sim['active_gpu'],sim['physics_gpu'] = 0,0
    app = SimulationApp({'headless':True,'hide_ui':False,'disable_viewport_updates':True,'renderer':sim['renderer'],
        'anti_aliasing':int(sim.get('anti_aliasing',3)),'denoiser':bool(sim.get('denoiser',True)),
        'samples_per_pixel_per_frame':int(sim.get('samples_per_pixel_per_frame',64)),
        'multi_gpu':False,'active_gpu':0,'physics_gpu':0,'fast_shutdown':True,
        'extra_args':['--/app/mdl/additionalSystemPaths=["/isaac-sim/kit/mdl/core/VRay"]']})
    from isaacsim.core.experimental.utils import app as app_utils
    app_utils.enable_extension('isaacsim.core.cloner')
    from isaacsim.core.api import World
    from isaacsim.core.cloner import Cloner
    from isaacsim.core.experimental.prims import Articulation,RigidPrim,XformPrim
    from pxr import Gf,Sdf,PhysxSchema,Usd,UsdGeom,UsdPhysics,UsdLux
    from arx_sim.inprocess_app import _apply_gripper_contact_overrides,_apply_camera_definition,_apply_task_initial_joint_positions
    from arx_sim.runtime.stage_loader import load_composed_stage
    from arx_sim.runtime.scene_cameras import disable_imported_cameras,demote_calibration_cameras,create_scene_cameras
    from arx_sim.runtime.determinism import configure_determinism
    from arx_sim.runtime.rendering import configure_camera_antialiasing
    from arx_sim.backend.context import IsaacContext
    from arx_sim.backend.command_queue import CommandQueue
    from arx_sim.robot.controller import IsaacRobotController
    from arx_sim.sensors.camera import IsaacRgbdCamera
    configure_determinism(int(sim['seed']))
    progress('composing_frozen_scene')
    stage = load_composed_stage(app,config)
    progress('authoring_original_contact_properties')
    articulation_path = sim['articulation_root']
    physics = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(articulation_path))
    physics.CreateEnabledSelfCollisionsAttr().Set(bool(sim.get('enable_self_collisions',False)))
    physics.CreateSolverPositionIterationCountAttr().Set(int(sim.get('solver_position_iteration_count',16)))
    physics.CreateSolverVelocityIterationCountAttr().Set(int(sim.get('solver_velocity_iteration_count',4)))
    for prim in Usd.PrimRange(stage.GetPrimAtPath(articulation_path)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(False)
    _apply_gripper_contact_overrides(stage,config)
    assets = registration['catalog']['objects']
    for asset in assets:
        if not asset['available']:
            # Missing historical assets cannot be silently omitted from the scene.
            if asset['catalog_index'] < registration['catalog']['initial_count']:
                raise RuntimeError('Cannot instantiate frozen historical catalog: '+asset['object_id'])
            continue
        prim = stage.DefinePrim('/World/Task/assets/asset_'+asset['object_id'],'Xform')
        prim.GetReferences().AddReference(asset['usd_path'])
        set_initial_asset_translation(prim,[10+asset['catalog_index']*.4,10,asset['size_m'][2]/2+.01])
    progress('cloning_arx_environments')
    disable_imported_cameras(stage,('/World/Environment','/World/Robot'))
    camera_cfg = config['controllers']['camera']
    for role,relative in camera_cfg['roles'].items():
        _apply_camera_definition(stage,articulation_path+'/'+relative,camera_cfg['definitions'][role])
    overrides=registration.get('render_camera_clipping',{})
    if overrides:
        if set(overrides)!={'right_wrist'}:raise ValueError('Only the registered new-teacher right wrist near clip may change')
        from .camera_render import correct_wrist_near_clip
        receipt=correct_wrist_near_clip(stage,articulation_path+'/'+camera_cfg['roles']['right_wrist'],overrides['right_wrist'])
        write(Path(out)/'camera_render_correction.json',receipt)
    demote_calibration_cameras(stage,tuple(articulation_path+'/'+p for p in camera_cfg['roles'].values()))
    cloner = Cloner(stage=stage)
    cloner.define_base_env('/World/Envs')
    roots = cloner.generate_paths('/World/Envs/env',count)
    UsdGeom.Xform.Define(stage,roots[0])
    edits = Sdf.BatchNamespaceEdit()
    for name in ('Environment','Robot','Task'):
        edits.Add('/World/'+name,roots[0]+'/'+name)
    if not stage.GetRootLayer().Apply(edits):
        raise RuntimeError('Cannot namespace canonical ARX scene for independent clone ownership')
    origins = np.asarray([[100.*i,0.,0.] for i in range(count)])
    cloner.clone(roots[0],roots,positions=origins,replicate_physics=False,copy_from_source=True)
    lights=[]
    for root in roots:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if prim.HasAPI(UsdLux.LightAPI):
                links=UsdLux.LightAPI(prim).GetLightLinkCollectionAPI()
                if links.GetIncludeRootAttr().Get():
                    links.CreateIncludeRootAttr().Set(False)
                    links.CreateIncludesRel().SetTargets([Sdf.Path(root)])
                lights.append({'path':str(prim.GetPath()),'owner':root,
                    'include_root':links.GetIncludeRootAttr().Get(),
                    'includes':[str(p) for p in links.GetIncludesRel().GetTargets()]})
    write(Path(out)/'light_isolation.json',{'lights':lights,
        'scope':'Prevent replicated dome/distant lights from illuminating other clones; preserve existing within-environment link exclusions and optical settings'})
    world = World(physics_dt=float(sim['physics_dt']),rendering_dt=float(sim['rendering_dt']),stage_units_in_meters=1.)
    physics_scenes = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(physics_scenes)!=1:
        raise RuntimeError('Expected exactly one frozen physics scene')
    # Native filter_collisions flips a global PhysX scene setting. Use ordinary
    # cross-group exclusions instead, retaining the historical scene settings.
    from .isolation import isolate
    isolation=isolate(stage,roots)
    write(Path(out)/'collision_isolation.json',isolation)
    robots = [Articulation(root+articulation_path.removeprefix('/World')) for root in roots]
    world.reset()  # The only whole-world reset, before any production claim.
    progress('initializing_controllers_and_calibrated_cameras')
    for robot in robots:
        _apply_task_initial_joint_positions(robot,config)
    world.step(render=False)
    configure_camera_antialiasing(app,int(sim.get('anti_aliasing',3)))
    contexts,bridges = [],[]
    Bridge = bridge_class()
    for i,(root,origin,robot) in enumerate(zip(roots,origins,robots)):
        cfg = copy.deepcopy(config)
        cfg['simulation']['articulation_root'] = root+articulation_path.removeprefix('/World')
        cfg['simulation']['robot_prim'] = root+'/Robot'
        cfg['simulation']['environment_prim'] = root+'/Environment'
        cfg['simulation']['camera_roles'] = ['head','right_wrist']
        frames = cfg['controllers']['frames']
        ap = cfg['simulation']['articulation_root']
        rigid_at = lambda key: RigidPrim(ap+'/'+frames[key])
        camera_prims = {role:XformPrim(ap+'/'+relative) for role,relative in camera_cfg['roles'].items()}
        cameras = {role:IsaacRgbdCamera('/World/COOLASSensors/env_'+str(i)+'/'+role,
                    tuple(camera_cfg['resolution']),float(sim.get('camera_tick_hz',30)),0,camera_prims[role],
                    camera_cfg['definitions'][role].get('color_intrinsics')) for role in ('head','right_wrist')}
        context = IsaacContext(simulation_app=app,world=world,stage=stage,robot=robot,
            base_prim=rigid_at('base_link'),lift_prim=rigid_at('lift_link'),arm_base_prim=rigid_at('arm_base_link'),
            ee_prim=rigid_at('ee_link'),left_ee_prim=RigidPrim(ap+'/'+cfg['controllers']['left_arm']['ee_link']),
            tcp_prim=rigid_at('gripper_tcp'),camera_prims=camera_prims,cameras=cameras,
            config=cfg,command_queue=CommandQueue(capacity=int(sim['queue_capacity'])))
        context.controller = IsaacRobotController(context)
        contexts.append(context)
        bridges.append(Bridge(context,root,origin,Path(out)/f'env_{i}'/'mailbox',assets))
        bridges[-1].collision_isolation=isolation[root]
        bridges[-1].additional_asset_contracts={a['object_id']:a['physics_contract'] for a in assets
            if a['catalog_index']>=registration['catalog']['initial_count'] and a.get('available') and a.get('physics_contract')}
        bridges[-1].scene_transport=registration.get('scene_transport')
    from omni.physx import get_physx_simulation_interface
    subscription = get_physx_simulation_interface().subscribe_contact_report_events(
        lambda headers,data: [bridge.contact(headers,data) for bridge in bridges])
    world.add_physics_callback('data_factory_arx_control',lambda dt:[b.before_step(False) for b in bridges])
    # Avoid repeatedly rendering partially composed USDs during startup only.
    # Restore normal rendering before uniform warmup or any production sample.
    from omni.kit.viewport.utility import get_active_viewport
    viewport=get_active_viewport()
    if viewport is None:raise RuntimeError('Missing viewport while restoring historical render updates')
    viewport.updates_enabled=True
    write(Path(out)/'environment_layout.json',{'roots':roots,'origins':origins.tolist(),
        'cloner':'Isaac Sim installed native Cloner; copy_from_source=True; replicate_physics=False',
        'collision_isolation':'cross-environment groups only; all original intra-environment constraints retained',
        'rendering':'shared World render tick; individual calibrated RTX RGB-D sensors, not tiled mosaic',
        'renderer':renderer,'physics_dt':world.get_physics_dt(),'rendering_dt':world.get_rendering_dt(),
        'startup_viewport_updates':False,'production_viewport_updates':bool(viewport.updates_enabled)})
    progress('ready_for_uniform_warmup')
    return app,world,bridges,subscription
