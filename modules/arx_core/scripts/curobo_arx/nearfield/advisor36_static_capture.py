"""Independent ARX static observations, explicitly not physical trajectories.

Only joint STATE setters are used. Original drives, limits, camera mounting,
colliders, object pose and physics remain unchanged. A unique sample identifier
binds each render; separate static samples may share one frozen physics time.
"""
import copy
import numpy as np
from scripts.curobo_arx.data_factory.common import digest
from .capture import SynchronizedCapture, AXES, validate_bundle, array_data, CaptureUnavailable


class RenderCameraPrim:
    """Read the unchanged mounted camera from the live render hierarchy.

    Tensor articulation setters update PhysX/Fabric, but USD may retain the
    last integrated state. Only camera READS use Fabric; rigid-link and joint
    readback still uses the original physics tensor backend.
    """
    def __init__(self, source):
        self.source = source

    def __getattr__(self, name):
        return getattr(self.source, name)

    def get_world_poses(self, **kwargs):
        from isaacsim.core.experimental.utils.backend import use_backend
        with use_backend('fabric', raise_on_unsupported=True, raise_on_fallback=True):
            return self.source.get_world_poses(**kwargs)


class StaticCapture(SynchronizedCapture):
    def __init__(self, bridge):
        super().__init__(bridge, render_every_update=True)
        self.published = set()
        self.frozen_mount = None
        self.fabric = None
        self.render_binding = None

    def bind_mount(self, reference_snapshot):
        mount=np.linalg.inv(reference_snapshot['link_world']['right_eef_link'])@np.asarray(reference_snapshot['camera_world']['right_wrist'])
        if self.frozen_mount is not None and not np.allclose(self.frozen_mount,mount,atol=1e-5,rtol=0):
            raise RuntimeError('Historical wrist mounting differs between released objects')
        self.frozen_mount=mount
        if self.fabric is None:
            import carb
            import omni.usd
            from isaacsim.core.experimental.utils import app as app_utils
            settings=carb.settings.get_settings()
            keys=('/physics/updateToUsd','/physics/updateVelocitiesToUsd',
                  '/physics/updateParticlesToUsd','/physics/updateForceSensorsToUsd')
            before={k:settings.get(k) for k in keys}
            # The historical ARX World uses CPU dynamics. Unlike GPU IsaacLab
            # environments it does not attach PhysX Fabric output automatically.
            # Attach ONLY the render-output interface to the existing stage;
            # never reattach PhysX, change its device, or alter scene physics.
            app_utils.enable_extension('omni.physx.fabric')
            from omni.physxfabric import get_physx_fabric_interface
            self.fabric=get_physx_fabric_interface()
            stage_id=omni.usd.get_context().get_stage_id()
            self.fabric.attach_stage(stage_id)
            for key,value in before.items():
                if value is not None and settings.get(key)!=value:settings.set(key,value)
            self.fabric.force_update(0.,float(self.bridge.context.world.current_time))
            self.render_binding={'stage_id':int(stage_id),'interface':'PhysXFabric',
                'original_physics_output_settings':before,
                'current_physics_output_settings':{k:settings.get(k) for k in keys},
                'fabric_scene_delegate':settings.get('/app/useFabricSceneDelegate'),
                'physics_device_and_scene_unchanged':True,'physics_stage_reattached':False}
        # No USD edits and no new mount/calibration. The separate RTX capture
        # camera continues to copy the actual mounted camera as before.
        context=self.bridge.context
        for role, source in list(context.camera_prims.items()):
            if not isinstance(source, RenderCameraPrim):
                reader=RenderCameraPrim(source)
                context.camera_prims[role]=reader
                if role in context.cameras:
                    context.cameras[role]._source_prim=reader

    def set_joint_state(self, sample_id, joint_names, q):
        from isaacsim.core.simulation_manager import SimulationManager
        b = self.bridge
        if sample_id in self.published: raise RuntimeError('Static sample already published')
        if b.active or b.fault or b.context.controller.active:
            raise RuntimeError('Static state setter requires idle healthy ARX')
        before = b.snapshot(); q = np.asarray(q, dtype=float)
        if list(before['joint_names']) != list(joint_names): raise ValueError('ARX joint order mismatch')
        limits = np.asarray(before['joint_limits'])
        if q.shape != (len(joint_names),) or not np.isfinite(q).all(): raise ValueError('Invalid joint state')
        if np.max(np.maximum(limits[:,0]-q, q-limits[:,1])) > 1e-6:
            raise ValueError('Static joint state outside unchanged ARX limits')
        robot = b.context.robot
        robot.set_dof_positions(q)
        robot.set_dof_position_targets(q)
        robot.set_dof_velocities(np.zeros_like(q))
        robot.set_dof_velocity_targets(np.zeros_like(q))
        simulation = SimulationManager.get_physics_simulation_view()
        if simulation is None: raise RuntimeError('Missing live ARX physics view')
        simulation.update_articulations_kinematic()
        self.fabric.force_update(0.,float(b.context.world.current_time))
        # Isaac6 World.render calls PhysXFabric.force_update even without an
        # integrated physics tick. Read the same Fabric hierarchy as rendering,
        # rather than USD's previous integrated camera-child transform.
        b.context.world.render()
        after = b.snapshot()
        if self.frozen_mount is None:raise RuntimeError('Static camera mounting has no historical binding')
        mount=np.linalg.inv(after['link_world']['right_eef_link'])@np.asarray(after['camera_world']['right_wrist'])
        if not np.allclose(mount,self.frozen_mount,atol=1e-5,rtol=0):
            error=RuntimeError('Camera child did not follow actual ARX FK after static joint setting')
            error.static_diagnostics={'sample_id':sample_id,'before':before,'after':after,
                'frozen_mount':self.frozen_mount.tolist(),'actual_mount':mount.tolist(),
                'render_binding':self.render_binding,
                'maximum_matrix_error':float(np.max(np.abs(mount-self.frozen_mount)))}
            raise error
        if not np.allclose(after['joint_positions'], q, atol=2e-7, rtol=0):
            raise RuntimeError('ARX static joint setter readback mismatch')
        if np.max(np.abs(after['joint_velocities'])) > 1e-7:
            raise RuntimeError('Static state did not have zero joint velocities')
        if after['sim_time'] != before['sim_time']: raise RuntimeError('Static setter advanced physics')
        for name, obj in before['objects'].items():
            if not np.allclose(obj['world'], after['objects'][name]['world'], atol=1e-8, rtol=0):
                raise RuntimeError('Static setter moved object')
        return {'sample_id':sample_id, 'joint_names':list(joint_names), 'requested_joint_positions':q.tolist(),
            'before':before, 'after':after, 'physical_trajectory_executed':False,
            'frozen_eef_from_camera':self.frozen_mount.tolist(),'actual_eef_from_camera':mount.tolist(),
            'pre_capture_transform_refresh_renders':1,
            'API':'Articulation.set_dof_positions + SimulationView.update_articulations_kinematic + World.render/PhysXFabric.force_update',
            'camera_pose_read_backend':'fabric; source hierarchy unchanged',
            'render_binding':self.render_binding,
            'physics_time_held_fixed':True}

    def capture_static(self, setter, max_renders=6):
        b=self.bridge; world=b.context.world; t=float(world.current_time)
        after=b.snapshot(); sample_id=setter['sample_id']
        if digest(after)!=digest(setter['after']):
            # Camera frame caches can be refreshed separately; physical state
            # must still match exactly before the render below.
            for key in ('joint_positions','joint_velocities','link_world','objects','sim_time'):
                if after[key]!=setter['after'][key]: raise RuntimeError('Static state changed before capture')
        if sample_id in self.published: raise RuntimeError('Repeated static observation')
        self.camera.sync_pose(force=True)
        expected=np.asarray(after['camera_world']['right_wrist'])@AXES
        expected[:3,3]+=b.roles.origin
        target=after['objects']['test_object']
        roles={'target':target.get('raw_path',target['path']),
               **{n:b.links[n].paths[0] for n in ('right_link27','right_link28')}}
        errors=[]
        for count in range(1,max_renders+1):
            world.render()
            if float(world.current_time)!=t: raise RuntimeError('Static render advanced frozen physics time')
            if count<2: continue
            try:
                ref0=copy.deepcopy(self.annotators['reference'].get_data())
                rgb=array_data(self.annotators['rgb'].get_data())
                depth=array_data(self.annotators['depth'].get_data()).squeeze()
                instance=self.annotators['instance'].get_data()
                ids=np.array(instance['data'],copy=True).squeeze()
                info=copy.deepcopy(instance['info'])
                params=copy.deepcopy(self.annotators['params'].get_data())
                ref1=copy.deepcopy(self.annotators['reference'].get_data())
                bundle=validate_bundle(rgb,depth,ids,info,ref0,ref1,t,params,expected,roles,b.environment_root)
            except (ValueError,KeyError,TypeError,np.linalg.LinAlgError) as error:
                errors.append({'render':count,'error':str(error)}); continue
            final=b.snapshot()
            for key in ('joint_positions','joint_velocities','link_world','objects','sim_time'):
                if final[key]!=after[key]: raise RuntimeError('Physical state changed during static render')
            self.sequence+=1; self.published.add(sample_id)
            bundle['metadata'].update(sequence=self.sequence,render_product=str(self.product),
                render_schedule=self.render_schedule,render_only_updates=count,
                physics_time_before_render=t,physics_time_after_render=t,
                before_physics=setter['before'],pose_sync_state=after,after_render=final,
                acquisition_mode='independent_static_joint_state_v1',sample_id=sample_id,
                static_setter_sha256=digest(setter),physics_tick_reuse_is_explicit_static_sampling=True,
                physical_stationary_dwell_measured=False,physical_trajectory_executed=False)
            c=self.camera; c._latest_rgb,c._latest_depth=bundle['rgb'],bundle['depth']
            c.frame_time=t; c.capture_metadata,c.capture_arrays=bundle['metadata'],bundle['arrays']
            return bundle
        raise CaptureUnavailable('Static ARX render did not synchronize',errors)
