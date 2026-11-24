"""Reloadable, simulation-only grasp diagnostic actions."""
def handle(bridge,request):
    if request['action'].startswith('grasp_lilo_'):
        from arx_sim.diagnostics import lilo_observation
        return lilo_observation.handle(bridge,request)
    if request['action']=='grasp_research_setup':
        from arx_sim.diagnostics.research_setup import begin
        return begin(bridge,request)
    if request['action']=='grasp_research_fingerprint':
        import importlib
        from arx_sim.diagnostics import research_fingerprint
        return importlib.reload(research_fingerprint).capture(bridge)
    if request['action']=='grasp_feedback_camera':
        if bridge.active is not None or bridge.context.controller.active is not None:
            raise RuntimeError('Cannot change feedback cameras during an active command')
        c=bridge.context
        role='right_wrist'
        enabled=bool(request.get('enabled',True))
        owned=bool(getattr(bridge,'feedback_camera_owned',False))
        if enabled and role not in c.cameras:
            from arx_sim.sensors.camera import IsaacRgbdCamera
            cfg=c.config['controllers']['camera'];sim=c.config['simulation']
            c.cameras[role]=IsaacRgbdCamera(
                '/World/COOLASSensors/'+role,tuple(cfg['resolution']),
                float(sim.get('camera_tick_hz',5)),int(sim.get('active_gpu',0)),
                c.camera_prims[role],cfg['definitions'][role].get('color_intrinsics'))
            bridge.feedback_camera_owned=True
            owned=True
        elif not enabled and owned and role in c.cameras:
            c.cameras.pop(role).destroy()
            c.controller._camera_errors.pop(role,None)
            bridge.feedback_camera_owned=False
            owned=False
        return {'camera_id':role,'serial':c.config['controllers']['camera']['definitions'][role]['serial'],
                'enabled':role in c.cameras,'owned_by_feedback':owned}
    if request['action'] in ('grasp_asset_load','grasp_asset_bind','grasp_video_start','grasp_video_stop','grasp_viewport_state','grasp_asset_info'):
        import importlib
        importlib.invalidate_caches()
        from arx_sim.diagnostics import tabletop_actions
        return importlib.reload(tabletop_actions).handle(bridge,request)
    if request['action']=='grasp_arm_gains':
        if bridge.active is not None:raise RuntimeError('Trajectory is active')
        stiffness=float(request['stiffness']);damping=float(request['damping'])
        if (stiffness,damping) not in ((20000.,800.),(80000.,1600.),(120000.,2000.)):raise ValueError('Unsupported diagnostic gains')
        c=bridge.context
        c.robot.set_dof_gains(stiffnesses=stiffness,dampings=damping,dof_indices=c.controller.arm_indices)
        return {'stiffness':stiffness,'damping':damping}
    if request['action']=='grasp_solver_iterations':
        if bridge.active is not None:raise RuntimeError('Trajectory is active')
        counts=int(request['position_counts'])
        if counts not in (16,32,64):raise ValueError('Expected 16, 32 or 64 position iterations')
        bridge.context.robot.set_solver_iteration_counts(position_counts=counts,velocity_counts=4)
        return {'position_counts':counts,'velocity_counts':4}
    if request['action']=='grasp_control_probe':
        c=bridge.context
        indices=c.controller.arm_indices
        def values(value):
            if isinstance(value,tuple):return [values(v) for v in value]
            return value.numpy().tolist()
        result={'snapshot':bridge.snapshot()}
        for method in ('get_dof_gains','get_dof_max_efforts','get_dof_positions','get_dof_position_targets','get_dof_velocities','get_dof_efforts','get_dof_gravity_compensation_forces'):
            try:result[method]=values(getattr(c.robot,method)(dof_indices=indices))
            except Exception as exc:result[method]={'error':str(exc)}
        return result
    if request['action']=='grasp_reset_setup':
        import json
        from pathlib import Path
        import numpy as np
        from arx_sim.diagnostics import curobo_runner
        import importlib
        if bridge.active is not None:
            raise RuntimeError('Cannot reset while a trajectory is active')
        module=importlib.reload(curobo_runner)
        bridge.__class__=module.ValidationBridge
        saved=json.loads((bridge.root.parents[1]/'arx_grasp/capture_snapshot.json').read_text())
        c=bridge.context
        limits=np.stack([v.numpy()[0] for v in c.robot.get_dof_limits()],axis=-1)
        q=np.clip(np.array(saved['joint_positions']),limits[:,0],limits[:,1])
        c.robot.set_dof_positions(q)
        c.robot.set_dof_position_targets(q)
        c.robot.set_dof_velocities(np.zeros_like(q))
        for name,entry in saved['objects'].items():
            if name not in bridge.objects:continue
            bridge.objects[name].set_world_poses(positions=np.array(entry['world'])[:3,3][None],orientations=np.array([[1.,0.,0.,0.]]))
            bridge.objects[name].set_velocities(linear_velocities=np.zeros((1,3)),angular_velocities=np.zeros((1,3)))
        return {'reset':True}
    if request['action']=='grasp_enable_physics_control':
        world=bridge.context.world
        import importlib
        from arx_sim.diagnostics import curobo_runner
        bridge.__class__=importlib.reload(curobo_runner).ValidationBridge
        if world.physics_callback_exists('arx_validation_control'):
            world.remove_physics_callback('arx_validation_control')
        world.add_physics_callback('arx_validation_control',lambda dt: bridge.before_step(False))
        return {'physics_dt':world.get_physics_dt(),'rendering_dt':world.get_rendering_dt()}
    if request['action']=='grasp_camera_probe':
        result={}
        for name,camera in bridge.context.cameras.items():
            try:
                camera.refresh(bridge.context.world.current_time)
                result[name]={'frame_time':camera.frame_time}
            except Exception:
                import traceback
                result[name]={'error':traceback.format_exc()}
        return result
    if request['action']=='grasp_viewport':
        import importlib
        from pathlib import Path
        import yaml
        from arx_sim.runtime import scene_cameras
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
        cameras=importlib.reload(scene_cameras)
        config_path=Path(__file__).resolve().parents[2]/'config/runtime.yaml'
        spec=yaml.safe_load(config_path.read_text())['simulation']['scene_cameras']
        cameras.create_scene_cameras(bridge.context.stage,bridge.context.config['simulation']['articulation_root'],spec)
        active=cameras.activate_scene_camera(spec)
        viewport=get_active_viewport()
        if viewport is not None and request.get('capture_path'):
            capture_viewport_to_file(viewport,request['capture_path'])
        from pxr import UsdGeom
        cache=UsdGeom.XformCache()
        matrix=cache.GetLocalToWorldTransform(bridge.context.stage.GetPrimAtPath(str(viewport.camera_path)))
        return {'active':active,'camera_path':str(viewport.camera_path) if viewport else None,'matrix':str(matrix),
                'robot_matrix':str(cache.GetLocalToWorldTransform(bridge.context.stage.GetPrimAtPath(bridge.context.config['simulation']['articulation_root'])))}
    raise ValueError('Unknown grasp action')
