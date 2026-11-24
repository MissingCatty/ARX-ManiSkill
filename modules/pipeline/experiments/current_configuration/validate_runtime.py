"""Verify current SAPIEN settings and native cuRobo spheres without a grasp."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from configuration import PROJECT, WORKSPACE, load_current

OUT=Path(__file__).resolve().parent/'records/validation_20261124_01'


def save(name, data):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def simulation():
    from configuration import runtime, read
    config=load_current(verify_dataset=True)
    dataset=Path(config['dataset'])
    entry=read(dataset/'manifest.json')['schedule'][0]
    environment,baseline,run,sapien,torch,make,profile,convert,maps=runtime()
    from camera_imaging import intrinsic_matrix
    env=make(entry['model_id'],entry['scene'])
    try:
        env.reset(seed=entry['seed'])
        # Hold the registered reset state; no approach/close/lift commands.
        for _ in range(40):env._step_action(torch.tensor(env.initial_q[None],device=env.device))
        obs=env.get_obs();state=run.snapshot(env,obs)
        scene=env.sim_config.scene_config
        k=np.asarray(state['intrinsics']);expected=intrinsic_matrix(profile)
        assert np.max(np.abs(k-expected))<1e-4
        assert abs(state['qpos'][state['joint_names'].index('joint4')]-.452)<.001
        assert env.sim_config.sim_freq==120 and env.sim_config.control_freq==40
        assert scene.solver_position_iterations==16 and scene.solver_velocity_iterations==4
        assert abs(scene.contact_offset-.001)<1e-9 and abs(scene.rest_offset)<1e-9
        shapes=env.obj._bodies[0].get_collision_shapes()
        assert shapes and all(abs(s.contact_offset-.001)<1e-8 for s in shapes)
        from head_camera_config import head_configuration
        # runtime() already installed the new mount; compare it to the saved matrix.
        mounted=np.asarray(environment.CURRENT_CONTROLLERS['camera']['definitions']['head']['xform_rows'])
        target=read(PROJECT/'configs/head_camera_mount.json')['camera_definition']['xform_rows']
        assert np.max(np.abs(mounted-target))<1e-8
        save('state.json',state)
        from PIL import Image
        Image.fromarray(obs['sensor_data']['head']['rgb'][0].cpu().numpy()).save(OUT/'head.jpg',quality=90)
        save('simulation.json',{'passed':True,'scene':entry['id'],'catalog_models':80,
            'head_model':profile['model'],'head_pitch_deg':45,'K_error_max':float(np.max(np.abs(k-expected))),
            'lift_joint_m':float(state['qpos'][state['joint_names'].index('joint4')]),
            'sim_freq_hz':120,'control_freq_hz':40,'solver_iterations':[16,4],
            'contact_offset_m':scene.contact_offset,'rest_offset_m':scene.rest_offset,
            'target_mass_kg':float(env.obj._bodies[0].mass),'physical_grasps':0,'frozen_scene_files_verified':2406})
    finally:env.close()


def native():
    config=load_current()
    import yaml,torch
    from scipy.spatial import cKDTree
    sys.path[:0]=[str(WORKSPACE/'maniskill/arx_pipeline'),str(PROJECT/'experiments/fingertip_table_tolerance')]
    from planner import Planner
    from geometry import Geometry
    from curobo.collision_checking import RobotCollisionChecker,RobotCollisionCheckerCfg
    state=json.loads((OUT/'state.json').read_text())
    (OUT/'native_planner').mkdir(exist_ok=True)
    planner=Planner(OUT/'native_planner',state)
    q0=planner.measured(state).cpu().numpy()
    # Independent URDF FK checks at several joint configurations.
    q=np.stack([q0,q0+np.array([.03,-.04,.02,-.03,.02,-.01]),q0+np.array([-.04,.02,-.03,.01,-.02,.04])])
    result=[]
    for name,locks in [('open',[.044,.044]),('partial',[.022,.022]),('closed',[0.,0.]),('asymmetric',[.031,.012])]:
        cfg=yaml.safe_load(Path(config['planning_models']['open' if name=='open' else 'closed']).read_text())
        kin=cfg['robot_cfg']['kinematics']
        kin['urdf_path']=str(WORKSPACE/'arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf')
        kin['lock_joints'].update(dict(zip(['right_joint27','right_joint28'],locks)))
        model=OUT/(name+'.yml');model.write_text(yaml.safe_dump(cfg,sort_keys=False))
        from configuration import require_current_profile
        receipt=require_current_profile(cfg)
        checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=str(model),
            scene_model={'cuboid':{'distant':{'pose':[100.,100.,100.,1.,0.,0.,0.],'dims':[.1,.1,.1]}}},
            n_meshes=1,n_cuboids=1,collision_activation_distance=0.))
        checker.self_collision_cost=None
        native=checker.get_kinematics(torch.as_tensor(q,device='cuda',dtype=torch.float32)[:,None]).robot_spheres
        native=native.detach().cpu().numpy().reshape(len(q),-1,4)
        geom=Geometry(model,{'links':{'body4':np.eye(4)}})
        expected=geom.spheres(geom.frames(q))
        errors=[]
        for actual,want in zip(native,expected):
            actual=actual[actual[:,3]>0]
            assert len(actual)==871
            error=max(float(cKDTree(actual).query(want)[0].max()),float(cKDTree(want).query(actual)[0].max()))
            assert error<1e-6,error
            errors.append(error)
        result.append({'finger_state':name,'measured_finger_joints_m':locks,'native_sphere_count':871,
            'independent_URDF_FK_max_error_m':max(errors),**receipt})
        print('NATIVE_PASS',name,max(errors),flush=True)
    loaded=planner.planner.compute_kinematics(planner.js(planner.measured(state))).robot_spheres
    assert int((loaded[...,3]>0).sum())==871
    save('native.json',{'passed':True,'motion_planner_loaded':True,'sphere_states':result,
        'CUDA_device':torch.cuda.get_device_name(),'physical_grasps':0})


def native7():
    import xml.etree.ElementTree as ET
    import yaml,torch
    from scipy.spatial.transform import Rotation
    from scipy.spatial import cKDTree
    from curobo.collision_checking import RobotCollisionChecker,RobotCollisionCheckerCfg
    config=load_current();model=Path(config['planning_models']['lift7'])
    kin=yaml.safe_load(model.read_text())['robot_cfg']['kinematics']
    state=json.loads((OUT/'state.json').read_text())
    joints=['joint4']+['right_joint'+str(i) for i in range(21,27)]
    values={name:state['qpos'][state['joint_names'].index(name)] for name in joints}
    q=torch.tensor([[values[n] for n in joints]],device='cuda',dtype=torch.float32)
    checker=RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=str(model),
        scene_model={'cuboid':{'distant':{'pose':[100.,100.,100.,1.,0.,0.,0.],'dims':[.1,.1,.1]}}},
        n_meshes=1,n_cuboids=1,collision_activation_distance=0.))
    loaded=checker.get_kinematics(q[:,None]).robot_spheres.detach().cpu().numpy().reshape(-1,4)
    loaded=loaded[loaded[:,3]>0];assert len(loaded)==871
    parents={j.find('child').get('link'):j for j in ET.parse(kin['urdf_path']).getroot().findall('joint')}
    cache={kin['base_link']:np.eye(4)}
    def frame(name):
        if name in cache:return cache[name]
        joint=parents[name];origin=joint.find('origin');local=np.eye(4)
        if origin is not None:
            local[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            local[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
        move=np.eye(4)
        if joint.get('type')!='fixed':
            name_q=joint.get('name');value=values[name_q] if name_q in values else kin['lock_joints'][name_q]
            axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
            if joint.get('type')=='prismatic':move[:3,3]=axis*value
            else:move[:3,:3]=Rotation.from_rotvec(axis*value).as_matrix()
        cache[name]=frame(joint.find('parent').get('link'))@local@move
        return cache[name]
    want=[]
    for name,balls in kin['collision_spheres'].items():
        t=frame(name)
        want.extend([[*(t[:3,:3]@b['center']+t[:3,3]),b['radius']] for b in balls])
    error=max(float(cKDTree(want).query(loaded)[0].max()),float(cKDTree(loaded).query(want)[0].max()))
    assert error<1e-6,error
    save('native7.json',{'passed':True,'dof':7,'native_sphere_count':871,'independent_URDF_FK_max_error_m':error,'physical_grasps':0})
    print('NATIVE_7DOF_PASS',error,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['simulation','native','native7'])
    globals()[parser.parse_args().mode]()
