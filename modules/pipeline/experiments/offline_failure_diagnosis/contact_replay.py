"""Controlled ARX closure diagnostics, separate from the stopped benchmark."""
from arx_project.assets import load_object_catalog
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
MS = PROJECT.parent / 'maniskill'
SOURCE = PROJECT / 'experiments/diverse_grasp_benchmark/runs/geometry200_fixed452_20261122_01'
DEFAULT = HERE / 'runs/contact_replay_20261123_01'
SLOTS = ['D0003_S1_open_mushroom', 'D0042_S2_edge_lemon']
CONDITIONS = {
    'original': {},
    'contact_offset_1mm': {'contact_offset': .001},
    'friction_08_06': {'finger_friction': [.8, .6]},
    'force_limit_5': {'force_limit': 5.},
    'ramp_1500ms': {'ramp_seconds': 1.5},
    'solver_64_16': {'solver_position_iterations': 64, 'solver_velocity_iterations': 16},
    'frequency_480hz': {'sim_freq': 480},
    'depenetration_02': {'max_depenetration_velocity': .2},
}


def read(p):
    return json.loads(Path(p).read_text())


def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    def encode(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v).__name__)
    tmp = p.with_suffix(p.suffix+'.pending')
    tmp.write_text(json.dumps(x, indent=2, default=encode)+'\n'); tmp.replace(p)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(root):
    assert not root.exists(), 'Use a new diagnostic registration'
    root.mkdir(parents=True)
    schedule = [{'id': f'{condition}__{slot}', 'slot': slot, 'condition': condition,
                 'settings': settings, 'replay': 'closure_only'}
                for condition, settings in CONDITIONS.items() for slot in SLOTS]
    schedule += [{'id': 'fullpath_original__'+slot, 'slot': slot, 'condition': 'original',
                  'settings': {}, 'replay': 'full_path'} for slot in SLOTS]
    inputs = {}
    for slot in SLOTS:
        d = SOURCE/'trials'/slot
        for name in ['entry.json', 'state.json', 'execution.jsonl', 'plan_result.json', 'approach_plan_result.json']:
            inputs[str(d/name)] = sha(d/name)
    save(root/'registration.json', {
        'purpose': 'Verify launch/contact causes using fixed saved grasps; not resume the 200-scene test',
        'authorization': '2026-11-23 user: verify launch cause and distinguish perception, executed pose and generated grasp errors',
        'required_components': ['ARX SAPIEN simulation'], 'new_model_inference': False,
        'source_run': str(SOURCE), 'schedule': schedule, 'maximum_closure_diagnostics': len(schedule),
        'automatic_retries': 0, 'baseline_physics': 'unchanged', 'input_sha256': inputs,
        'diagnostic_source_sha256': {str(Path(__file__)): sha(__file__)},
        'source_benchmark_stop_sha256': sha(SOURCE/'STOP'),
        'stop_case_displacement_m': .5, 'each_condition_max_close_seconds': 3.,
        'video': 'original conditions only, closure stage',
    })
    save(root/'progress.json', {'phase': 'registered', 'completed': 0, 'total': len(schedule)})


def run_one(root, item, runtime, catalog, keeper):
    environment, run, sapien, torch, gym, override = runtime
    d = SOURCE/'trials'/item['slot']; out = root/'trials'/item['id']
    out.mkdir(parents=True, exist_ok=False)
    save(out/'claim.json', item)
    entry, selected = read(d/'entry.json'), read(d/'plan_result.json')['selected']
    cfg = item['settings']; sim = {'scene_config': {}}
    for key in ['contact_offset', 'solver_position_iterations', 'solver_velocity_iterations']:
        if key in cfg: sim['scene_config'][key] = cfg[key]
    if 'sim_freq' in cfg: sim['sim_freq'] = cfg['sim_freq']
    env = None; writer = None; trace = None
    try:
        env = gym.make('ARXPickCatalog-v1', model_id=entry['model_id'], object_catalog=catalog,
            perception_camera='head', scene_spec=deepcopy(entry['scene']), num_envs=1,
            obs_mode='rgb+depth+segmentation', control_mode='pd_joint_pos', reward_mode='none',
            render_mode='rgb_array', sim_backend='cpu', render_backend='sapien_cuda',
            sensor_configs=override, sim_config=sim).unwrapped
        if not keeper: keeper.append(sapien.render.SapienRenderer(env._render_device))
        env.reset(seed=entry['seed'])
        names = [j.name for j in env.agent.robot.active_joints]
        arm = [names.index(f'right_joint{i}') for i in range(21, 27)]
        finger = [names.index(f'right_joint{i}') for i in [27, 28]]
        for _ in range(80): env._step_action(torch.tensor(env.initial_q[None], device=env.device))
        command = env.agent.robot.qpos[0].cpu().numpy().copy()

        if item['replay'] == 'closure_only':
            rows = [json.loads(s) for s in (d/'execution.jsonl').open()]
            ix = next(i for i, r in enumerate(rows) if r['stage'] == 'close')
            prior = rows[ix-1]
            command = np.array(prior['qpos'], dtype=np.float32)
            env.agent.reset(command)
            env.obj.set_pose(sapien.Pose(prior['object_pose'][:3], prior['object_pose'][3:]))
            env.obj._bodies[0].linear_velocity = np.zeros(3)
            env.obj._bodies[0].angular_velocity = np.zeros(3)
        else:
            for key, path in [('pregrasp', read(d/'plan_result.json')['paths']['pregrasp']),
                              ('approach', read(d/'approach_plan_result.json')['paths']['approach'])]:
                q, t = np.asarray(path['q']), np.asarray(path['times'])
                steps = int(np.ceil(t[-1]*40))
                for step in range(steps+1):
                    command[arm] = [np.interp(min(step/40, t[-1]), t, q[:, j]) for j in range(6)]
                    command[finger] = .044
                    env._step_action(torch.tensor(command[None], device=env.device))
                for _ in range(160):
                    if np.max(np.abs(env.agent.robot.qpos[0, arm].cpu().numpy()-q[-1])) < .005: break
                    env._step_action(torch.tensor(command[None], device=env.device))
        command[arm] = selected['q_grasp']
        target_body = env.obj._bodies[0]
        fingers = [env.agent.robot.links_map[f'right_link{i}']._objs[0] for i in [27, 28]]
        if 'finger_friction' in cfg:
            material = sapien.physx.PhysxMaterial(*cfg['finger_friction'], 0.)
            for body in fingers:
                for shape in body.get_collision_shapes(): shape.physical_material = material
        if 'force_limit' in cfg:
            for j in env.agent.robot.active_joints:
                if j.name in ['right_joint27', 'right_joint28']:
                    j.set_drive_properties(8000., 800., force_limit=cfg['force_limit'], mode='acceleration')
        if 'max_depenetration_velocity' in cfg:
            for body in [target_body, *fingers]: body.max_depenetration_velocity = cfg['max_depenetration_velocity']
        info = {'simulation': env.sim_config.dict(), 'object_mass_kg': target_body.mass,
                'object_inertia': target_body.inertia, 'object_cmass': target_body.cmass_local_pose.to_transformation_matrix(),
                'object_max_depenetration_velocity': target_body.max_depenetration_velocity,
                'target_collision_shapes': len(target_body.get_collision_shapes()),
                'finger_collision_shapes': [len(body.get_collision_shapes()) for body in fingers],
                'finger_contact_offsets': [sorted(set(s.contact_offset for s in b.get_collision_shapes())) for b in fingers],
                'object_contact_offsets': sorted(set(s.contact_offset for s in target_body.get_collision_shapes())),
                'object_restitution': sorted(set(s.physical_material.restitution for s in target_body.get_collision_shapes())),
                'finger_static_friction': [sorted(set(s.physical_material.static_friction for s in b.get_collision_shapes())) for b in fingers],
                'qpos_at_close': env.agent.robot.qpos[0].cpu().numpy(),
                'eef_at_close': env.agent.tcp.pose.to_transformation_matrix().cpu().numpy()[0],
                'selected_eef': selected['grasp_world'],
                'object_at_close': env.obj.pose.raw_pose[0].cpu().numpy(),
                'settings': cfg, 'diagnostic': True}
        save(out/'configuration.json', info)
        initial = np.asarray(env.obj.pose.p[0].cpu().numpy())
        start_finger = command[finger].copy()
        if item['condition'] == 'original':
            import imageio.v2 as imageio
            writer = imageio.get_writer(out/'closure.mp4', fps=20, codec='libx264', quality=6,
                                       macro_block_size=2)
            writer.append_data(env.render().cpu().numpy()[0])
        tick = 0; rows = []
        trace = (out/'contact_trace.jsonl').open('x')
        target_entity = env.obj._objs[0]
        robot_entities = {l._objs[0].entity for l in env.agent.robot.get_links()}

        def record():
            nonlocal tick
            tick += 1
            events = []
            for contact in env.scene.get_contacts():
                a, b = [body.entity for body in contact.bodies]
                if target_entity not in [a, b]: continue
                if a not in robot_entities and b not in robot_entities: continue
                points = []
                for point in contact.points:
                    impulse = np.asarray(point.impulse)
                    if np.linalg.norm(impulse) <= 1e-8: continue
                    points.append({'position': np.asarray(point.position).tolist(),
                        'normal': np.asarray(point.normal).tolist(), 'separation_m': point.separation,
                        'impulse_ns': impulse.tolist()})
                if points: events.append({'a': a.name, 'b': b.name, 'points': points})
            row = {'tick': tick, 't': tick/env.sim_config.sim_freq,
                'position': np.asarray(env.obj.pose.p[0].cpu().numpy()).tolist(),
                'velocity': np.asarray(target_body.linear_velocity).tolist(),
                'angular_velocity': np.asarray(target_body.angular_velocity).tolist(),
                'finger_qpos': env.agent.robot.qpos[0, finger].cpu().numpy().tolist(),
                'finger_qvel': env.agent.robot.qvel[0, finger].cpu().numpy().tolist(),
                'target_finger': command[finger].tolist(), 'contacts': events}
            rows.append(row); trace.write(json.dumps(row)+'\n')
        env.trace_callback = record
        stop = 'time_complete'
        for step in range(121):
            t = step/40
            fraction = min(1., t/cfg['ramp_seconds']) if 'ramp_seconds' in cfg else 1.
            command[finger] = start_finger*(1-fraction)
            env._step_action(torch.tensor(command[None], device=env.device))
            if writer is not None and step%2 == 0: writer.append_data(env.render().cpu().numpy()[0])
            if np.linalg.norm(np.asarray(env.obj.pose.p[0].cpu().numpy())-initial) > .5:
                stop = 'displacement_limit'; break
        env.trace_callback = None
        speeds = np.array([np.linalg.norm(r['velocity']) for r in rows])
        net = []
        for r in rows:
            pts = [p for c in r['contacts'] for p in c['points']]
            net.append(sum(np.linalg.norm(p['impulse_ns']) for p in pts))
        first = next((r for r in rows if r['contacts']), None)
        summary = {'id': item['id'], 'source_slot': item['slot'], 'settings': cfg,
            'replay': item['replay'], 'diagnostic_only': True, 'stop': stop, 'ticks': tick,
            'max_speed_mps': float(speeds.max()), 'max_speed_tick': rows[int(np.argmax(speeds))]['tick'],
            'max_step_impulse_sum_ns': float(max(net)),
            'max_displacement_m': float(max(np.linalg.norm(np.asarray(r['position'])-initial) for r in rows)),
            'first_contact_tick': None if first is None else first['tick'],
            'min_contact_separation_m': min((p['separation_m'] for r in rows for c in r['contacts'] for p in c['points']), default=None),
            'final_finger_qpos': rows[-1]['finger_qpos'],
            'actual_eef_error_mm': float(np.linalg.norm(env.agent.tcp.pose.p[0].cpu().numpy()-np.asarray(selected['grasp_world'])[:3, 3])*1000)}
        save(out/'result.json', summary)
        print(json.dumps(summary), flush=True)
        return summary
    except BaseException:
        save(out/'fault.json', {'error': traceback.format_exc()}); raise
    finally:
        if env is not None: env.trace_callback = None
        if trace is not None: trace.close()
        if writer is not None: writer.close()
        if env is not None: env.close()


def run(root, only):
    reg = read(root/'registration.json')
    for file, h in {**reg['input_sha256'], **reg['diagnostic_source_sha256']}.items(): assert sha(file) == h, file
    assert sha(SOURCE/'STOP') == reg['source_benchmark_stop_sha256']
    os.environ.setdefault('MS_ASSET_DIR', str(MS/'assets'))
    sys.path[:0] = [str(PROJECT/'experiments/diverse_object_scenes'), str(MS),
                   str(MS/'arx_pipeline'), str(PROJECT/'experiments/pointcloud_benchmark')]
    import arx_runtime_imports
    arx_runtime_imports.install()
    import environment, run as execution_source, sapien, torch, gymnasium as gym
    import asset_environment
    from head_camera_config import head_configuration
    environment.INITIAL_JOINTS = {**environment.INITIAL_JOINTS, 'joint4': .452}
    override, head = head_configuration(environment.CURRENT_CONTROLLERS)
    environment.CURRENT_CONTROLLERS = deepcopy(environment.CURRENT_CONTROLLERS)
    environment.CURRENT_CONTROLLERS['camera']['definitions']['head'] = head['camera_definition']
    catalog = load_object_catalog(SOURCE/'object_catalog.json')['models']
    runtime = (environment, execution_source, sapien, torch, gym, override)
    keeper = []
    results = []
    save(root/'runtime_ready.json', {'pid': os.getpid(), 'components': ['SAPIEN ARX'],
        'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'), 'neural_models_loaded': 0})
    for item in reg['schedule']:
        if only and item['id'] not in only: continue
        if (root/'trials'/item['id']).exists():
            print('ALREADY_CLAIMED', item['id'], flush=True); continue
        if (root/'STOP').exists(): raise RuntimeError('Diagnostic STOP present')
        results.append(run_one(root, item, runtime, catalog, keeper))
        save(root/'progress.json', {'phase': 'diagnosing', 'completed_this_invocation': len(results),
            'last': results[-1], 'formal_benchmark_resumed': False})
    all_results = [read(p) for p in sorted((root/'trials').glob('*/result.json'))]
    save(root/'summary.json', {'completed': len(all_results), 'registered': len(reg['schedule']),
        'results': all_results, 'formal_benchmark_resumed': False})
    save(root/'progress.json', {'phase': 'invocation_complete', 'completed': len(all_results),
        'registered': len(reg['schedule']), 'formal_benchmark_resumed': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'run'])
    p.add_argument('--root', type=Path, default=DEFAULT)
    p.add_argument('--only', nargs='*')
    a = p.parse_args()
    if a.stage == 'prepare': prepare(a.root.resolve())
    else: run(a.root.resolve(), a.only)
