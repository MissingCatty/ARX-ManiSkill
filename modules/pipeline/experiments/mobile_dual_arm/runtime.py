"""Mobile observation preparation before the unchanged stationary grasp executor."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from arx_project.layout import read_json, write_json
from modules.mobile_manipulation.placement import SweepChecker, backproject, hypothetical_state, station_proposals


def capture(env, directory, executor, entry):
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'camera').mkdir()
    obs = env.get_obs()
    state = executor.snapshot(env, obs)
    data = obs['sensor_data']['head']
    Image.fromarray(data['rgb'][0].cpu().numpy()).save(directory/'camera/color.png')
    Image.fromarray(data['depth'][0, :, :, 0].cpu().numpy().astype(np.uint16)).save(directory/'camera/depth.png')
    k = np.asarray(state['intrinsics'])
    write_json(directory/'camera/params.json', {'depth_scale': .001, 'intrinsics': {
        'width': 640, 'height': 480, 'fx': k[0, 0], 'fy': k[1, 1], 'ppx': k[0, 2], 'ppy': k[1, 2]}})
    write_json(directory/'state.json', state)
    write_json(directory/'entry.json', entry)
    Image.fromarray(env.render().cpu().numpy()[0]).save(directory/'overview.jpg', quality=86)
    return state


class Stationing:
    def __init__(self, services, executor, model, settings, seed, *, station_search=None):
        self.services, self.executor, self.model, self.settings, self.seed = services, executor, model, settings, seed
        self.station_search = station_search

    def prepare(self, env, out, entry):
        from repair_contract import make_candidates
        coarse = out/'stationing/coarse'
        state = capture(env, coarse, self.executor, entry)
        sam = self.services.request('sam3', coarse)
        if not sam['result'].get('target_found', True):
            return {'success': False, 'reason': 'coarse_sam3_no_target_mask', 'motion_started': False}
        mask = np.asarray(Image.open(coarse/'camera/mask.png')) > 0
        depth = np.load(coarse/'camera/depth_m.npy', allow_pickle=False)
        if depth.ndim == 3:
            depth = depth.squeeze()
        try:
            points = backproject(mask, depth, state['intrinsics'], state['camera_world_optical'])
        except ValueError:
            return {'success': False, 'reason': 'coarse_invalid_depth', 'motion_started': False}
        center = np.median(points, axis=0)
        write_json(out/'stationing/measurement.json', {'source': 'SAM3_mask_and_measured_RGBD',
            'points': len(points), 'world_median': center.tolist(), 'simulator_target_pose_used': False})
        ray = self.services.request('rayst3r', coarse)
        if ray['result'].get('point_count', 0) < 20:
            return {'success': False, 'reason': 'coarse_rayst3r_empty_pointcloud', 'motion_started': False}
        self.services.request('graspgenx', coarse, seed=self.seed)
        make_candidates(coarse)
        if self.station_search is not None:
            selected, search_report = self.station_search.select(state,
                read_json(coarse/'candidates/candidate_pool.json'), center, out/'stationing/search')
            if selected is None:
                return {'success': False, 'reason': 'no_safe_reachable_station', 'motion_started': False,
                        'search_report_file': str(out/'stationing/search/search_result.json'),
                        'search_summary': {k:v for k,v in search_report.items() if k!='events'}}
            chosen, path = selected['proposal'], selected['path']
            write_json(out/'stationing/selected.json', {**chosen, 'coarse_plan': selected['work']+'/plan_result.json',
                       'transit': selected['transit'], 'grasp_will_be_reobserved_and_replanned': True})
            write_json(out/'stationing/command.json', {'path': path, 'goal': chosen})
            receipt = self.execute(env, out/'stationing', state, chosen, path)
            receipt['search_report_file'] = str(out/'stationing/search/search_result.json')
            receipt['search_summary'] = {k:v for k,v in search_report.items() if k!='events'}
            write_json(out/'stationing/receipt.json', receipt)
            return receipt
        checker = SweepChecker(self.model, state, self.settings['transit_clearance_m'])
        proposals = station_proposals(state, center, self.settings)
        checked, eligible = [], []
        # Geometry first; actual full-pose native plans decide station admission.
        for proposal in proposals:
            result = checker.check(proposal['base_world'], proposal['lift_m'])
            checked.append({**proposal, 'destination_geometry': result})
            if result['passed']:
                eligible.append(proposal)
        write_json(out/'stationing/proposals.json', {'generated': len(proposals), 'checked': checked,
                                                     'eligible': len(eligible)})
        chosen = None
        path_passes, planning_calls = 0, 0
        for i, proposal in enumerate(eligible):
            if i >= self.settings['maximum_full_planning_stations']:
                break
            work = out/'stationing'/f'candidate_{i:03d}'
            work.mkdir(); (work/'candidates').mkdir()
            prospective = hypothetical_state(self.model, state, proposal['base_world'], proposal['lift_m'])
            write_json(work/'state.json', prospective)
            write_json(work/'candidates/candidate_pool.json', read_json(coarse/'candidates/candidate_pool.json'))
            gate, path = checker.path(proposal, self.settings)
            write_json(work/'transit_check.json', gate)
            if not gate['passed']:
                continue
            path_passes += 1
            planning_calls += 1
            response = self.services.request('curobo_dual_arm', work)
            if response['result']['success']:
                chosen = proposal
                write_json(out/'stationing/selected.json', {**proposal, 'candidate_index': i,
                    'coarse_plan': str(work/'plan_result.json'), 'transit': gate,
                    'grasp_will_be_reobserved_and_replanned': True})
                write_json(out/'stationing/command.json', {'path': path, 'goal': proposal})
                break
        if chosen is None:
            return {'success': False, 'reason': 'no_safe_reachable_station', 'motion_started': False,
                    'proposals': len(proposals), 'geometry_eligible': len(eligible),
                    'transit_passes': path_passes, 'grasp_planning_calls': planning_calls,
                    'failed_stage': 'destination_geometry' if not eligible else
                        'transit_geometry' if not path_passes else 'native_grasp_planning'}
        receipt = self.execute(env, out/'stationing', state, chosen, path)
        write_json(out/'stationing/receipt.json', receipt)
        return receipt

    def execute(self, env, directory, start, goal, path):
        """Kinematic SE2 base path; real PD lift; measured idle before recapture."""
        import imageio.v2 as imageio
        import sapien
        import torch
        robot = env.agent.robot
        names = [j.name for j in robot.active_joints]
        lift_index = names.index('joint4')
        command = robot.qpos[0].cpu().numpy().copy()
        held = [i for i in range(len(names)) if i != lift_index]
        reference = command[held].copy()
        robot_entities = {link._objs[0].entity for link in robot.get_links()}
        objects_before = {n: a.pose.p[0].cpu().numpy().copy() for n, a in env.objects.items()}
        frames, events, rows, stopped = 0, [], [], False
        writer = imageio.get_writer(directory/'motion.mp4', fps=20, codec='libx264', quality=7, macro_block_size=2)
        head = imageio.get_writer(directory/'head.mp4', fps=20, codec='libx264', quality=7, macro_block_size=2)
        trace = (directory/'execution.jsonl').open('x')

        def record():
            nonlocal frames, stopped
            frames += 1
            q = robot.qpos[0].cpu().numpy()
            vel = robot.qvel[0].cpu().numpy()
            base = robot.links_map['base_link'].pose.to_transformation_matrix()[0].cpu().numpy()
            for contact in env.scene.get_contacts():
                a, b = [body.entity for body in contact.bodies]
                if (a in robot_entities) == (b in robot_entities):
                    continue
                if not contact.points or sum(np.linalg.norm(p.impulse) for p in contact.points) <= 1e-8:
                    continue
                other = b if a in robot_entities else a
                link = a if a in robot_entities else b
                # Original fixed-root wheels intersect the ground by 25.5mm;
                # base XY/yaw keeps this pre-existing support geometry frozen.
                if link.name in ['wheel1', 'wheel2', 'wheel3', 'base_link'] and 'ground' in other.name:
                    continue
                event = {'tick': frames, 'robot': link.name, 'other': other.name}
                events.append(event); stopped = True
            row = {'tick': frames, 'time_s': frames/120, 'qpos': q.tolist(), 'qvel': vel.tolist(),
                   'base_world': base.tolist(), 'command': command.tolist()}
            trace.write(json.dumps(row)+'\n')
            if frames % 6 == 0:
                writer.append_data(env.render().cpu().numpy()[0])
                head.append_data(env.get_obs()['sensor_data']['head']['rgb'][0].cpu().numpy())
            if np.abs(q[held]-reference).max() >= .005:
                stopped = True
                events.append({'tick': frames, 'reason': 'parked_joint_drift'})
            rows.append((q.copy(), vel.copy(), base.copy()))

        previous_callback = getattr(env, 'trace_callback', None)
        env.trace_callback = record
        try:
            # Commands resampled to fixed 40Hz control, using the exact checked
            # smooth path and interpolated SE2 poses between stored samples.
            times = np.array([r['time_s'] for r in path])
            matrices = np.array([r['base_world'] for r in path])
            lifts = np.array([r['lift_m'] for r in path])
            yaws = np.unwrap(Rotation.from_matrix(matrices[:, :3, :3]).as_euler('xyz')[:, 2])
            from modules.mobile_manipulation.placement import pose_matrix
            for t in np.arange(0., times[-1]+.025, .025):
                xyz = [np.interp(t, times, matrices[:, k, 3]) for k in range(3)]
                matrix = pose_matrix(*xyz[:2], np.interp(t, times, yaws), xyz[2])
                robot.set_pose(sapien.Pose(matrix))
                command[lift_index] = np.interp(t, times, lifts)
                env._step_action(torch.tensor(command[None], device=env.device))
                if stopped:
                    raise RuntimeError('station_motion_contact_or_drift')
            # Maintain final command and require a full half-second idle window.
            for _ in range(160):
                env._step_action(torch.tensor(command[None], device=env.device))
                if stopped:
                    raise RuntimeError('station_motion_contact_or_drift')
                recent = rows[-60:]
                if len(recent) == 60 and max(abs(r[0][lift_index]-goal['lift_m']) for r in recent) < .001 \
                        and max(np.abs(r[1]).max() for r in recent) < .01:
                    break
            else:
                raise RuntimeError('station_robot_idle_unconfirmed')
            final = self.executor.snapshot(env, env.get_obs())
            write_json(directory/'settled_state.json', final)
            displacements = {n: float(np.linalg.norm(a.pose.p[0].cpu().numpy()-objects_before[n]))
                             for n, a in env.objects.items()}
            if max(displacements.values(), default=0.) > .003:
                raise RuntimeError('station_motion_moved_scene_object')
            base = np.asarray(final['links']['base_link'])
            if np.max(np.abs(base-np.asarray(goal['base_world']))) > .001:
                raise RuntimeError('station_base_not_at_goal')
            return {'success': True, 'reason': 'station_ready', 'motion_started': True,
                    'ticks': frames, 'lift_m': float(robot.qpos[0, lift_index]),
                    'object_displacements_m': displacements, 'idle_samples': 60,
                    'base_world': base.tolist(), 'base_execution': 'kinematic_SE2_rate_limited',
                    'lift_execution': 'original_joint4_PD', 'forbidden_contacts': len(events)}
        finally:
            env.trace_callback = previous_callback
            trace.close(); writer.close(); head.close()
            write_json(directory/'contacts.json', events)
