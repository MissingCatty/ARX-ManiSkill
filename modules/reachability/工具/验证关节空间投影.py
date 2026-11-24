"""IK first, then bounded joint-space projection, on archived observations only."""
from 公共 import EXPERIMENTS
import argparse
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from 公共 import ROOT, MS, URDF, TCP, JOINTS, read, save, sha, now
from 关节空间位姿投影 import JointProjection

DIAG = EXPERIMENTS / '抓取位姿局部修正_20261121_01'
MAP = ROOT/'产物/前方精细_20261121_01'
CACHE = EXPERIMENTS / '地图流程比较_20261121_01/结果.json'
MODEL = EXPERIMENTS / '升降联合规划_20261120_01/七关节.yml'
NAMES = ['joint4']+JOINTS
PADS = MS.parents[1]/'simulation/robots/arx/finger_pad_geometry.json'


def guard(reg):
    for path, digest in reg['source_hashes'].items():
        assert sha(Path(path)) == digest, path


def prepare(out, gpu):
    assert not out.exists()
    source = read(DIAG/'登记.json')
    files = [Path(__file__).resolve(), Path(__file__).with_name('关节空间位姿投影.py'),
             Path(__file__).with_name('公共.py'), MS/'planning_rejection_audit.py',
             MS/'七轴升降规划.py', MS/'assets/arx/robot.urdf', MS/'table_truncation_model.py',
             URDF, TCP, MODEL, PADS, DIAG/'登记.json', CACHE, MAP/'完成.json']
    import yaml
    files.append(Path(yaml.safe_load(MODEL.read_text())['robot_cfg']['kinematics']['urdf_path']))
    for case in source['cases']:
        directory = Path(case['source'])
        files += [directory/'state.json', directory/'graspcl/candidate_pool.json',
                  directory/'camera/depth.png', directory/'camera/mask.png', directory/'camera/params.json']
    out.mkdir(parents=True, exist_ok=False)
    (out/'源码').mkdir()
    for f in files:
        if f.suffix in {'.py', '.yml', '.urdf'}:
            destination = out/'源码'/f.name
            if destination.exists():
                assert sha(destination) == sha(f)
            else:
                shutil.copy2(f, destination)
    save(out/'登记.json', {'created_at': now(), 'gpu_uuid': gpu, 'cases': source['cases'],
         'source_hashes': {str(p): sha(p) for p in files}, 'physical_attempts': 0,
         'source_map_hashes': read(MAP/'完成.json')['file_sha256'],
         'model': str(MODEL), 'pipeline': 'Original pure IK first; map/direct projection only after no validated original witness.',
         'baseline_seeds': 128, 'baseline_return_seeds': 32,
         'original_eef_position_tolerance_m': .002, 'original_rotation_tolerance_rad': .03,
         'modes': ['direct_joint_projection', 'map_seeded_joint_projection'],
         'shared_start_count': 6, 'total_start_count': 8, 'max_nfev_per_start': 250,
         'wall_budget_per_goal_mode_s': 20., 'position_change_limit_m': .005,
         'rotation_change_limit_deg': 3., 'nominal_contact_change_limit_m': .003,
         'geometry_gate': 'Width unchanged; nominal endpoint drift <=3mm; observed original and new authored-pad corridors nonempty.',
         'scope': 'Development-only paired offline test. No scene collision/path check or physical execution. No training or feedback writes.',
         'initialization_note': 'Map witnesses are cached from the frozen map query; the anchor may be far from G. Only the final pose is subject to change caps. No new map targets added.'})
    save(out/'进度.json', {'phase': 'prepared', 'completed_original_goals': 0, 'total_original_goals': 22,
                          'physical_attempts': 0})
    print('PREPARED', len(source['cases']), 'cases, 22 candidates', flush=True)


def observed_points(directory):
    params = read(directory/'camera/params.json')
    k = params['intrinsics']
    depth = np.asarray(Image.open(directory/'camera/depth.png'), dtype=float)*params['depth_scale']
    mask = np.asarray(Image.open(directory/'camera/mask.png')) > 0
    v, u = np.indices(depth.shape)
    keep = mask & np.isfinite(depth) & (depth > .12) & (depth < 2.)
    points = np.stack([(u-k['ppx'])*depth/k['fx'], (v-k['ppy'])*depth/k['fy'], depth], -1)[keep]
    camera = np.asarray(read(directory/'state.json')['camera_world_optical'])
    points = points@camera[:3, :3].T+camera[:3, 3]
    if len(points):
        _, indices = np.unique(np.floor(points/.002).astype(np.int64), axis=0, return_index=True)
        points = points[np.sort(indices)]
    return points


def pad_support(points, eef, width, pads):
    low = np.max([v['pad_neutral_EEF_bounds_m'][0] for v in pads.values()], axis=0)
    high = np.min([v['pad_neutral_EEF_bounds_m'][1] for v in pads.values()], axis=0)
    local = (points-eef[:3, 3])@eef[:3, :3]
    mask = ((local[:, 0] >= low[0]) & (local[:, 0] <= high[0]) &
            (local[:, 2] >= low[2]) & (local[:, 2] <= high[2]) & (np.abs(local[:, 1]) <= width/2))
    return int(mask.sum())


def run(out):
    reg = read(out/'登记.json')
    guard(reg)
    with (out/'已开始.json').open('x') as f:
        f.write(now())
    os.environ['CUDA_VISIBLE_DEVICES'] = reg['gpu_uuid']
    sys.path[:0] = [str(MS), str(MS/'arx_pipeline'), str(MS.parent/'arx_core/scripts/curobo_arx')]
    import torch
    from curobo.types import GoalToolPose, JointState
    from planning_rejection_audit import make_pure
    from 七轴升降规划 import BodySweep
    began = time.monotonic()
    tcp = np.asarray(read(TCP)['T_eef_action'])
    pads = read(PADS)['parts']
    map_meta = read(MAP/'地图元数据.json')
    assert sha(MAP/map_meta['witness_file']) == reg['source_map_hashes'][map_meta['witness_file']]
    bank = np.memmap(MAP/map_meta['witness_file'], dtype=map_meta['dtype'], mode='r',
                     shape=(map_meta['witnesses'], len(map_meta['fields'])))
    cache = {(r['case'], r['source_index']): r for r in read(CACHE)['rows']}
    independent = BodySweep.__new__(BodySweep)
    independent.joints = ET.parse(MS/'assets/arx/robot.urdf').getroot().findall('joint')
    pure = make_pure(MODEL)

    def native_fk(state, q):
        return independent.fk(state, dict(zip(NAMES, q)))['right_eef_link']

    def original_ik(state, goal, current):
        inverse = np.linalg.inv(np.asarray(state['links']['base_link']))
        local = inverse@goal
        quat = Rotation.from_matrix(local[:3, :3]).as_quat()[[3, 0, 1, 2]]
        target = GoalToolPose(tool_frames=['right_eef_link'],
            position=torch.as_tensor(local[:3, 3], device='cuda', dtype=torch.float32).reshape(1, 1, 1, 1, 3),
            quaternion=torch.as_tensor(quat, device='cuda', dtype=torch.float32).reshape(1, 1, 1, 1, 4))
        t = time.monotonic()
        result = pure.solve_pose(target, current_state=JointState.from_position(
            torch.as_tensor(current, device='cuda', dtype=torch.float32)[None], joint_names=NAMES), return_seeds=32)
        qs = result.solution.reshape(-1, 7).detach().cpu().numpy().copy()
        claims = result.success.reshape(-1).detach().cpu().numpy().copy()
        bounds = np.array([state['joint_limits'][state['joint_names'].index(n)] for n in NAMES])
        evidence = []
        for q, claimed in zip(qs, claims):
            if not np.isfinite(q).all():
                continue
            actual = native_fk(state, q)
            pe = float(np.linalg.norm(actual[:3, 3]-goal[:3, 3]))
            re = float(Rotation.from_matrix(actual[:3, :3]@goal[:3, :3].T).magnitude())
            bounded = bool(((q >= bounds[:, 0]-1e-6) & (q <= bounds[:, 1]+1e-6)).all())
            evidence.append({'q': q.tolist(), 'solver_claimed': bool(claimed),
                'position_error_m': pe, 'orientation_error_rad': re, 'joint_limits_valid': bounded,
                'valid': bool(bounded and pe <= .002001 and re <= .030001)})
        valid = [e for e in evidence if e['valid']]
        return {'independently_valid': len(valid), 'solver_claimed': int(claims.sum()),
                'witness': valid[0] if valid else None, 'returned_solutions': evidence,
                'wall_seconds': time.monotonic()-t}

    warm = read(Path(reg['cases'][0]['source'])/'state.json')
    warm_projector = JointProjection(warm, tcp)
    probe = original_ik(warm, np.asarray(warm['links']['right_eef_link']), warm_projector.current)
    assert probe['independently_valid'] > 0
    checks = []
    for case in reg['cases']:
        state = read(Path(case['source'])/'state.json')
        candidate = read(Path(case['source'])/'graspcl/candidate_pool.json')['candidates'][0]
        goal = np.asarray(candidate['pose']).copy()
        goal[:3, 3] -= .043*goal[:3, 2]
        projector = JointProjection(state, tcp)
        error = projector.check_derivatives(goal, candidate['width'], np.asarray(candidate['point1']), np.asarray(candidate['point2']))
        rng = np.random.default_rng(1122)
        fk_errors = []
        for _ in range(8):
            q = projector.bounds[:, 0]+rng.uniform(.05, .95, 7)*np.ptp(projector.bounds, axis=1)
            own = projector.fk(q)[0]
            other = native_fk(state, q)
            pe = float(np.linalg.norm(own[:3, 3]-other[:3, 3]))
            re = float(Rotation.from_matrix(own[:3, :3]@other[:3, :3].T).magnitude())
            assert pe < 2e-5 and re < 2e-4
            fk_errors.append([pe, re])
        checks.append({'case': case['case'], 'jacobian_max_error': error, 'independent_fk_errors': fk_errors})
    save(out/'组件就绪.json', {'passed': True, 'startup_seconds': time.monotonic()-began,
         'load_counts': {'pure_ik': 1, 'map_cache': 1, 'independent_fk': 1},
         'measured_pose_probe': probe, 'model_checks': checks, 'physical_attempts': 0})
    print('READY_PROJECTION', round(time.monotonic()-began, 2), flush=True)
    rows, count = [], 0
    for case in reg['cases']:
        directory = Path(case['source'])
        state = read(directory/'state.json')
        pool = read(directory/'graspcl/candidate_pool.json')['candidates']
        points = observed_points(directory)
        projector = JointProjection(state, tcp)
        for index, c in enumerate(pool):
            goal = np.asarray(c['pose']).copy()
            goal[:3, 3] -= .043*goal[:3, 2]
            p1, p2 = np.asarray(c['point1']), np.asarray(c['point2'])
            baseline = original_ik(state, goal, projector.current)
            folder = out/'候选'/case['case']/f'{index:03d}'
            save(folder/'原位姿IK.json', baseline)
            row = {'case': case['case'], 'role': case['role'], 'index': index,
                   'original_eef_goal_world': goal.tolist(), 'original_ik': baseline,
                   'observed_pad_support_original': pad_support(points, goal, c['width'], pads),
                   'projection_triggered': not bool(baseline['independently_valid']), 'modes': {},
                   'physical_attempts': 0}
            if baseline['independently_valid']:
                for mode in reg['modes']:
                    row['modes'][mode] = {'accepted': True, 'status': 'original_pose_retained',
                        'map_query_performed': False, 'pose_changed': False,
                        'q_witness_7': baseline['witness']['q'], 'eef_goal_world': goal.tolist(),
                        'source': 'Shared original IK gate; not a new independent repeat.'}
            else:
                returned = sorted(baseline['returned_solutions'], key=lambda r:
                    (r['position_error_m']/.005)**2+(r['orientation_error_rad']/np.deg2rad(3))**2)
                rng = np.random.default_rng(112200+count)
                common, extras = [], []
                for q in [projector.current]+[np.asarray(r['q']) for r in returned]:
                    q = np.clip(q, projector.bounds[:, 0]+1e-8, projector.bounds[:, 1]-1e-8)
                    if common and min(np.linalg.norm(q-v) for v in common) < 1e-5:
                        continue
                    common.append(q)
                    if len(common) == 8:
                        break
                while len(common) < 8:
                    common.append(rng.uniform(projector.bounds[:, 0]+1e-8, projector.bounds[:, 1]-1e-8))
                extras = common[6:8]
                common = common[:6]
                cached = cache[(case['case'], index)]
                map_extras = []
                for key in ['fixed_lift', 'joint7_lift']:
                    record = cached[key]
                    q = np.asarray(record['q_witness_7'])
                    assert np.allclose(q[1:], bank[record['bank_row'], :6], atol=1e-7)
                    map_extras.append(q)
                for mode in reg['modes']:
                    t = time.monotonic()
                    starts = common+(map_extras if mode == 'map_seeded_joint_projection' else extras)
                    attempts, selected, best = [], None, None
                    for seed_id, seed in enumerate(starts):
                        if time.monotonic()-t >= reg['wall_budget_per_goal_mode_s']:
                            break
                        started = time.monotonic()
                        result = projector.solve(seed, goal, c['width'], p1, p2,
                                                 max_nfev=reg['max_nfev_per_start'])
                        q = result.x
                        actual = native_fk(state, q)
                        own = projector.fk(q)[0]
                        pe = float(np.linalg.norm(actual[:3, 3]-own[:3, 3]))
                        re = float(Rotation.from_matrix(actual[:3, :3]@own[:3, :3].T).magnitude())
                        assert pe < 2e-5 and re < 2e-4
                        bounded = bool(((q >= projector.bounds[:, 0]-1e-7) & (q <= projector.bounds[:, 1]+1e-7)).all())
                        assert bounded
                        action, target_action = actual@tcp, goal@tcp
                        dp = float(np.linalg.norm(action[:3, 3]-target_action[:3, 3]))
                        dr = float(Rotation.from_matrix(actual[:3, :3]@goal[:3, :3].T).magnitude())
                        a1 = action[:3, 3]-c['width']/2*actual[:3, 1]
                        a2 = action[:3, 3]+c['width']/2*actual[:3, 1]
                        contact = float(max(np.linalg.norm(a1-p1), np.linalg.norm(a2-p2)))
                        support = pad_support(points, actual, c['width'], pads)
                        front = bool((np.linalg.inv(state['links']['body4'])@action)[0, 3] >= 0)
                        pose_pass = bool(dp <= .005+1e-8 and dr <= np.deg2rad(3)+1e-8 and front)
                        geometry = bool(contact <= .003+1e-8 and support > 0 and row['observed_pad_support_original'] > 0)
                        detail = {'start': seed_id, 'q_seed': np.asarray(seed).tolist(), 'q_witness_7': q.tolist(),
                            'eef_goal_world': actual.tolist(), 'position_change_m': dp,
                            'orientation_change_deg': float(np.rad2deg(dr)), 'nominal_contact_change_m': contact,
                            'observed_pad_support': support, 'width_unchanged_m': c['width'],
                            'independent_fk_position_error_m': pe, 'independent_fk_rotation_error_rad': re,
                            'joint_limits_valid': bounded, 'front_scope': front, 'pose_caps_passed': pose_pass,
                            'observed_geometry_proxy_passed': geometry, 'accepted': bool(pose_pass and geometry),
                            'cost': float(result.cost), 'nfev': result.nfev, 'njev': result.njev,
                            'optimizer_status': result.status, 'optimizer_message': result.message,
                            'wall_seconds': time.monotonic()-started,
                            'physical_grasp_quality_verified': False, 'scene_or_path_checked': False}
                        attempts.append(detail)
                        if best is None or detail['cost'] < best['cost']:
                            best = detail
                        if detail['accepted']:
                            selected = detail
                            break
                    output = {'accepted': selected is not None, 'status': 'verified_projection' if selected else 'no_admissible_projection_in_budget',
                        'pose_changed': selected is not None, 'map_query_performed': mode == 'map_seeded_joint_projection',
                        'map_query_source': str(CACHE) if mode == 'map_seeded_joint_projection' else None,
                        'map_query_latency_remeasured': False, 'attempts': attempts, 'selected': selected, 'best': best,
                        'wall_seconds': time.monotonic()-t, 'starts_attempted': len(attempts),
                        'initialization_only_difference': True, 'global_nearest_proven': False}
                    save(folder/(mode+'.json'), output)
                    row['modes'][mode] = output
                    print('PROJECTED', case['case'], index, mode, output['accepted'],
                          'best_mm', round(best['position_change_m']*1000, 3) if best else None,
                          'best_deg', round(best['orientation_change_deg'], 3) if best else None, flush=True)
            save(folder/'结果.json', row)
            rows.append(row)
            count += 1
            save(out/'进度.json', {'phase': 'ik_then_projection', 'completed_original_goals': count,
                 'total_original_goals': 22, 'latest_case': case['case'], 'latest_index': index,
                 'physical_attempts': 0})
            print('GOAL_COMPLETE', case['case'], index, 'original_valid', baseline['independently_valid'], flush=True)
    guard(reg)
    summary = {'original_goals': len(rows), 'original_ik_passed': sum(not r['projection_triggered'] for r in rows),
               'projection_inputs': sum(r['projection_triggered'] for r in rows), 'modes': {}}
    for mode in reg['modes']:
        accepted = [r for r in rows if r['projection_triggered'] and r['modes'][mode]['accepted']]
        summary['modes'][mode] = {'total_pose_witnesses': sum(r['modes'][mode]['accepted'] for r in rows),
            'new_admissible_projections': len(accepted),
            'new_pose_cap_passes': sum(any(a['pose_caps_passed'] for a in r['modes'][mode].get('attempts', [])) for r in rows if r['projection_triggered']),
            'retained_original_poses': sum(not r['projection_triggered'] for r in rows)}
    save(out/'结果汇总.json', {'passed': True, 'created_at': now(), 'summary': summary, 'rows': rows,
         'source_hashes_unchanged': True, 'physical_attempts': 0, 'production_pipeline_modified': False,
         'wall_seconds': time.monotonic()-began, 'scope': reg['scope']})
    save(out/'进度.json', {'phase': 'complete', 'completed_original_goals': count, 'total_original_goals': 22,
                          'summary': summary, 'physical_attempts': 0})
    print('COMPLETE_PROJECTION', summary, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'run'])
    p.add_argument('--输出', type=Path, required=True)
    p.add_argument('--显卡')
    args = p.parse_args()
    if args.stage == 'prepare':
        assert args.显卡
        prepare(args.输出.resolve(), args.显卡)
    else:
        try:
            run(args.输出.resolve())
        except BaseException:
            save(args.输出/'故障.json', {'time': now(), 'error': traceback.format_exc(), 'physical_attempts': 0})
            raise
