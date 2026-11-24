"""CPU bounded least-squares cross-check; never controls a robot."""
from datetime import datetime
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import 核查挡物规划 as first
from scene_tools import ArmFK

read, save = first.read, first.save
OUT = first.DEST.parent / '20261119_03_02'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    fk = ArmFK();random = np.random.default_rng(20261119)
    save(OUT / '核查登记.json', {'time': datetime.now().astimezone().isoformat(), 'method': 'independent CPU scipy bounded least_squares on unchanged URDF',
         'physical_attempts': 0, 'same_position_tolerance_m': .002, 'same_orientation_tolerance_rad': .03,
         'joint_bound_changes': False, 'source_code_sha256': first.sha(__file__),
         'max_cpu_starts_per_goal': 12, 'max_function_evaluations_per_start': 180})
    began = time.monotonic();all_rows = []
    for case in first.cases():
        if not case['reason'].startswith('抓取规划:'):continue
        name = case['case'];state = read(Path(case['directory']) / 'state.json')
        base = np.asarray(state['links']['body4']);inverse = np.linalg.inv(base)
        bounds = np.array([state['joint_limits'][state['joint_names'].index('right_joint' + str(n))] for n in range(21, 27)])
        lower, upper = bounds[:, 0]+1e-4, bounds[:, 1]-1e-4
        actual = np.array([state['qpos'][state['joint_names'].index('right_joint' + str(n))] for n in range(21, 27)])
        grasp = dict(np.load(first.DEST / name / 'grasp_运动学与碰撞.npz'))
        pre = dict(np.load(first.DEST / name / 'pregrasp_运动学与碰撞.npz'))
        positions = dict(np.load(first.DEST / name / '仅位置解.npz'))
        by_index = {int(index): i for i,index in enumerate(positions['candidate_indices'])}
        rows = []
        tasks = [('grasp', i, grasp) for i in np.flatnonzero(~grasp['cpu_pose_valid'].any(-1))]
        tasks += [('pregrasp', i, pre) for i in np.flatnonzero(grasp['cpu_pose_valid'].any(-1) & ~pre['cpu_pose_valid'].any(-1))]
        for stage, index, arrays in tasks:
            goal = inverse @ arrays['goals'][index]
            def residual(q):
                pose = fk.fk(q)
                return np.r_[pose[:3, 3]-goal[:3, 3], .08*Rotation.from_matrix(goal[:3, :3].T @ pose[:3, :3]).as_rotvec()]
            raw = arrays['q'][index]
            scores = [np.linalg.norm(residual(q)) if np.isfinite(q).all() else np.inf for q in raw]
            seeds = [raw[i] for i in np.argsort(scores)[:2]]
            if stage == 'pregrasp':
                seeds = list(grasp['q'][index][grasp['cpu_pose_valid'][index]][:2]) + seeds
            elif index in by_index:
                p = positions['q'][by_index[index]][positions['cpu_pose_valid'][by_index[index]]]
                order = sorted(range(len(p)), key=lambda j: np.linalg.norm(residual(p[j])))
                seeds += [p[j] for j in order[:2]]
            seeds += [actual] + [random.uniform(lower, upper) for _ in range(7)]
            best = None;found = None
            for n, seed in enumerate(seeds[:12]):
                seed = np.clip(seed, lower+1e-7, upper-1e-7)
                solved = least_squares(residual, seed, bounds=(lower, upper), max_nfev=180,
                                       ftol=1e-10, xtol=1e-10, gtol=1e-10)
                pose = fk.fk(solved.x)
                p = float(np.linalg.norm(pose[:3, 3]-goal[:3, 3]))
                r = float(Rotation.from_matrix(goal[:3, :3].T @ pose[:3, :3]).magnitude())
                margin = np.minimum(solved.x-lower, upper-solved.x)
                item = {'seed_index': n, 'q': solved.x.tolist(), 'position_error_m': p,
                        'rotation_error_rad': r, 'near_joint_bounds': [int(21+j) for j in np.flatnonzero(margin < .001)],
                        'residual_norm': float(np.linalg.norm(residual(solved.x))), 'nfev': solved.nfev}
                if best is None or item['residual_norm'] < best['residual_norm']:best=item
                if p <= .002 and r <= .03:
                    found=item;break
            row = {'case': name, 'candidate': int(index), 'stage': stage, 'world_goal': arrays['goals'][index].tolist(),
                   'ik_found': found is not None, 'witness': found, 'best_failed': None if found else best}
            rows.append(row);all_rows.append(row)
        save(OUT / (name + '_独立复核.json'), {'queries': rows, 'physical_attempts': 0})
        print('CPU_IK', name, 'queries', len(rows), 'recovered', sum(r['ik_found'] for r in rows), flush=True)
    save(OUT / '独立运动学汇总.json', {'queries': all_rows, 'physical_attempts': 0,
         'wall_seconds': time.monotonic()-began, 'not_found_is_not_a_proof_of_infeasibility': True})


if __name__ == '__main__':main()
