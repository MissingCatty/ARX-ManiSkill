"""Refine the legal upper-height band on the unchanged, matched pose probes."""
import argparse
from pathlib import Path
import shutil

import numpy as np
import 比较升降高度 as base
import 比较升降高度_参考点修正 as corrected

SELF = Path(__file__).resolve()
PRIOR = base.ROOT / '产物/高度比较_20261120_03'
HEIGHTS = [.22] + [round(value / 1000, 3) for value in range(320, 461, 2)]


def prepare(out, gpu):
    old = base.read(PRIOR / '高度扫描.json')
    assert old['passed'] and max(old['ranked_lifts_m']) == .46
    base.LIFTS = HEIGHTS
    base.prepare(out, gpu)
    reg = base.read(out / '登记.json')
    for path in [SELF, Path(corrected.__file__), PRIOR/'登记.json', PRIOR/'高度扫描.json']:
        reg['source_sha256'][str(path)] = base.sha(path)
    reg.update(
        phase='Fine height selection before the separately registered 400 physical tests',
        predecessor=str(PRIOR), prior_height_results_preserved=True,
        fine_height_step_m=.002, fine_height_band_m=[.32,.46],
        paired_reference_height_m=.22,
        query_budget=len(HEIGHTS)*reg['query_poses'],
        selection='Rank the same vertical-probe coverage, then any-yaw coverage, then smaller height change. Confirm finalists on the actual frozen two-pipeline candidate pools before physical testing.',
        physical_attempts=0, robot_motion_commands=0,
        downstream_physical_budget=400,
        downstream_design='At the selected single height, ZeroGrasp+cuRobo and ZeroGrasp+GraspCL+cuRobo, 200 original scenes each. Fresh observations. No MPC or other algorithm changes.',
    )
    base.save(out/'登记.json', reg)
    for path in [SELF, Path(corrected.__file__)]:
        shutil.copy2(path, out/'代码快照'/path.name)
    prior_data = np.load(PRIOR/'固定查询位姿.npz')
    actual_data = np.load(out/'固定查询位姿.npz')
    assert all(np.array_equal(prior_data[key], actual_data[key]) for key in ['positions','rotations'])
    base.save(out/'预检.json', {'passed':True,'same_world_pose_probes':True,
        'legal_joint4_limits_m':reg['legal_lift_limits_m'],'height_count':len(HEIGHTS),
        'query_budget':reg['query_budget'],'robot_geometry_and_success_criteria_changed':False,
        'selected_height_not_yet_fixed':True,'physical_attempts':0})
    print('PREPARED',len(HEIGHTS),'heights',reg['query_budget'],'matched pose probes',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','scan'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu')
    args=parser.parse_args();out=args.output.resolve()
    if args.stage=='prepare':prepare(out,args.gpu)
    else:
        base.LIFTS=base.read(out/'登记.json')['lift_values_m']
        corrected.compiled_scan()(out)
