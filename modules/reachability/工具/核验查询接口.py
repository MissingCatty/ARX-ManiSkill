"""Functional and fault checks for map queries, with no simulator access."""
import argparse
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from 公共 import ROOT,save
from 能力索引 import CapabilityMap


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--地图',type=Path,default=ROOT/'产物/首版_01');args=parser.parse_args()
    atlas=CapabilityMap(args.地图);checks=[]
    assert int(atlas.cells['sample_count'].sum())==atlas.meta['witnesses'];checks.append('体素样本数与见证数量一致')
    assert (atlas.cells['orientation_count']<=1152).all();assert (atlas.cells['down_orientation_count']<=288).all()
    checks.append('朝向覆盖的分母与上界一致')
    rng=np.random.default_rng(77120)
    for index in rng.choice(len(atlas.bank),8,replace=False):
        row=atlas.bank[index];pose=np.eye(4);pose[:3,3]=row[6:9]
        pose[:3,:3]=Rotation.from_quat(row[9:13][[1,2,3,0]]).as_matrix()
        result=atlas.query_pose(pose)
        assert result['direct_fk_witness_within_original_pose_tolerances']
        assert len(result['seeds'])==32 and len(result['seed_bank_rows'])==32
    checks.append('已有FK见证能被按位姿检索，种子可追溯')
    action=pose.copy();eef=action@np.linalg.inv(atlas.tcp)
    body=atlas.query({'matrix':action.tolist(),'frame':'body4'})
    world=atlas.query({'matrix':(atlas.base@eef).tolist(),'frame':'world','tool':'right_eef_link'})
    assert np.allclose(body['query_pose_body4_action'],world['query_pose_body4_action'],atol=1e-9)
    checks.append('世界/基座与EEF/TCP坐标转换一致')
    unknown=atlas.coverage(np.array([3.,3.,3.]))
    assert not unknown['covered_position_voxel'] and not unknown['uncovered_is_unreachable'] and not unknown['hard_candidate_rejection']
    checks.append('未覆盖返回未知，不判不可达或否决候选')
    try:atlas.query({'matrix':action.tolist(),'frame':'bad_frame'})
    except ValueError:checks.append('错误坐标系被拒绝')
    else:raise AssertionError('Unknown frame accepted')
    with tempfile.TemporaryDirectory(prefix='arx_map_check_') as directory:
        save(Path(directory)/'完成.json',{'passed':False})
        try:CapabilityMap(directory)
        except ValueError:checks.append('未完成地图不能被查询')
        else:raise AssertionError('Incomplete map accepted')
    save(args.地图/'查询接口核验.json',{'passed':True,'checks':checks,'map_load_seconds':atlas.load_seconds,'physical_attempts':0})
    print('QUERY_CHECKS_PASSED',len(checks),flush=True)


if __name__=='__main__':main()
