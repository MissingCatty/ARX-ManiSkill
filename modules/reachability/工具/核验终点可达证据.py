"""Independently verify FK/limits; keep endpoint reachability separate from paths."""
from 公共 import EXPERIMENTS
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from 诊断七轴抓取终点 import ROOT,MS,read,save,sha
sys.path.insert(0,str(MS))
from 七轴升降规划 import BodySweep,NAMES

def main():
    source=EXPERIMENTS / '抓取位姿局部修正_20261121_01'
    run=EXPERIMENTS / '抓取位姿局部修正_20261121_02'
    reg=read(run/'登记.json');fk=BodySweep();rows=[];sources=[Path(__file__),MS/'assets/arx/robot.urdf',MS/'七轴升降规划.py']
    for case in reg['cases']:
        state_path=Path(case['source'])/'state.json';state=read(state_path);sources.append(state_path)
        limits=np.array([state['joint_limits'][state['joint_names'].index(n)] for n in NAMES])
        diag=read(source/'诊断'/case['case']/'汇总.json')
        previous=[r for r in diag['rows'] if r.get('pure_ik',{}).get('independently_valid',0)>0]
        rows.append({'case':case['case'],'role':case['role'],'mode':'original','verified_reachable':bool(previous),
          'meaning':'At least one original full-pose kinematic witness in the prior bounded diagnostic search; collision is separate',
          'witnesses_recorded_in':str(source/'诊断'/case['case']),'unverified_is_global_unreachable':False})
        for mode in ['local_search','map_local_search']:
            p=run/'比较'/case['case']/mode/'结果.json'
            if not p.exists():continue
            sources.append(p);d=read(p);selected=d['selected']
            if not selected:
                rows.append({'case':case['case'],'role':case['role'],'mode':mode,'verified_reachable':False,
                  'status':'No retained joint witness in this output; not a proof of unreachability'})
                continue
            q=np.array(selected['q_grasp_7']);goal=np.array(selected['variant']['eef_goal_world'])
            actual=fk.fk(state,dict(zip(NAMES,q)))['right_eef_link']
            position=float(np.linalg.norm(actual[:3,3]-goal[:3,3]))
            rotation=float(Rotation.from_matrix(actual[:3,:3]@goal[:3,:3].T).magnitude())
            bounded=bool(((q>=limits[:,0]-1e-6)&(q<=limits[:,1]+1e-6)).all())
            passed=bool(position<=.002001 and rotation<=.030001 and bounded)
            assert passed,(case['case'],mode,position,rotation,bounded)
            rows.append({'case':case['case'],'role':case['role'],'mode':mode,'verified_reachable':passed,
              'joint_names':NAMES,'q_witness':q.tolist(),'position_error_m':position,'orientation_error_rad':rotation,
              'joint_limits_valid':bounded,'eef_goal_world':goal.tolist(),'independent_fk_world':actual.tolist(),
              'position_tolerance_m':.002,'orientation_tolerance_rad':.03,'numeric_tolerance_m_or_rad':1e-6,
              'angle_change_deg':selected['variant']['angle_deg'],'original_candidate_index':selected['variant']['source_index'],
              'collision_and_path_result_is_separate':True,'physical_grasp_quality_verified':False,'physical_attempts':0})
    summary={}
    for role in ['diagnostic_failure','successful_control']:
        summary[role]={}
        for mode in ['original','local_search','map_local_search']:
            chosen=[r for r in rows if r['role']==role and r['mode']==mode]
            summary[role][mode]={'completed':len(chosen),'verified_reachable':sum(r['verified_reachable'] for r in chosen)}
    target=ROOT/'docs/resource/抓取位姿局部修正首轮/终点位姿可达证据.json'
    save(target,{'primary_objective':'Full grasp endpoint reachability only, not path/holding success',
       'scope':'Independent URDF FK and unchanged joint-limit verification of retained solver joint witnesses',
       'physical_attempts':0,'summary':summary,'rows':rows,'source_sha256':{str(p):sha(p) for p in sources}})
    print(summary,flush=True)

if __name__=='__main__':main()
