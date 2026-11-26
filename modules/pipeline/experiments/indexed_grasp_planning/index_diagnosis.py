"""Measure fixed-map hit rates and query costs before spending planning runs."""
import argparse
import importlib
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

from arx_project.assets import resource_path
from arx_project.layout import ROOT,ensure_run_path,now,read_json,sha256,write_json
from modules.curobo.indexed_planning.index_reuse.lookup import FullPoseIndex,MAP
from .replay import verify_hashes


def prepare(root):
    if root.exists():raise ValueError('Use a fresh E registration; preserve previous records')
    config=read_json(ROOT/'configs/experiments/indexed_grasp_planning.json')
    source=ROOT/config['native_batch_implementation']['full_run']
    if not read_json(source/'independent_review.json')['passed']:raise ValueError('Baseline is not independently accepted')
    cases=[];inputs={};queries=0;seed_queries=0
    for case in read_json(source/'registration.json')['cases']:
        directory=Path(case['source_directory'])
        paths=[directory/'state.json',directory/'candidates/candidate_pool.json']
        arms={}
        for side in ['right','left']:
            file=source/'plans'/case['id']/'control/arms'/side/'plan_result.json'
            if file.is_file():
                paths.append(file)
                events=read_json(file).get('candidate_events',[])
                indices=[int(event['candidate_index']) for event in events]
            else:indices=[]
            arms[side]=indices;queries+=len(indices)
            seed_queries+=len(set([indices[0],indices[-1]])) if indices else 0
        cases.append({**case,'checked_candidate_indices':arms})
        inputs.update({str(p):sha256(p) for p in paths})
    if queries!=10322 or len(cases)!=69:raise ValueError('Unexpected original control-query workload')
    sources=[Path(__file__),ROOT/'modules/curobo/indexed_planning/geometric/kinematics.py',
        ROOT/'modules/curobo/dual_arm/model.py',ROOT/'modules/curobo/fingertip_contact/geometry.py',
        resource_path('arx_robot')/'robot.urdf',ROOT/'modules/reachability/工具/能力索引.py',
        ROOT/'modules/reachability/工具/公共.py']
    sources+=list((ROOT/'modules/curobo/indexed_planning/index_reuse').glob('*.py'))
    root.mkdir(parents=True)
    for path in sources:
        target=root/'source_registered'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    complete=read_json(MAP/'完成.json')
    for name in ['完成.json','地图元数据.json',read_json(MAP/'地图元数据.json')['witness_file']]:
        inputs[str(MAP/name)]=sha256(MAP/name)
    write_json(root/'registration.json',{'at':now(),'experiment':'index_lookup_diagnosis',
        'authorization':'继续E索引复用；补测3/5/10度近邻范围，最终原目标与验收容差保持。',
        'source_run':str(source),'cases':cases,'map_directory':str(MAP),
        'query_workload':'all_10322_candidate_checks_in_original_69_control_requests',
        'direct_queries':queries,'neighborhood_queries':queries,'seed_query_fixtures':seed_queries,
        'angular_limits_rad':[.03,*np.deg2rad([3,5,10])],
        'seed_position_radius_m':.02,'final_EEF_position_tolerance_m':.002,'final_rotation_tolerance_rad':.03,
        'frozen_map_has_no_source_grasp_solutions_added':True,'random_self_checks':128,
        'random_seed':20261126,'physical_grasps':0,'planning_requests':0,'GPU_calls':0,
        'source_hashes':{str(p):sha256(p) for p in sources},'input_hashes':inputs})
    write_json(root/'progress.json',{'at':now(),'phase':'registered','completed_queries':0,'total_queries':queries,
        'completed_requests':0,'total_requests':0})


def stats(values):
    return {'queries':len(values),'median_ms':float(np.median(values)*1000),
        'p90_ms':float(np.percentile(values,90)*1000),'total_seconds':float(np.sum(values))}


def run(root):
    if (root/'started.json').exists() or (root/'STOP').exists():raise ValueError('Do not repeat a registered E diagnosis')
    reg=read_json(root/'registration.json');verify_hashes(reg)
    write_json(root/'started.json',{'at':now()})
    index=FullPoseIndex(Path(reg['map_directory']))
    rng=np.random.default_rng(reg['random_seed']);self_checks=[]
    for row_id in rng.integers(0,len(index.bank),reg['random_self_checks']):
        row=np.asarray(index.bank[row_id],dtype=float)
        actual=index.chain.fk(row[:6])[0]
        answer=index.direct(actual,row[:6])
        if not any(w['bank_row']==int(row_id) for w in answer['witnesses']):
            raise RuntimeError('The exact map index missed a known original witness')
        from scipy.spatial.transform import Rotation
        action=actual@index.tcp
        pe=float(np.linalg.norm(action[:3,3]-row[6:9]))
        re=float(Rotation.from_matrix(action[:3,:3]@Rotation.from_quat(row[[10,11,12,9]]).as_matrix().T).magnitude())
        if pe>2e-6 or re>2e-6:raise RuntimeError('Current physical FK differs from the frozen witness bank')
        self_checks.append({'bank_row':int(row_id),'position_error_m':pe,'rotation_error_rad':re})
    sys.path.insert(0,str(ROOT/'modules/reachability/工具'))
    legacy_type=importlib.import_module('能力索引').CapabilityMap
    legacy=legacy_type.__new__(legacy_type);legacy.bank=index.bank;legacy.tree=index.tree
    legacy.bounds=index.chain.limits;legacy.coverage=lambda *args:{}
    write_json(root/'startup.json',{'at':now(),'index_load_seconds':index.load_seconds,
        'witnesses':len(index.bank),'self_checks':self_checks,'passed':True,
        'index_TCP_radius_m':index.radius,'direct_acceptance_is_original_EEF_tolerance':True})
    totals={str(a):0 for a in reg['angular_limits_rad']};rows=[];hit_records=[];seed_rows=[]
    times=[];neighborhood_times=[];completed=0;equivalence_checks=0
    for case in reg['cases']:
        directory=Path(case['source_directory']);state=read_json(directory/'state.json')
        candidates=read_json(directory/'candidates/candidate_pool.json')['candidates']
        case_direct=0;case_angles={a:0 for a in totals};case_queries=0
        for side,indices in case['checked_candidate_indices'].items():
            base=np.asarray(state['links']['body4']).copy()
            if side=='left':base=base@np.array([[1,0,0,0],[0,1,0,.5],[0,0,1,0],[0,0,0,1.]])
            inverse=np.linalg.inv(base)
            from modules.arx_core.arms import get_arm
            arm=get_arm(side)
            bounds=np.asarray([state['joint_limits'][state['joint_names'].index(n)] for n in arm.joints])
            current=np.array([state['qpos'][state['joint_names'].index(n)] for n in arm.joints])
            current=np.maximum(np.minimum(current,bounds[:,1]-1e-4),bounds[:,0]+1e-4).astype(np.float32)
            fixtures=set([indices[0],indices[-1]]) if indices else set()
            for candidate_index in indices:
                contact=np.asarray(candidates[candidate_index]['pose']).copy()
                contact[:3,3]-=.043*contact[:3,2]
                eef=inverse@contact
                direct=index.direct(eef,current);times.append(direct['seconds'])
                nearby=index.neighborhood(eef,current);neighborhood_times.append(nearby['seconds'])
                completed+=1;case_queries+=1
                if direct['witnesses']:
                    case_direct+=1;hit_records.append({'id':case['id'],'arm':side,'candidate_index':candidate_index,
                        'eef_goal_body4':eef.tolist(),'witnesses':direct['witnesses']})
                for key,count in nearby['angular_counts'].items():
                    if count:case_angles[key]+=1;totals[key]+=1
                if candidate_index in fixtures:
                    action=index.action_pose(eef)
                    legacy_result=legacy.query_pose(action,current_q=current,num_seeds=32,neighbors=8192)
                    if equivalence_checks<32:
                        same=index.seeds(eef,current)
                        if not np.array_equal(same['seeds'],legacy_result['seeds']):
                            raise RuntimeError('The reference map seed strategy changed')
                        equivalence_checks+=1
                    short=index.seeds(eef,current,neighbors=1024)
                    ranked=index.seeds(eef,current,neighbors=8192,diversify=False)
                    seed_rows.append({'id':case['id'],'arm':side,'candidate_index':candidate_index,
                        'legacy_8192_seconds':legacy_result['search_seconds'],
                        'position_1024_seconds':short['seconds'],'rank_only_8192_seconds':ranked['seconds'],
                        'neighborhood_seconds':nearby['seconds'],
                        'seed_counts_by_angle':nearby['angular_counts']})
        rows.append({'id':case['id'],'queries':case_queries,'direct_hits':case_direct,'neighborhood_hits_by_angle':case_angles})
        write_json(root/'progress.json',{'at':now(),'phase':'querying','completed_queries':completed,
            'total_queries':reg['direct_queries'],'completed_requests':0,'total_requests':0,
            'completed_cases':len(rows),'cases':rows})
        print('MAP_QUERY',case['id'],completed,'direct_hits',sum(r['direct_hits'] for r in rows),flush=True)
    verify_hashes(reg)
    result={'at':now(),'passed':True,'record_integrity':True,'task_complete':completed==reg['direct_queries'],
        'queries':completed,'cases':rows,'direct_hits':len(hit_records),'direct_witnesses':hit_records,
        'neighborhood_hits_by_angle':totals,'direct_query_timing':stats(times),
        'neighborhood_timing':stats(neighborhood_times),'seed_fixture_queries':len(seed_rows),
        'seed_query_timing':{name:stats([r[name] for r in seed_rows]) for name in
            ['legacy_8192_seconds','position_1024_seconds','rank_only_8192_seconds','neighborhood_seconds']},
        'legacy_seed_equivalence_checks':equivalence_checks,'seed_results':seed_rows,
        'planning_requests':0,'GPU_calls':0,'physical_grasps':0,'target_pose_changed':False}
    write_json(root/'acceptance.json',result)
    write_json(root/'progress.json',{'at':now(),'phase':'complete','completed_queries':completed,
        'total_queries':reg['direct_queries'],'completed_requests':0,'total_requests':0})
    print('MAP_QUERY_COMPLETE',{k:v for k,v in result.items() if k not in ['cases','direct_witnesses','seed_results']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=ensure_run_path(args.root)
    if args.stage=='prepare':prepare(root)
    else:
        try:run(root)
        except BaseException:
            write_json(root/'STOP',{'at':now(),'error':traceback.format_exc()});raise
