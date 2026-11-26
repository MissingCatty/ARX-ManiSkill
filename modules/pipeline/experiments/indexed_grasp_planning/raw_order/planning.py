"""Run the matched full69 comparison through the unchanged execution planner API."""
import json
from pathlib import Path
import shutil
import time

from arx_project.layout import ROOT, now, sha256, write_json
from ..physical.registration import guard, raw
from .proxy import OriginalOrderProxy


def comparison(pair):
    a, b = pair['control_1'], pair['optimized_1']
    return {'control_success_retained': not a['success'] or b['success'],
        'same_grasp_and_arm': all(a[k] == b[k] for k in
            ['success','selected_arm','selected_original_source_index','contact_pose_world']),
        'same_joint_endpoints': a['joint_endpoints'] == b['joint_endpoints'],
        'same_candidate_decisions': a['event_decisions'] == b['event_decisions']}


def run(root, socket):
    reg = guard(root); proxy = OriginalOrderProxy(root, socket); rows = []; completed = 0
    apple = next(c for c in reg['cases'] if c['id']=='M0001_S1_open_apple')
    for mode in ['baseline','optimized']:
        directory = root/'preflight'/('stationary_'+mode); directory.mkdir(parents=True)
        shutil.copy2(Path(apple['source_directory'])/'state.json',directory/'state.json')
        if not proxy.request(directory,execution_variant=mode,case_id=apple['id'])['success']:
            raise RuntimeError('Raw-order planning warmup failed')
    write_json(root/'preflight/all_ready.json',{'at':now(),'passed':True,
        'prewarm_planning_requests':2,'physical_grasps':0,'NMS_enabled':False})
    for ordinal, case in enumerate(reg['cases']):
        guard(root, full=False); source = Path(case['source_directory']); pair = {}
        order = ['control_1','optimized_1'] if ordinal % 2 == 0 else ['optimized_1','control_1']
        for variant in order:
            mode = reg['variant_modes'][variant]
            directory = root/'plans'/case['id']/variant
            claim = root/'claims'/(case['id']+'_'+variant+'.json')
            if directory.exists() or claim.exists(): raise ValueError('A planning slot cannot be retried')
            directory.mkdir(parents=True)
            with claim.open('x') as stream: json.dump({'at':now(),'state':'claimed'},stream)
            shutil.copy2(source/'state.json',directory/'state.json')
            write_json(root/'progress.json', {'at':now(),'phase':'planning','completed_requests':completed,
                'total_requests':138,'current':{'id':case['id'],'variant':mode},'results':rows})
            print('RAW_ORDER_PLAN_BEGIN',completed+1,case['id'],mode,flush=True)
            began = time.perf_counter()
            plan = proxy.request(directory,execution_variant=mode,case_id=case['id'])
            seconds = time.perf_counter()-began
            io = raw(directory/'candidate_io.json')
            if io['source'] != 'shared_memory' or io['decode_count'] != 1 or io['full_arm_pool_copies'] != 0:
                raise RuntimeError('Original single-decode shared-memory contract changed')
            selected = plan.get('selected',{}); decisions = {}
            for side in ['right','left']:
                arm = raw(directory/'arms'/side/'plan_result.json')
                decisions[side] = [{k:e[k] for k in ['candidate_index','kind','kinematic_solutions','reasons'] if k in e}
                    for e in arm.get('candidate_events',[])]
            metrics = {'success':plan['success'],'selected_arm':selected.get('arm'),
                'selected_original_source_index':selected.get('original_source_index'),
                'contact_pose_world':selected.get('contact_pose_world'),
                'joint_endpoints':{k:selected.get(k) for k in ['q_pre','q_grasp','q_lift']},
                'event_decisions':decisions,'candidate_checks':raw(directory/'interleaved_search.json')['candidate_checks'],
                'initial_planning_with_postprocessing_seconds':seconds,
                'execution_planner':raw(directory/'execution_planner_initial.json')}
            write_json(directory/'metrics.json',metrics); pair[variant] = metrics; completed += 1
            evidence = ['execution_planner_initial.json','planner_rpc_initial.json','candidate_io.json',
                'candidate_preparation.json','candidate_snapshot_commit.json','interleaved_search.json',
                'arms/right/plan_result.json','arms/left/plan_result.json']
            write_json(claim,{'at':now(),'state':'complete',
                'metrics_sha256':sha256(directory/'metrics.json'),'plan_sha256':sha256(directory/'plan_result.json'),
                'order_sha256':sha256(directory/'candidate_order.json'),
                'evidence_sha256':{name:sha256(directory/name) for name in evidence}})
            print('RAW_ORDER_PLAN_END',completed,case['id'],mode,plan['success'],round(seconds,3),flush=True)
        compared = comparison(pair)
        rows.append({'id':case['id'],'variants':pair,'comparison':compared})
        write_json(root/'progress.json',{'at':now(),'phase':'between_pairs','completed_requests':completed,
            'total_requests':138,'results':rows})
        if not compared['control_success_retained'] or not compared['same_grasp_and_arm']:
            raise RuntimeError('Original grasp/arm or planning success changed; stop before physical follow-up')
    guard(root)
    write_json(root/'acceptance.json',{'at':now(),'task_complete':completed==138,'record_integrity':True,
        'control_successes_retained':all(r['comparison']['control_success_retained'] for r in rows),
        'original_grasps_and_arms_retained':all(r['comparison']['same_grasp_and_arm'] for r in rows),
        'physical_grasps':0,'planning_requests':completed,'cases':rows})

