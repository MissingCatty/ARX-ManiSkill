"""Concurrent summaries with explicit denominators and failure stages."""
from collections import Counter
import fcntl
from pathlib import Path

from arx_project.artifacts import atomic_json
from arx_project.json_io import read_json
from arx_project.layout import now
from ..metrics import distribution
from .contracts import METHODS, completed_rows

FAILURE_LABELS = {
    'stationing':'共同高可达站位未通过', 'pointcloud':'点云未生成',
    'empty_candidates':'没有候选', 'initial_planning':'初始规划不出来',
    'approach_replanning':'接近重规划不通过', 'lift_replanning':'闭合后抬升重规划不通过',
    'opening':'开爪未到位','pregrasp':'预抓取运动未到位','approach':'接近运动未到位',
    'close':'闭合未到位','lift':'抬升运动未到位','hold':'保持动作未到位',
    'bilateral_contact':'未形成双指夹持','hold_height':'抬升或保持高度不足',
    'forbidden_contact':'禁止接触','other':'其他抓取失败'}


def failure_category(result):
    reason=result['reason']
    if result.get('success'):return None
    if reason=='unable_to_reach_high_workspace':return 'stationing'
    if reason=='rayst3r_empty_pointcloud':return 'pointcloud'
    if reason.endswith('_no_postprocessed_candidate'):return 'empty_candidates'
    if reason=='no_arm_has_executable_candidate':return 'initial_planning'
    if reason=='measured_pregrasp_approach_unreachable':return 'approach_replanning'
    if reason=='measured_closure_lift_unreachable':return 'lift_replanning'
    if reason in ['gripper_open_not_converged','open_not_converged']:return 'opening'
    for stage in ['pregrasp','approach','close','lift','hold']:
        if reason==stage+'_not_converged':return stage
    return {'no_bilateral_target_contact':'bilateral_contact','insufficient_hold_height':'hold_height',
            'forbidden_robot_contact':'forbidden_contact'}.get(reason,'other')


def slot_counts(claimed, completed, *, current=None, active=False, total=74):
    claimed,completed=set(claimed),set(completed)
    if not completed<=claimed or len(claimed)>total:
        raise RuntimeError('Completed/claimed ledger does not fit the registered cohort')
    pending=claimed-completed
    running={current}&pending if active else set()
    return {'interrupted':len(pending-running),'in_progress':len(running),'not_started':total-len(claimed)}


def summarize(root, phase=None):
    root=Path(root)
    with (root/'summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        reg=read_json(root/'registration.json');methods={};indexes={}
        for method in METHODS:
            folder=root/'methods'/method;rows=completed_rows(folder)
            indexes[method]={r['id']:r for r in rows}
            completed=len(rows)
            claimed={p.stem for p in (folder/'claims').glob('*.json')}
            active=((folder/'started.json').exists() and not any(p.exists() for p in
                [folder/'STOP',folder/'complete.json',root/'STOP',root/'ABORT']))
            current=read_json(folder/'current_trial.json')['scene'] if (folder/'current_trial.json').exists() else None
            counts=slot_counts(claimed,indexes[method],current=current,active=active)
            stages=sorted({k for r in rows for k in r['latency'].get('timings',{})})
            failures=Counter(failure_category(r['result']) for r in rows if not r['result']['success'])
            methods[method]={'completed':completed,'registered':74,**counts,
                'planning_passes':sum(bool(r['result'].get('planning_success')) for r in rows),
                'successes':sum(bool(r['result']['success']) for r in rows),
                'failures':dict(failures),
                'latency':{stage:distribution([r['latency'].get('timings',{}).get(stage) for r in rows
                    if not r['latency'].get('common_precondition_failure')]) for stage in stages}}
        paired=[key for key in [p['id'] for p in reg['cohort']['pairs']]
                if all(key in indexes[m] for m in METHODS)]
        paired_results=[{'id':key,**{m:{'success':indexes[m][key]['result']['success'],
            'planning_success':bool(indexes[m][key]['result'].get('planning_success')),
            'reason':indexes[m][key]['result']['reason']} for m in METHODS}} for key in paired]
        if phase is None:
            active=[m for m in METHODS if (root/'methods'/m/'started.json').exists() and not (root/'methods'/m/'complete.json').exists()]
            phase=('stopped' if (root/'STOP').exists() else 'complete' if all((root/'methods'/m/'complete.json').exists() for m in METHODS)
                else 'running' if any((root/'methods'/m/'preflight/release.json').exists() for m in active)
                else 'warming' if active else 'waiting_for_resources')
        result={'at':now(),'phase':phase,'registered_pairs':74,'completed_pairs':len(paired),
            'methods':methods,'paired_results':paired_results,'failure_labels':FAILURE_LABELS,
            'sam3_excluded':25,'original_interrupted':1,'common_stationing_failures_registered':5}
        atomic_json(root/'summary.json',result)
        atomic_json(root/'progress.json',{k:v for k,v in result.items() if k!='paired_results'})
        return result
