"""Additional independent audit for the ordinary H013 preshape action."""
import json
from pathlib import Path
import numpy as np

from ring_tilt_candidates import INNER_CENTER_OFFSET_M
from settling import assess_settling


def audit(folder):
    folder=Path(folder);read=lambda n:json.loads((folder/n).read_text())
    policy=read('ring_policy.json') if (folder/'ring_policy.json').exists() else {}
    exists=(folder/'ring_preshape_execution.json').exists()
    guard_issues=[];metric_issues=[];row=None
    if policy.get('applied') and not exists and (folder/'approach_execution.json').exists():
        guard_issues.append('Ring approach executed without archived preshape')
    if exists:
        try:
            snap=read('capture_snapshot.json');names=snap['right_arm']
            command=read('ring_preshape_command.json');execution=read('ring_preshape_execution.json')
            selected=read('selected_grasp.json');report=read('ring_preshape_report.json')
            cfg=read('initial_fingerprint.json')['invariant']['configured']['controllers']['gripper']
            change=selected['research_change'];fit=read('ring_fit.json');a,b=fit['ellipse_radii_m'];theta=change['ring_angle_rad'];sign=change['closing_sign']
            desired=1./np.sqrt((np.cos(theta)/a)**2+(np.sin(theta)/b)**2)-INNER_CENTER_OFFSET_M[sign]
            if abs(desired-change['preshape_finger_joint_m'])>1e-9:guard_issues.append('Preshape opening differs from frozen-CAD normal-motion proposal')
            api=cfg.get('api_closed',0.)+(desired-cfg['closed_m'])/(cfg['open_m']-cfg['closed_m'])*(cfg.get('api_open',-3.2)-cfg.get('api_closed',0.))
            q=np.asarray(command['position']);times=np.asarray(command['time_from_start'])
            limits=np.asarray([snap['joint_limits'][snap['joint_names'].index(n)] for n in names])
            if command['joint_names']!=names or q.shape!=(2,len(names)) or times.shape!=(2,):
                guard_issues.append('Wrong preshape command shape or ARX joint order')
            if not np.isfinite(q).all() or not np.isfinite(times).all() or not np.array_equal(q[0],q[1]) or not np.array_equal(times,[0.,3.]):
                guard_issues.append('Preshape must hold a finite constant arm target for3seconds')
            maximum=float(np.maximum(limits[:,0]-q,q-limits[:,1]).clip(0).max())
            if maximum>1e-6:guard_issues.append('Preshape arm command exceeds frozen limits')
            previous=read('pregrasp_execution.json')['records'][-1]['target']
            if not np.array_equal(q[0],np.asarray(previous,dtype=np.float32)):
                guard_issues.append('Preshape changed last accepted arm target')
            if abs(command['gripper_target']-api)>1e-12:
                guard_issues.append('Preshape command does not use unchanged API mapping')
            targets=np.asarray([record['target'] for record in execution['records']])
            if targets.ndim!=2 or targets.shape[1]!=len(names) or not np.allclose(targets,q[0],atol=1e-7,rtol=0):
                guard_issues.append('Recorded preshape targets do not match constant commanded arm target')
            if (folder/'approach_command.json').exists() and abs(read('approach_command.json')['gripper_target']-api)>1e-12:
                guard_issues.append('Ring approach reopened or changed the preshape opening')
            external=[c for c in execution['contacts'] if c['num_contacts']>0 and not ('/World/Robot/' in c['a'] and '/World/Robot/' in c['b'])]
            if external:guard_issues.append('Physical external contact during overhead preshape')
            assessment=assess_settling(execution,selected['pregrasp_world'])
            if not assessment['accepted']:metric_issues.append('Preshape did not meet original arm convergence criteria')
            final=execution['final'];measured={n:final['joint_positions'][final['joint_names'].index(n)] for n in ('right_joint27','right_joint28')}
            finger_error=max(abs(v-desired) for v in measured.values())
            if finger_error>.0001:metric_issues.append('Preshape fingers did not reach prescribed ordinary opening')
            if not report.get('accepted'):metric_issues.append('Preshape state update/completion was not accepted')
            row={'stage':'ring_preshape','planned_max_limit_exceed_rad':maximum,'external_contact_events':len(external),
                 'convergence':assessment,'measured_fingers_m':measured,'finger_goal_max_error_m':finger_error}
        except Exception as exc:guard_issues.append(repr(exc))
    result={'pass':not guard_issues and not metric_issues,'guard_pass':not guard_issues,
            'extra_completed_actions':int(exists),'stage':row,'guard_violations':guard_issues,'metric_failures':metric_issues,
            'scope':'Extra overhead preshape action only. Original core five-stage evaluator unchanged; no optimizer needed for a constant arm command.'}
    (folder/'extra_action_evaluation.json').write_text(json.dumps(result,indent=2));return result
