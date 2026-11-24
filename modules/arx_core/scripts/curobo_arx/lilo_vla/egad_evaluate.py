"""Read-only adaptation of the original ARX execution gates to policy steps.

No synthetic approach/lift files are made. The reported phase membership points
to the real commands around an actual three-second bilateral closure.
"""
import json
import math
from pathlib import Path


def evaluate(folder, *, sequence=None):
    folder=Path(folder);issues=[];missing=[];fingers=set();heights=[]
    def read(name):return json.loads((folder/name).read_text())
    result=read('grasp_report.json') if (folder/'grasp_report.json').exists() else {}
    if not (folder/'capture_snapshot.json').exists():
        return {'pass':False,'outcome':'observation_or_setup_failure','issues':['Missing fresh capture snapshot']}
    snapshot=read('capture_snapshot.json');target='test_object'
    target_path=snapshot['objects'][target]['path'];initial_z=snapshot['objects'][target]['world'][2][3]
    labels=list(sequence or result.get('stage_timing',{}))
    # Tests may map phases to real historical ARX filenames without copying or
    # inventing executions. Normal evaluation reads the actual stage labels.
    entries=[(label,label) if isinstance(label,str) else tuple(label) for label in labels]
    names=[semantic for semantic,source in entries]
    closed=names.index('close') if 'close' in names else -1
    phases={'pregrasp':[],'approach':[],'close':[],'lift':[],'hold':[]}
    for index,(semantic,label) in enumerate(entries):
        phase='pregrasp' if semantic=='pregrasp_via' else (semantic if semantic in phases else ('approach' if closed<0 or index<closed else 'lift'))
        phases[phase].append(label)
    for phase,values in phases.items():
        if not values:missing.append(phase)
    executed=[];converged=[];forbidden=[];planned_max=0.;dense_invalid=0;hold_duration=0.
    listed={source for semantic,source in entries}
    unlisted=[p.name for p in folder.glob('*_command.json') if p.name.removesuffix('_command.json') not in listed]
    if unlisted:issues.append({'unaccounted_commands':unlisted})
    for semantic,label in entries:
        required=[label+suffix for suffix in ('_command.json','_execution.json','_settling.json')]
        if any(not (folder/name).exists() for name in required):
            missing.append(label);continue
        command=read(label+'_command.json');execution=read(label+'_execution.json');settling=read(label+'_settling.json')
        executed.append(label)
        if settling.get('accepted') is True:converged.append(label)
        else:issues.append({'stage':label,'error':'Original ARX convergence gate rejected execution'})
        if command['joint_names']!=snapshot['right_arm']:
            issues.append({'stage':label,'error':'Wrong ARX joint order'})
        limits=[snapshot['joint_limits'][snapshot['joint_names'].index(n)] for n in command['joint_names']]
        for q in command['position']:
            if len(q)!=len(limits):issues.append({'stage':label,'error':'Malformed joint command'});break
            values=[max(lo-v,v-hi,0.) for v,(lo,hi) in zip(q,limits)]
            if any(not math.isfinite(v) for v in q):issues.append({'stage':label,'error':'Nonfinite joint command'});break
            planned_max=max(planned_max,*values)
        for event in execution.get('contacts',[]):
            if event['num_contacts']<=0:continue
            a,b=event['a'],event['b'];pair=a+b
            if '/World/Robot/' in a and '/World/Robot/' in b:continue
            if not (target_path in pair and any(n in pair for n in ('right_link27','right_link28'))):
                forbidden.append({'stage':label,**event})
            if semantic=='close' and target_path in pair:
                fingers.update(n for n in ('right_link27','right_link28') if n in pair)
        if label.startswith('local_'):
            path=folder/(label+'_validation.json')
            if not path.exists():missing.append(label+'_validation');continue
            cert=read(label+'_validation.json')
            candidates=cert.get('candidates',[])
            chosen=candidates[-1] if candidates else cert
            if cert.get('accepted',not candidates) is not True or chosen.get('rejection'):
                issues.append({'stage':label,'error':'Invalid cuRobo local-path certificate'})
            dense_invalid+=int(chosen.get('invalid_samples',1))
        elif semantic in ('pregrasp_via','pregrasp','approach','lift'):
            path=folder/(label+'_optimization.json')
            if not path.exists():missing.append(label+'_optimization');continue
            cert=read(label+'_optimization.json');value=float(cert['max_limit_exceed_rad'])
            if not cert.get('success') or not math.isfinite(value):
                issues.append({'stage':label,'error':'Invalid original planning certificate'})
            else:planned_max=max(planned_max,value)
            dense_invalid+=int(cert['invalid_samples'])
        if semantic=='close' and float(execution['sim_duration_s'])<2.99:
            issues.append({'stage':label,'error':'Original closing dwell incomplete'})
        if semantic=='hold':
            heights=[r['objects'][target][2][3]-initial_z for r in execution['records']]
            hold_duration=float(execution['sim_duration_s'])
            times=[r['time'] for r in execution['records']]
            if len(times)<10 or times[-1]-times[0]<2.5:
                issues.append({'stage':label,'error':'Incomplete original hold-record coverage'})
    stable=bool(heights) and all(math.isfinite(z) and z>.06 for z in heights) and hold_duration>=2.99
    safe=not forbidden and planned_max<=1e-6 and dense_invalid==0 and not issues
    passed=not missing and len(converged)==len(entries) and len(fingers)==2 and stable and safe
    return {'pass':bool(passed),'score':float(passed),'outcome':'success' if passed else 'failed_execution_or_planning',
        'phase_membership':phases,'executed_stages':executed,'missing_stages':missing,
        'converged_stages':converged,'bilateral_closure':len(fingers)==2,
        'contacting_fingers':sorted(fingers),'hold_min_rise_m':min(heights) if heights else None,
        'recorded_hold_duration_s':hold_duration,'all_recorded_hold_samples_above_6cm':stable,
        'forbidden_contact_events':len(forbidden),'forbidden_contacts':forbidden,
        'planning':{'max_limit_exceed_rad':planned_max,'dense_invalid_samples':dense_invalid},
        'issues':issues,'reported_success':result.get('success'),'reported_error':result.get('error'),
        'scope':'Actual ARX commands and measured contacts/height; original convergence, >6cm/all-hold-samples/3s gates'}
