"""Summarize paired physical trials; never count an unexecuted plan as a grasp."""
from pathlib import Path
import json,hashlib
import numpy as np
root=Path(__file__).resolve().parents[2];out=root/'runs/weight_comparison_20261015'
rows=json.loads((out/'comparison.json').read_text());groups={};pairs=[]
for w in (1,100):
    rr=[r for r in rows if r['weight_multiplier']==w]
    stages=[o for r in rr for o in r['optimizations'].values()]
    groups[str(w)]={'trials':len(rr),'started_execution':sum(r['started_execution'] for r in rr),'lift_executed':sum(r['lift_executed'] for r in rr),'physical_successes':sum(r['success'] for r in rr),'physical_success_rate':sum(r['success'] for r in rr)/len(rr) if rr else None,'optimization_stages_attempted':len(stages),'optimization_stages_passed':sum(o['success'] for o in stages)}
    groups[str(w)].update(optimization_wall_seconds_total=sum(o['planning_wall_seconds'] for o in stages),optimization_wall_seconds_mean=float(np.mean([o['planning_wall_seconds'] for o in stages])) if stages else None,stages_requiring_retry=sum(len(o['attempts'])>1 for o in stages),max_planned_position_limit_exceed_rad=max([o.get('max_limit_exceed_rad',0) for o in stages],default=0))
for name in dict.fromkeys(r['name'] for r in rows):
    rr=[r for r in rows if r['name']==name]
    if len(rr)!=2:continue
    rr.sort(key=lambda r:r['weight_multiplier']);folders=[Path(r['folder']) for r in rr]
    states=[json.loads((f/'fixed_trial_state.json').read_text()) for f in folders]
    hashes=[hashlib.sha256((f/'selected_grasp.json').read_bytes()).hexdigest() for f in folders]
    assert hashes[0]==hashes[1],name
    pose=[np.array(s['target_world']) for s in states]
    seeds=[r['optimizations'].get('pregrasp_optimization',{}).get('prm_seed_sha256') for r in rr]
    pair={'name':name,'same_selected_grasp':True,'selected_grasp_sha256':hashes[0],'same_pregrasp_prm_seed':seeds[0]==seeds[1],'initial_arm_max_difference_rad':float(abs(np.array(states[0]['initial_arm'])-states[1]['initial_arm']).max()),'initial_object_position_difference_m':float(np.linalg.norm(pose[0][:3,3]-pose[1][:3,3])),'baseline_success':rr[0]['success'],'high_weight_success':rr[1]['success'],'baseline_error':rr[0]['error'],'high_weight_error':rr[1]['error']}
    a,b=[json.loads((f/'trajopt_effective_config.json').read_text()) for f in folders]
    a['rollout']['cost_cfg']['cspace_cfg']['weight'][0]*=100
    assert a==b, f'Unexpected optimizer config difference: {name}'
    pair['only_optimizer_config_difference_is_weight']=True
    paths=[np.array(json.loads((f/'pregrasp_prm_seed.json').read_text())) for f in folders]
    pair['pregrasp_prm_seed_max_difference_rad']=float(abs(paths[0]-paths[1]).max()) if paths[0].shape==paths[1].shape else None
    pairs.append(pair)
for row in rows:
    folder=Path(row['folder']);unwanted=0;violation=0.
    for f in folder.glob('*_execution.json'):
        execution=json.loads(f.read_text());snap=execution['final'];names=snap['right_arm']
        limits=np.array([snap['joint_limits'][snap['joint_names'].index(n)] for n in names])
        q=np.array([record['position'] for record in execution['records']])
        violation=max(violation,float(np.maximum(limits[:,0]-q,q-limits[:,1]).clip(0).max()))
        target=snap['objects']['test_object']['path']
        for c in execution['contacts']:
            if c['num_contacts']<=0 or ('/World/Robot/' in c['a'] and '/World/Robot/' in c['b']):continue
            if not (target in c['a']+c['b'] and any(n in c['a']+c['b'] for n in ('right_link27','right_link28'))):unwanted+=1
    row['actual_max_position_limit_exceed_rad']=violation;row['unwanted_contact_events']=unwanted
    result=json.loads((folder/'result.json').read_text());row['lift_m']=result.get('lift_m')
    row['video_error']=result.get('video_error')
report={'method':{'robot':'ARX in Isaac Sim','weight_original':10000,'weight_modified':1000000,'optimizer':'PRM seed + L-BFGS B-spline','fallback_to_prm_only':False,'initial_iters':[100,500],'finetune_attempts':3,'frozen_grasp_goals':True,'random_seed':0,'self_collision_enabled':False,'success_definition':'Object stays at least 6 cm above initial height throughout final hold, all execution stages converge, bilateral finger contact before lift, no unwanted contact','trials_per_object_per_group':1,'order':'Alternating group order between objects','caveat':'End-to-end paired trials: measured states and later PRM seeds can differ due physical execution; not a statistical estimate over repeated trials.'},'groups':groups,'pairs':pairs,'trials':rows}
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(groups,indent=2));print(json.dumps(pairs,indent=2))
