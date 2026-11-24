"""Five-layout ARX grasp baseline: immutable admission, one attempt, independent ledger."""
import argparse
from collections import Counter
import csv
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import shutil
import time
import traceback
import numpy as np
import sapien
import torch
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation
from environment import make, ROOT, INITIAL_JOINTS, ARM, BASE_POSE, SCENE_CUBOIDS, RECORDING_VIEW
from scene_tools import DB, corners, ArmFK, visibility, export_geometry
from run import trial, snapshot, save, sha, CHECKPOINT, ARX, META

from arx_project.assets import AssetStore, CATALOG as ASSET_CATALOG
from arx_project.scenes import current_dataset, current_schedule
PLAN=current_dataset()
LAYOUTS=['S1_open','S2_edge','S3_sparse','S4_clutter','S5_occluded']
_store=AssetStore()
CATALOG=[{**_store.model(name),'family':row['semantic_class']} for name,row in _store.data['models'].items()]
MODEL_IDS=[o['model_id'] for o in CATALOG]
CODE=Path(__file__).parent
TABLE_MIN=np.array([-.7245,-1.209]);TABLE_MAX=np.array([.4845,1.209])


def actor_spec(name,model,center,yaw):
    r=Rotation.from_euler('z',yaw).as_matrix();pts=corners(DB[model])@r.T
    p=np.r_[np.array(center)-(pts.min(0)[:2]+pts.max(0)[:2])/2, -pts.min(0)[2]+.002]
    return {'name':name,'model_id':model,'pose':[*p.tolist(),np.cos(yaw/2),0.,0.,np.sin(yaw/2)]}


def bounds(spec):
    p=np.array(spec['pose']);r=Rotation.from_quat([*p[4:],p[3]]).as_matrix()
    v=corners(DB[spec['model_id']])@r.T+p[:3]
    return v.min(0),v.max(0)


def separate(candidate,others,gap=.012):
    lo,hi=bounds(candidate)
    if (lo[:2]<TABLE_MIN+.012).any() or (hi[:2]>TABLE_MAX-.012).any():return False
    for other in others:
        a,b=bounds(other)
        if not ((hi[:2]+gap<a[:2]).any() or (b[:2]+gap<lo[:2]).any()):return False
    return True


def sample_scene(entry,variant):
    rng=np.random.default_rng(entry['seed']+variant*1009)
    yaw=float(rng.uniform(-np.pi,np.pi))
    center=np.array([rng.uniform(-.48,-.32),rng.uniform(-.14,.14)])
    if entry['layout']=='S2_edge':
        raw=actor_spec('target',entry['model_id'],[0,0],yaw);lo,hi=bounds(raw)
        center=np.array([TABLE_MIN[0]+.020+(hi[0]-lo[0])/2,rng.uniform(-.28,-.18)])
    target=actor_spec('target',entry['model_id'],center,yaw)
    actors=[target];layout=entry['layout']
    count=int(rng.integers(2,4)) if layout=='S3_sparse' else int(rng.integers(4,7)) if layout=='S4_clutter' else int(rng.integers(2,5)) if layout=='S5_occluded' else 0
    used={entry['model_id']}
    for i in range(count):
        candidates=[m for m in MODEL_IDS if m not in used]
        if layout=='S5_occluded' and i==0:
            candidates=[m for m in ['004_sugar_box','006_mustard_bottle','021_bleach_cleanser','022_windex_bottle','003_cracker_box'] if m not in used]
        model=str(rng.choice(candidates));used.add(model)
        for _ in range(300):
            xy=center+rng.uniform([-.22,-.34],[.36,.34])
            spec=actor_spec(f'distractor_{i:02d}',model,xy,float(rng.uniform(-np.pi,np.pi)))
            if separate(spec,actors,gap=.05 if layout=='S3_sparse' else .012):break
        else:raise RuntimeError('Could not place nonoverlapping distractors')
        actors.append(spec)
    return {'actors':actors,'layout':layout,'variant':variant}


def near_duplicate(a,b):
    first={x['model_id']:x for x in a['actors']};second={x['model_id']:x for x in b['actors']}
    if set(first)!=set(second):return False
    for model,x in first.items():
        y=second[model];p=np.array(x['pose']);q=np.array(y['pose'])
        if np.linalg.norm(p[:2]-q[:2])>=.03:return False
        if model not in ['055_baseball','056_tennis_ball','057_racquetball','058_golf_ball']:
            r1=Rotation.from_quat([*p[4:],p[3]]);r2=Rotation.from_quat([*q[4:],q[3]])
            if (r1.inv()*r2).magnitude()>=np.deg2rad(20):return False
    return True


def queue():
    return deepcopy(current_schedule())


def settle(env):
    events=[];entities={l._objs[0].entity for l in env.agent.robot.get_links()}
    def record():
        for c in env.scene.get_contacts():
            a,b=[v.entity for v in c.bodies]
            if (a in entities)==(b in entities):continue
            impulse=sum(float(np.linalg.norm(v.impulse)) for v in c.points)
            if impulse>1e-8:events.append({'a':a.name,'b':b.name,'impulse_ns':impulse})
    env.trace_callback=record
    try:
        for _ in range(80):env._step_action(torch.tensor(env.initial_q[None],device=env.device))
    finally:env.trace_callback=None
    if events:raise RuntimeError('Robot/environment contact in initial reset: '+str(events[:2]))
    for name,actor in env.objects.items():
        if float(actor.pose.p[0,2])<-.02:raise RuntimeError('Reset object fell: '+name)
        if float(torch.linalg.norm(actor.linear_velocity[0]))>.02 or float(torch.linalg.norm(actor.angular_velocity[0]))>.2:
            raise RuntimeError('Reset object still moving: '+name)


def adjust_occlusion(env,scene,seed):
    rng=np.random.default_rng(seed)
    p=env.obj.pose.to_transformation_matrix().cpu().numpy()[0]
    pts=corners(env.object_info['target'])@p[:3,:3].T+p[:3,3];target=(pts.min(0)+pts.max(0))/2
    obs=env.get_obs();gl=obs['sensor_param']['right_wrist']['cam2world_gl'][0].cpu().numpy();eye=gl[:3,3]
    ray=eye[:2]-target[:2];side=np.array([-ray[1],ray[0]]);side/=max(1e-9,np.linalg.norm(side))
    original=scene['actors'][1]
    resting={n:a.pose.raw_pose[0].cpu().numpy().copy() for n,a in env.objects.items()}
    others=[{'name':n,'model_id':env.object_models[n],'pose':value.tolist()} for n,value in resting.items() if n!=original['name']]
    observation_q=env.agent.robot.qpos.clone()
    for alpha in [.35,.50,.65,.25,.75]:
        for lateral in rng.permutation(np.linspace(-.12,.12,17)):
            dims=np.ptp(corners(DB[original['model_id']]),axis=0)
            yaw=float(np.arctan2(ray[1],ray[0])+(0 if dims[0]<dims[1] else np.pi/2))
            spec=actor_spec(original['name'],original['model_id'],target[:2]+alpha*ray+lateral*side,yaw)
            if not separate(spec,others,gap=.008):continue
            for name,pose in resting.items():
                env.objects[name].set_pose(sapien.Pose(pose[:3],pose[3:]))
                body=env.objects[name]._bodies[0]
                body.linear_velocity=np.zeros(3);body.angular_velocity=np.zeros(3)
            actor=env.objects[spec['name']];q=spec['pose'];actor.set_pose(sapien.Pose(q[:3],q[3:]))
            env.agent.robot.set_qpos(torch.tensor(env.initial_q[None],device=env.device))
            env.agent.robot.set_qvel(torch.zeros_like(env.agent.robot.qpos))
            try:settle(env)
            except RuntimeError:continue
            env.agent.robot.set_qpos(observation_q)
            v=visibility(env)
            if v['admitted_view'] and .22<=v['occlusion_fraction']<=.38:
                for item in scene['actors']:item['pose']=env.objects[item['name']].pose.raw_pose[0].cpu().numpy().tolist()
                return v
    raise RuntimeError('Cannot realize stable 20-40% physical occlusion in this variant')


def admit(entry,out,max_variants=12,start_variant=0):
    out.mkdir(parents=True,exist_ok=False);errors=[];fk=ArmFK()
    for variant in range(start_variant,start_variant+max_variants):
        env=None
        try:
            scene=sample_scene(entry,variant)
            if any(near_duplicate(scene,old) for old in entry.get('avoid_scenes',[])):
                raise RuntimeError('Near-duplicate geometry rejected before registration')
            env=make(entry['model_id'],perception_camera='right_wrist',scene_spec=scene)
            env.reset(seed=entry['seed']);settle(env)
            initial_q=env.agent.robot.qpos[0].cpu().numpy().copy()
            observation=fk.observation(env,entry['seed'])
            names=[j.name for j in env.agent.robot.active_joints];q=initial_q.copy()
            for i,name in enumerate(ARM):q[names.index(name)]=observation['q'][i]
            env.agent.robot.set_qpos(torch.tensor(q[None],device=env.device))
            if entry['layout']=='S5_occluded':
                v=adjust_occlusion(env,scene,entry['seed']+variant)
            else:v=visibility(env)
            if not v['admitted_view']:raise RuntimeError('Target is clipped or insufficiently visible: '+str(v))
            if any(near_duplicate(scene,old) for old in entry.get('avoid_scenes',[])):
                raise RuntimeError('Near-duplicate settled geometry rejected before registration')
            obs=env.get_obs();imageio.imwrite(out/'wrist.png',obs['sensor_data']['right_wrist']['rgb'][0].cpu().numpy())
            imageio.imwrite(out/'overview.png',env.render().cpu().numpy()[0])
            export_geometry(env)
            save(out/'observation_probe_state.json',snapshot(env,obs))
            accepted={**entry,'scene':scene,'observation':observation,'admission':v,'admission_attempts':variant+1}
            save(out/'admission.json',{'passed':True,'scope':'Initialization and render-only observation pose; no grasp screening or motion execution','entry':accepted,'rejected_variants':errors})
            return accepted
        except Exception as exc:
            errors.append({'variant':variant,'error':str(exc),'traceback':traceback.format_exc()})
        finally:
            if env is not None:env.close()
    failed={**entry,'admission_failed':True,'admission_errors':errors}
    save(out/'admission.json',{'passed':False,'entry':failed,'rejected_variants':errors})
    return failed


def isolated_admit(entry,out):
    out.mkdir(parents=True,exist_ok=True);entry=deepcopy(entry);prior=[]
    for receipt in out.parent.glob('*/admission.json'):
        previous=json.loads(receipt.read_text())
        if previous['passed'] and previous['entry']['model_id']==entry['model_id']:
            prior.append(previous['entry']['scene'])
    entry['avoid_scenes']=prior
    if (out/'entry.json').exists() and json.loads((out/'entry.json').read_text())!=entry:
        raise RuntimeError('Admission entry changed across continuation: '+str(out))
    save(out/'entry.json',entry);errors=[]
    for variant in range(12):
        destination=out/f'variant_{variant:02d}'
        cmd=[sys.executable,str(Path(__file__).resolve()),'--output',str(destination),'--admit-entry',str(out/'entry.json'),
             '--variant',str(variant),'--neural-gpu','unused','--planner-gpu','unused']
        receipt=destination/'admission.json'
        if not receipt.exists() and destination.exists():
            errors.append({'variant':variant,'error':'Interrupted admission-only variant preserved; never a grasp attempt'})
            continue
        if not receipt.exists():
            with (out/f'variant_{variant:02d}.log').open('w') as stream:
                subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        if not receipt.exists():raise RuntimeError('Admission infrastructure exited without receipt: '+str(destination))
        r=json.loads(receipt.read_text())
        if not r['passed'] and any(any(word in err['error'].lower() for word in ['out of memory','supported physical device','vk_error','resource descriptor']) for err in r['rejected_variants']):
            raise RuntimeError('GPU/render infrastructure failure; not a rejected scene: '+str(destination))
        if r['passed']:
            for name in ['wrist.png','overview.png','observation_probe_state.json']:shutil.copy2(destination/name,out/name)
            r['prior_variants']=errors;save(out/'admission.json',r);return r['entry']
        errors+=r['rejected_variants']
    failed={**entry,'admission_failed':True,'admission_errors':errors}
    save(out/'admission.json',{'passed':False,'entry':failed,'rejected_variants':errors})
    return failed


def guarded_sources():
    files=list(CODE.glob('*.py'))+[CHECKPOINT,ROOT/'assets/arx/manifest.json',PLAN/'object_catalog.json',PLAN/'manifest.json',ASSET_CATALOG,
            ROOT/'assets/data/assets/mani_skill2_ycb/info_pick_v0.json']
    files += [Path(META['source'])]+[ROOT/'assets/arx'/name for name in META['files']]
    package=ARX/'servers/zerograsp/vendor/zerograsp_mainline_minimal'
    files += list(package.rglob('*.py'))+list((package/'configs').glob('*.yaml'))
    files += [ARX/'servers/zerograsp/coolas_zerograsp/grasp.py',ARX/'shared/arx_shared/schemas.py']
    files += [ARX/p for p in ['arx-sim/config/controllers.yaml','arx-sim/config/runtime.yaml','scripts/curobo_arx/optimize_path.py','scripts/curobo_arx/environment.sh','scripts/curobo_arx/static_pick/reference/historical_reference/initial_setup.json']]
    for kind in ['model','model_closed']:files+=list((ARX/'scripts/curobo_arx/static_pick/reference'/kind).glob('arx.*'))
    for record in CATALOG:files += [Path(p) for p in record['source_hashes']]
    return {str(p):sha(p) for p in set(files)}


def summary(out,schedule):
    results=[]
    for e in schedule:
        p=out/'trials'/e['id']/'result.json'
        if p.exists():results.append(json.loads(p.read_text()))
    groups={k:{'registered':40,'completed':sum(r['layout']==k for r in results),
               'physical_attempts':sum(r['physical_attempt'] for r in results if r['layout']==k),
               'successes':sum(r['success'] for r in results if r['layout']==k)} for k in LAYOUTS}
    family_counts=Counter(e['family'] for e in schedule)
    families={k:{'registered':n,'completed':sum(r['family']==k for r in results),
                 'physical_attempts':sum(r['physical_attempt'] for r in results if r['family']==k),
                 'successes':sum(r['success'] for r in results if r['family']==k)} for k,n in family_counts.items()}
    r={'registered':len(schedule),'completed':len(results),'planning_passes':sum(x['planning_success'] for x in results),
       'physical_attempts':sum(x['physical_attempt'] for x in results),'successes':sum(x['success'] for x in results),
       'by_layout':groups,'by_family':families,'failure_reasons':dict(Counter(x.get('reason','unknown') for x in results if not x['success'])),
       'overall_success_fraction':sum(x['success'] for x in results)/len(schedule),'results':results}
    save(out/'summary.json',r);return r


def verify_trial(path):
    r=json.loads((path/'result.json').read_text());ok=not r['success'];details={}
    if (path/'execution.jsonl').exists():
        origin=path/'execution_origin.json'
        z0=(json.loads(origin.read_text())['object_initial_z_m'] if origin.exists()
            else json.loads((path/'state.json').read_text())['object_world'][2][3])
        count=0;minimum=float('inf');bad=0;fingers=set()
        for line in (path/'execution.jsonl').open():
            row=json.loads(line)
            if row['stage']=='hold':count+=1;minimum=min(minimum,row['object_pose'][2]-z0)
            for c in row['contacts']:
                bad+=not c['allowed']
                if row['stage']=='close' and c['allowed']:
                    fingers.update(n for n in ['right_link27','right_link28'] if n in c['a']+c['b'])
        required={'pregrasp','approach','close','lift','hold'}
        physical_success=bool(minimum>.06 and count/120>=3 and bad==0 and len(fingers)==2
            and required.issubset(r.get('stages',{})) and all(s['settled'] for s in r.get('stages',{}).values()))
        ok=bool(r['success']==physical_success)
        details={'independently_successful':physical_success,'minimum_lift_m':minimum if count else None,'hold_seconds':count/120,'forbidden':bad,'bilateral':sorted(fingers)}
    save(path/'independent_evaluation.json',{'passed':ok,'scope':'Independent executed-trace readback; non-executed and failed cases retain failure','reported_success':r['success'],**details})
    if not ok:raise RuntimeError('Independent physical evaluation disagrees: '+str(path))


def await_resources(args,out):
    required={os.environ['CUDA_VISIBLE_DEVICES']:8500,args.neural_gpu:4500}
    required[args.planner_gpu]=max(required.get(args.planner_gpu,0),4500)
    while True:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        free={line.split(',')[0].strip():int(line.split(',')[1]) for line in raw.strip().splitlines()}
        if all(free.get(g,0)>=n for g,n in required.items()):return
        save(out/'progress.json',{'phase':'waiting_for_resources','required_free_mib':required,'free_mib':free,'total':200})
        if (out/'STOP').exists():raise RuntimeError('User STOP while waiting for GPU resources')
        time.sleep(15)


def campaign(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'STOP').exists():raise RuntimeError('Campaign stopped')
    lockpath=out/'campaign.lock'
    import fcntl
    lock=lockpath.open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    planned=queue();save(out/'selected_plan.json',{'layouts':LAYOUTS,'models':MODEL_IDS,'formal_slots':200,'one_attempt_per_slot':True,'previous_240_plan_preserved':True})
    frozen=out/'registration.json'
    if not frozen.exists() and not args.limit:
        sources=guarded_sources()
        save(frozen,{'schedule':planned,'attempt_budget':200,'layouts':LAYOUTS,'source_hashes':sources,'table_height_m':.50,
             'camera':'right_wrist','recording_camera':RECORDING_VIEW,
             'gpu':{'simulator':os.environ.get('CUDA_VISIBLE_DEVICES'),'neural':args.neural_gpu,'planner':args.planner_gpu},
             'scope':'New ManiSkill five-layout baseline with simulator instance target mask; no cloud policy or training',
             'registration_mode':'Frozen 200 target/layout/seed allocations and sources; deterministic scene admission and geometry receipt before each grasp',
             'success':{'bilateral_target_contact':True,'minimum_lift_m_exclusive':.06,'hold_seconds':3,'joint_tolerance_rad':.005,'forbidden_contacts':0},
             'self_collision_enabled':False,'one_attempt_only':True})
        (out/'source').mkdir()
        for p in CODE.glob('*.py'):shutil.copy2(p,out/'source'/p.name)
        save(out/'storage_policy.json',{'minimum_free_bytes':1073741824,'source':str(ARX/'AGENTS.md'),
             'historical_snapshot_missing':not (ARX/'runs/research_contract_20261019/storage_policy.json').exists()})
        (out/'admission').mkdir()
    if not frozen.exists():
        (out/'admission').mkdir(exist_ok=True);accepted=[]
        for index,e in enumerate(planned):
            p=out/'admission'/e['id']/'admission.json'
            if p.exists():a=json.loads(p.read_text())['entry']
            else:a=isolated_admit(e,p.parent)
            accepted.append(a)
            save(out/'progress.json',{'phase':'admission','checked':index+1,'total':200,'admitted':sum(not x.get('admission_failed',False) for x in accepted),'current':e['id'],'grasp_attempts':0})
            print('ADMISSION',index+1,e['id'],not a.get('admission_failed',False),flush=True)
            if args.limit and index+1>=args.limit:return
        sources=guarded_sources()
        save(frozen,{'schedule':accepted,'attempt_budget':200,'layouts':LAYOUTS,'source_hashes':sources,'table_height_m':.50,
                     'camera':'right_wrist','recording_camera':RECORDING_VIEW,'gpu':{'simulator':os.environ.get('CUDA_VISIBLE_DEVICES'),'neural':args.neural_gpu,'planner':args.planner_gpu},
                     'scope':'New ManiSkill five-layout grasp baseline with oracle target mask; no cloud policy or training','admitted':sum(not a.get('admission_failed',False) for a in accepted),
                     'success':{'bilateral_target_contact':True,'minimum_lift_m_exclusive':.06,'hold_seconds':3,'joint_tolerance_rad':.005,'forbidden_contacts':0},
                     'self_collision_enabled':False,'one_attempt_only':True})
        (out/'source').mkdir()
        for p in CODE.glob('*.py'):shutil.copy2(p,out/'source'/p.name)
        save(out/'storage_policy.json',{'minimum_free_bytes':1073741824,'source':str(ARX/'AGENTS.md'),'historical_snapshot_missing':not (ARX/'runs/research_contract_20261019/storage_policy.json').exists()})
    reg=json.loads(frozen.read_text());schedule=reg['schedule'];(out/'trials').mkdir(exist_ok=True)
    summary(out,schedule)
    for e in schedule:
        p=out/'trials'/e['id']
        if (p/'result.json').exists():continue
        if p.exists():raise RuntimeError('Interrupted/unknown slot must not be replayed: '+str(p))
        if (out/'STOP').exists():return
        for name,digest in reg['source_hashes'].items():
            if sha(Path(name))!=digest:raise RuntimeError('Frozen source changed: '+name)
        if shutil.disk_usage(out).free<=1073741824:raise RuntimeError('Storage reserve reached')
        await_resources(args,out)
        admission=out/'admission'/e['id']/'admission.json'
        if 'scene' not in e and not e.get('admission_failed'):
            save(out/'progress.json',{'phase':'admission','current':e['id'],'completed':sum((out/'trials'/x['id']/'result.json').exists() for x in schedule),'total':200})
            if admission.exists():e=json.loads(admission.read_text())['entry']
            else:e=isolated_admit(e,admission.parent)
            for name,digest in reg['source_hashes'].items():
                if sha(Path(name))!=digest:raise RuntimeError('Source changed during admission: '+name)
        save(out/'progress.json',{'phase':'running','current':e['id'],'layout':e['layout'],'completed':sum((out/'trials'/x['id']/'result.json').exists() for x in schedule),'total':200})
        if e.get('admission_failed'):
            p.mkdir();save(p/'result.json',{**e,'planning_success':False,'physical_attempt':False,'success':False,'reason':'scene_admission_failed'})
        else:
            e['campaign_registration']=str(frozen)
            entries=out/'entries';entries.mkdir(exist_ok=True);save(entries/(e['id']+'.json'),e)
            cmd=[sys.executable,str(Path(__file__).resolve()),'--output',str(p),'--execute-entry',str(entries/(e['id']+'.json')),
                 '--neural-gpu',args.neural_gpu,'--planner-gpu',args.planner_gpu]
            with (entries/(e['id']+'.log')).open('w') as stream:
                subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=3600)
            if not (p/'result.json').exists():raise RuntimeError('Unknown execution outcome; do not replay '+str(p))
        verify_trial(p);r=summary(out,schedule)
        print('RESULT',e['id'],r['completed'],r['physical_attempts'],r['successes'],flush=True)
    r=summary(out,schedule);save(out/'progress.json',{'phase':'complete','completed':r['completed'],'total':200,'successes':r['successes']})
    (out/'STOP').write_text('Registered 200-slot baseline completed. No automatic retries.\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--neural-gpu',required=True);parser.add_argument('--planner-gpu',required=True)
    parser.add_argument('--limit',type=int,help='Admission-only bounded smoke check; never launches grasps')
    parser.add_argument('--admit-entry',type=Path)
    parser.add_argument('--execute-entry',type=Path)
    parser.add_argument('--variant',type=int,default=0)
    args=parser.parse_args()
    try:
        if args.admit_entry:
            admit(json.loads(args.admit_entry.read_text()),args.output,max_variants=1,start_variant=args.variant)
        elif args.execute_entry:
            trial(json.loads(args.execute_entry.read_text()),args.output,args.neural_gpu,args.planner_gpu,'right_wrist')
        else:campaign(args)
    except Exception:
        args.output.mkdir(parents=True,exist_ok=True)
        save(args.output/'halt.json',{'error':traceback.format_exc(),'time':time.time(),'pid':os.getpid()})
        raise


if __name__=='__main__':main()
