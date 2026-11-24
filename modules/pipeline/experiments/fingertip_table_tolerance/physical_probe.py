"""Three registered physical replays; no neural reruns or automatic retries."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from audit import HERE, PROJECT, WORKSPACE, SOURCE, now, read, save, sha
from plan_probe import ROOT as PLAN
from geometry import FINGERS as FINGER_LINKS

ROOT = HERE/'runs/physical3_20261123_02'
V1 = PROJECT/'experiments/repaired_grasp_benchmark/runs/geometry200_repaired_20261123_01'
SIM_GPU = 'GPU-3164d5f3-6a9a-17b4-53f9-a70e3a82b834'
CHECK_GPU = 'GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b'
SCENES = ['D0008_S1_open_teapot_d0','D0036_S1_open_cell_phone_d0','D0125_S4_clutter_bar_soap_d0']


def prepare():
    assert read(PLAN/'completion_verification.json')['passed']
    assert not ROOT.exists()
    items=[]
    inputs={}
    for source in SCENES:
        plan=PLAN/'cases'/(source+'_mesh2mm')/'result.json'
        assert read(plan)['initial_planning_success']
        case=SOURCE/'cases'/source
        items.append({'id':source,'source_case':str(case),'plan_file':str(plan),'tolerance_m':.002})
        for path in [plan,case/'state.json',case/'entry.json']:
            inputs[str(path)]=sha(path)
    for path in [V1/'object_catalog.json',V1/'common_initial_conditions.json',V1/'registration.json']:
        inputs[str(path)]=sha(path)
    save(ROOT/'registration.json',{'at':now(),'version':'V2','schedule':items,'physical_attempt_budget':3,
        'automatic_retries':0,'user_authorization':'2026-11-23: begin implementation of V2 including separately registered physical verification after offline checks',
        'scope':'Replay the same archived grasp and scene; isolate local table-contact policy, no new perception or candidate generation',
        'simulation_gpu':SIM_GPU,'closed_path_checker_gpu':CHECK_GPU,'physics_contact_offset_m':.001,
        'success':{'minimum_lift_m':.01,'hold_s':3.,'bilateral_fingers':2,'joint_error_rad':.005,
                   'contact_results':'Report original V1 and scoped V2 independently'},
        'code_sha256':{str(p):sha(p) for p in [Path(__file__),HERE/'closed_lift.py',HERE/'runtime_adapter.py',HERE/'narrow_phase.py']},
        'input_sha256':inputs})
    for name in ['object_catalog.json','common_initial_conditions.json']:
        shutil.copy2(V1/name,ROOT/name)
    save(ROOT/'progress.json',{'phase':'registered','registered':3,'completed':0})


def install():
    sys.path[:0]=[str(PROJECT/'experiments/repaired_grasp_benchmark'),str(PROJECT/'experiments/rayst3r_graspgenx'),
                 str(WORKSPACE/'maniskill'),str(WORKSPACE/'maniskill/arx_pipeline')]
    spec=importlib.util.spec_from_file_location('v2_repaired_physics',PROJECT/'experiments/repaired_grasp_benchmark/simulation.py')
    sim=importlib.util.module_from_spec(spec);spec.loader.exec_module(sim)
    return sim.install(ROOT)[0]


def run():
    reg=read(ROOT/'registration.json')
    for p,h in {**reg['code_sha256'],**reg['input_sha256']}.items(): assert sha(p)==h
    assert not (ROOT/'started.json').exists() and not (ROOT/'STOP').exists()
    save(ROOT/'started.json',{'at':now(),'pid':os.getpid()})
    run=install(); import torch
    parent=run.Execution

    class ContactExecution(parent):
        def __init__(self,env,out):
            super().__init__(env,out)
            self.original_forbidden=[];self.tip_table=[];self.release_depth=None
        def record(self):
            self.tick+=1;contacts=[]
            robot={link._objs[0].entity for link in self.env.agent.robot.get_links()};target=self.env.obj._objs[0]
            eef=self.env.agent.tcp.pose.to_transformation_matrix().cpu().numpy()[0]
            for c in self.env.scene.get_contacts():
                a,b=[body.entity for body in c.bodies]
                if (a not in robot and b not in robot) or (a in robot and b in robot):continue
                impulse=sum(float(np.linalg.norm(p.impulse)) for p in c.points)
                if not c.points or impulse<=1e-8:continue
                link,other=(a,b) if a in robot else (b,a)
                finger=any(n in link.name for n in FINGER_LINKS)
                original=self.label!='observe' and other==target and finger
                points=[{'world':np.asarray(p.position).tolist(),'normal':np.asarray(p.normal).tolist(),
                         'separation_m':float(p.separation),'impulse_ns':float(np.linalg.norm(p.impulse))} for p in c.points]
                allowed=original;table_contact=False
                if finger and 'table-workspace' in other.name and self.label in ['approach','close','lift']:
                    active=[p for p in points if p['impulse_ns']>1e-8]
                    local=[(np.asarray(p['world'])-eef[:3,3])@eef[:3,:3] for p in active]
                    maximum=max([max(0.,-p['separation_m']) for p in active],default=0.)
                    top=all(abs(p['normal'][2])>.95 and abs(p['world'][2])<=.0031 for p in active)
                    distal=all(v[2]>=.034970-1e-6 for v in local)
                    allowed=bool(active) and top and distal and maximum<=.002
                    table_contact=allowed
                event={'a':a.name,'b':b.name,'impulse_ns':impulse,'allowed':allowed,'original_allowed':original,
                       'tip_table_contact':table_contact,'points':points}
                contacts.append(event)
            table_events=[e for e in contacts if e['tip_table_contact']]
            frame_depth=max([max(0.,-p['separation_m']) for e in table_events for p in e['points'] if p['impulse_ns']>1e-8],default=0.)
            if self.label=='close':self.release_depth=frame_depth
            if self.label=='lift':
                ceiling=0. if self.release_depth is None else self.release_depth
                if frame_depth>ceiling+.0001:
                    for e in table_events:e['allowed']=False
                self.release_depth=min(ceiling,frame_depth)
            for event in contacts:
                if not event['original_allowed']:self.original_forbidden.append({'tick':self.tick,'stage':self.label,**event})
                if not event['allowed']:self.forbidden.append({'tick':self.tick,'stage':self.label,**event})
                if event['tip_table_contact'] and event['allowed']:self.tip_table.append({'tick':self.tick,'stage':self.label,**event})
                if event['original_allowed'] and self.label=='close':
                    self.bilateral.add('right_link27' if 'right_link27' in event['a']+event['b'] else 'right_link28')
            z=float(self.env.obj.pose.p[0,2])
            if self.label=='hold':self.heights.append(z)
            row={'tick':self.tick,'sim_seconds':self.tick/120,'stage':self.label,
                 'qpos':self.env.agent.robot.qpos[0].tolist(),'object_pose':self.env.obj.pose.raw_pose[0].tolist(),
                 'eef_world':eef.tolist(),'contacts':contacts,
                 'objects':{n:a.pose.raw_pose[0].tolist() for n,a in self.env.objects.items()}}
            self.trace.write(json.dumps(row)+'\n')
            body=self.env.obj._bodies[0];speed=float(np.linalg.norm(body.linear_velocity))
            pos=self.env.obj.pose.p[0].cpu().numpy();self.physics_max_speed=max(self.physics_max_speed,speed)
            self.physics_max_displacement=max(self.physics_max_displacement,float(np.linalg.norm(pos-self.physics_initial_position)))
            self.physics_stream.write(json.dumps({'tick':self.tick,'stage':self.label,'speed_mps':speed,'position':pos.tolist()})+'\n')
        def finish(self):
            save(self.out/'original_v1_forbidden_contacts.json',self.original_forbidden)
            save(self.out/'allowed_tip_table_contacts.json',self.tip_table)
            super().finish()
    run.Execution=ContactExecution
    results=[]
    for item in reg['schedule']:
        assert not (ROOT/'STOP').exists()
        out=ROOT/'cases'/item['id'];out.mkdir(parents=True,exist_ok=False)
        entry=read(Path(item['source_case'])/'entry.json');expected=read(Path(item['source_case'])/'state.json')
        plan=read(item['plan_file']);selected=plan['selected']
        env=None;execution=None;result={'id':item['id'],'physical_attempt':False,'success_v2':False,'success_v1':False}
        try:
            env=run.make(entry['model_id'],perception_camera='head',scene_spec=entry['scene']);env.reset(seed=entry['seed'])
            for _ in range(80):env._step_action(torch.tensor(env.initial_q[None],device=env.device))
            actual=run.snapshot(env,env.get_obs());save(out/'state.json',actual)
            arm=[actual['joint_names'].index(n) for n in [f'right_joint{i}' for i in range(21,27)]]
            q=np.array(actual['qpos'])[arm];q0=np.array(plan['paths']['pregrasp']['q'][0])
            delta=float(np.linalg.norm(np.asarray(actual['object_world'])[:3,3]-np.asarray(expected['object_world'])[:3,3]))
            save(out/'initial_replay_check.json',{'robot_start_max_error_rad':float(np.max(np.abs(q-q0))),'object_position_error_m':delta})
            if np.max(np.abs(q-q0))>=.005 or delta>.002:raise ValueError('Replay initial conditions differ from registered plan')
            obs=env.get_obs();run.Image.fromarray(obs['sensor_data']['head']['rgb'][0].cpu().numpy()).save(out/'initial_head.png')
            run.Image.fromarray(env.render().cpu().numpy()[0]).save(out/'initial_overview.png')
            execution=run.Execution(env,out);result['physical_attempt']=True
            for label in ['pregrasp','approach']:
                path=plan['paths'][label];execution.segment(label,path['q'],path['times'],goal_world=selected['pregrasp_world'] if label=='pregrasp' else selected['grasp_world'])
            q=selected['q_grasp'];execution.segment('close',[q,q],[0,3],finger=0.,goal_world=selected['grasp_world'])
            if len(execution.bilateral)!=2:raise RuntimeError('no_bilateral_target_contact')
            save(out/'closed_state.json',run.snapshot(env))
            save(out/'lift_request.json',{'q_lift':selected['q_lift'],'tolerance_m':.002})
            worker_env=dict(os.environ);worker_env.update(CUDA_VISIBLE_DEVICES=CHECK_GPU,ARX_SIM_GPU=CHECK_GPU,ARX_PLANNER_GPU=CHECK_GPU,ARX_NEURAL_GPU=CHECK_GPU,LILO_GPU=CHECK_GPU)
            cmd=['bash','-c','source "$1"; shift; exec "$@"','v2-closed-lift',str(WORKSPACE/'arx_core/scripts/curobo_arx/environment.sh'),str(WORKSPACE/'curobo/.venv/bin/python'),str(HERE/'closed_lift.py'),str(out)]
            with (out/'lift_checker.log').open('x') as stream:
                subprocess.run(cmd,env=worker_env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=300)
            lift=read(out/'lift_plan_result.json')
            if not lift['success']:raise RuntimeError(lift['reason'])
            path=lift['paths']['lift'];execution.segment('lift',path['q'],path['times'],finger=0.,goal_world=selected['lift_world'])
            q=path['q'][-1];execution.segment('hold',[q,q],[0,3],finger=0.,goal_world=selected['lift_world'])
            height=min(execution.heights)-execution.initial_z;hold=len(execution.heights)/120
            result.update(minimum_hold_lift_m=height,hold_seconds=hold)
            result['success_v2']=bool(height>=.01 and hold>=3 and not execution.forbidden and len(execution.bilateral)==2)
            result['success_v1']=bool(result['success_v2'] and not execution.original_forbidden)
            result['reason']='success' if result['success_v2'] else 'insufficient_hold_height'
        except RuntimeError as error:
            known={'forbidden_robot_contact','no_bilateral_target_contact','measured_closed_lift_path_collision',
                   'pregrasp_not_converged','approach_not_converged','close_not_converged','lift_not_converged',
                   'hold_not_converged','pregrasp_trajectory_start_mismatch','approach_trajectory_start_mismatch',
                   'close_trajectory_start_mismatch','lift_trajectory_start_mismatch','hold_trajectory_start_mismatch'}
            if str(error) not in known:raise
            result['reason']=str(error)
        finally:
            if execution:
                execution.finish();result.update(stages=execution.stages,bilateral_fingers=sorted(execution.bilateral),
                    allowed_tip_table_contact_events=len(execution.tip_table),v2_forbidden_events=len(execution.forbidden),v1_forbidden_events=len(execution.original_forbidden))
            if env:
                save(out/'final_state.json',run.snapshot(env));env.close()
            save(out/'result.json',result)
        results.append(result)
        save(ROOT/'progress.json',{'phase':'running','registered':3,'completed':len(results),'last':result})
        print('PHYSICAL',item['id'],result,flush=True)
    for p,h in {**reg['code_sha256'],**reg['input_sha256']}.items(): assert sha(p)==h
    save(ROOT/'summary.json',{'registered':3,'completed':3,'physical_attempts':sum(r['physical_attempt'] for r in results),
        'success_v2':sum(r['success_v2'] for r in results),'success_v1':sum(r['success_v1'] for r in results),'cases':results})
    save(ROOT/'completion_verification.json',{'passed':True,'completed':3,'inputs_unchanged':True})
    save(ROOT/'progress.json',{'phase':'complete','completed':3,'registered':3})


if __name__=='__main__':
    try:prepare();run()
    except BaseException:
        import traceback
        if ROOT.exists():(ROOT/'STOP').write_text(traceback.format_exc())
        raise
