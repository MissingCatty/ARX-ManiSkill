"""Run fresh ZeroGrasp -> frozen ARX cuRobo -> actual bottle execution/video.

Requires reproduce_setup.py to have restored and verified the explicit historical
bundle. Outputs never overwrite the original successful bottle_clear_00 record.
"""
import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
import time
from pathlib import Path

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts/curobo_arx'),str(ROOT/'shared'),str(ROOT/'servers/zerograsp')]
from client import Client
from research_guard import invariant_digest
from arx_shared.schemas import ZeroGraspInputRequest
from coolas_zerograsp.grasp import prepare_zerograsp_input
from gpu_policy import resolve,compute_env


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--curobo-python',type=Path,required=True)
    parser.add_argument('--neural-gpu',required=True)
    parser.add_argument('--planner-gpu',required=True)
    parser.add_argument('--trajopt-policy',choices=('historical','first_valid','interactive'),default=os.environ.get('GRASP_TRAJOPT_POLICY','historical'))
    parser.add_argument('--mailbox',type=Path,help='Reuse an already running simulator transport')
    parser.add_argument('--audit',action=argparse.BooleanOptionalAction,default=False,
                        help='Enable full configuration audits for research comparison; default off')
    parser.add_argument('--skip-encoding',action='store_true',help='Keep actual frames for a campaign-specific encoder')
    parser.add_argument('--fixed-selection',type=Path,help='Explicit qualified planned target; still retain fresh RGBD and network inference')
    args=parser.parse_args()
    args.neural_gpu=resolve(args.neural_gpu)['uuid'];args.planner_gpu=resolve(args.planner_gpu)['uuid']
    out=args.output.resolve();source=args.reference_run.resolve();transport=args.mailbox.resolve() if args.mailbox else out/'mailbox'
    os.environ['ARX_CUROBO_VALIDATION_DIR']=str(transport);client=Client(transport)
    def read(path):return json.loads(path.read_text())
    def save(name,value):
        temp=out/(name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(out/name)
    def stage(label,python,script,argv,env):
        print(label,flush=True)
        started=time.monotonic()
        save('pipeline_stage.json',{'stage':label,'status':'running','started_monotonic':started})
        try:
            with (out/(label+'.log')).open('x') as log:
                subprocess.run([str(python),str(script),*map(str,argv)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        finally:
            with (out/'pipeline_timing.jsonl').open('a') as stream:
                stream.write(json.dumps({'stage':label,'seconds':time.monotonic()-started})+'\n')
    assert read(out/'setup_verification.json')['pass']
    assert shutil.disk_usage(out).free>1073741824
    environment=os.environ.copy();environment.update(ARX_CUROBO_VALIDATION_DIR=str(client.root),GRASP_OUTPUT_DIR=str(out),GRASP_TARGET='test_object',ARX_OPS_URL=os.environ.get('ARX_OPS_URL','http://127.0.0.1:8101'),GRASP_PREPARED_CLOSED_MODEL='1',GRASP_PLANNER='prm_trajopt',GRASP_BOUND_WEIGHT_MULTIPLIER='100',ARX_RESEARCH_GUARD_FILE=str(out/'guard.json'),GRASP_VIDEO_DIR=str(out/'frames'),PYTHONUNBUFFERED='1')
    environment['GRASP_TRAJOPT_POLICY']=args.trajopt_policy
    environment['ARX_RESEARCH_AUDIT']='1' if args.audit else '0'
    for key in ('GRASP_FIXED_SELECTION','GRASP_CAMERA_VIDEO_DIR'):environment.pop(key,None)
    if args.fixed_selection:
        selected=args.fixed_selection.resolve();assert selected.is_file()
        environment['GRASP_FIXED_SELECTION']=str(selected)
        save('goal_selection_source.json',{'kind':'explicit_previous_planned_target','source':str(selected),
            'sha256':hashlib.sha256(selected.read_bytes()).hexdigest(),'fresh_network_inference_retained':True})
    neural=compute_env(environment,args.neural_gpu)
    planning=compute_env(environment,args.planner_gpu)
    simulation_gpu=read(out/'setup.json')['fingerprint']['invariant']['configured']['simulation']['active_gpu']
    save('compute_assignment.json',{'neural_gpu':args.neural_gpu,'planner_gpu':args.planner_gpu,'simulation_gpu_index':simulation_gpu,'neural_python':sys.executable,'curobo_python':str(args.curobo_python)})
    assignment=read(out/'compute_assignment.json')
    assignment['simulation_gpu_index_scope']='Simulator-local visible index; physical placement is established by PID/UUID, not this number'
    save('compute_assignment.json',assignment)
    save('audit_mode.json',{'enabled':args.audit,'per_trajectory_configuration_scans':args.audit,
                          'post_run_configuration_scan':args.audit,
                          'planning_collision_limits_and_execution_contact_checks':True})
    dependency=ROOT.parent/'curobo/source'
    dependency_patch=subprocess.check_output(['git','-C',str(dependency),'diff'],text=True)
    (out/'environment_evidence/curobo_source.patch').write_text(dependency_patch)
    kernel=dependency/'curobo/_src/geom/collision/wp_sweep_collision_kernel.py'
    save('planning_implementation.json',{'policy':environment['GRASP_TRAJOPT_POLICY'],'curobo_commit':subprocess.check_output(['git','-C',str(dependency),'rev-parse','HEAD'],text=True).strip(),'sweep_kernel_sha256':hashlib.sha256(kernel.read_bytes()).hexdigest(),'dependency_patch_sha256':hashlib.sha256(dependency_patch.encode()).hexdigest(),'acceptance_gates':'Unchanged solver success, joint/endpoint limits, dense collision checking and actual execution/contact/hold evaluator; full historical optimizer attempts retained as fallback','promotion':'first_valid is an explicit benchmark option; historical remains the global default pending broader success/failure campaign validation'})
    implementation=json.loads((out/'planning_implementation.json').read_text())
    implementation.update(swept_mesh_bounds=environment.get('CUROBO_SWEPT_MESH_BOUNDS','0')=='1',mesh_data_sha256=hashlib.sha256((dependency/'curobo/_src/geom/data/data_mesh.py').read_bytes()).hexdigest(),audit_transport_timeout_seconds=180,promotion='first_valid and interactive are explicit benchmark options; historical remains the global default pending broader validation')
    save('planning_implementation.json',implementation)
    for name in ('grasp_input.py','prepare.py','grasp_execute.py','optimize_path.py','gripper_update.py','settling.py','client.py','model_archive.py','research_guard.py'):
        shutil.copy2(ROOT/'scripts/curobo_arx'/name,out/'environment_evidence'/name)
    shutil.copy2(Path(__file__),out/'environment_evidence'/Path(__file__).name)
    exit_code=1;error=None
    try:
        stage('fresh_rgbd',sys.executable,ROOT/'scripts/curobo_arx/grasp_input.py',[],neural)
        prepared=prepare_zerograsp_input(ZeroGraspInputRequest(camera_dir=str(out/'camera'),mask_path=str(out/'camera/mask.png'),output_dir=str(out/'zg_input'),label=environment.get('ARX_OBJECT_LABEL','bbq_sauce_bottle')))
        save('network_input.json',prepared.model_dump())
        package=ROOT/'servers/zerograsp/vendor/zerograsp_mainline_minimal'
        checkpoint=Path(environment.get('ARX_ZEROGRASP_CHECKPOINT',package/'checkpoints/zerograsp_cvpr2025/zerograsp_synthetic.ckpt'))
        save('checkpoint.json',{'path':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'historical_checkpoint_sha256':'unverified; not present in archived evidence'})
        stage('fresh_zerograsp',sys.executable,package/'run_mainline.py',['--img_path',prepared.rgb_path,'--depth_path',prepared.depth_path,'--mask_path',prepared.mask_path,'--camera_info_path',prepared.camera_info_path,'--output_dir',out/'network','--checkpoint',checkpoint,'--config',package/'configs/demo_640x480.yaml'],neural)
        paths=sorted((out/'network/raw_outputs').glob('*.raw_grasps.npy'))
        assert paths,'Raw network proposals were not retained'
        rows=np.concatenate([np.load(path) for path in paths]);rows=rows[np.argsort(-rows[:,0])]
        (out/'zerograsp').mkdir();np.save(out/'zerograsp/object_1.grasp.npy',rows)
        save('network_candidates.json',{'fresh_inference':True,'count':len(rows),'source_files':[str(path) for path in paths],'mask_method':'Unchanged historical test_object RGB-D/ground-truth projection mask','selection':'Raw proposals, then ARX cuRobo IK, collision checks, PRM and trajectory optimization'})
        stage('prepare',args.curobo_python,ROOT/'scripts/curobo_arx/prepare.py',['--output',out,'--model-source',source],planning)
        # Preserve historical geometry, using the original pre-motion nominal
        # finger joint state before measuring the new physical closure.
        path=out/'model_closed/arx.yml';config=yaml.safe_load(path.read_text());original=copy.deepcopy(config)
        for name in ('right_joint27','right_joint28'):config['robot_cfg']['kinematics']['lock_joints'][name]=.022
        check=copy.deepcopy(config);check['robot_cfg']['kinematics']['lock_joints']=original['robot_cfg']['kinematics']['lock_joints'];assert check==original
        path.write_text(yaml.safe_dump(config,sort_keys=False))
        save('closed_model_nominal_state.json',{'right_joint27':.022,'right_joint28':.022,'source':'Original bottle_clear_00 grasp_execute.py nominal closed prewarm state','geometry_changed':False})
        guard=read(out/'guard.json')
        for directory in ('model','model_closed'):
            for name in ('arx.yml','arx.urdf'):
                path=source/directory/name;guard['file_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        save('guard.json',guard)
        stage('grasp_execution',args.curobo_python,ROOT/'scripts/curobo_arx/grasp_execute.py',[],planning)
        exit_code=0
    except Exception:
        error=traceback.format_exc();save('client_error.json',{'error':error});print(error,flush=True)
    finally:
        managed=environment.get('ARX_BATCH_MANAGED_CLEANUP')=='1'
        if environment.get('ARX_LILO_TEACHER_DIR') and not managed:
            try:
                save('teacher_recording.json',client.request('grasp_lilo_record_stop'))
            except Exception as recording_error:
                save('teacher_recording.json',{'error':str(recording_error)})
        if (out/'frames').exists() and not managed:save('recording_stop.json',client.request('grasp_video_stop',timeout=120))
        final=None;changed=None;unchanged=None
        if args.audit:
            final=client.request('grasp_research_fingerprint',timeout=180);save('fingerprint_after.json',final)
            guard=read(out/'guard.json');changed=[p for p,h in guard['file_sha256'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
            unchanged=invariant_digest(final)==guard['invariant_sha256']
        if not managed:save('final_snapshot.json',client.request('snapshot'))
        commands=[];contacts=[];executions=[]
        labels=['pregrasp',*[p.name.removesuffix('_command.json') for p in sorted(out.glob('local_*_command.json'))],'hold'] if environment.get('ARX_LILO_POLICY_URL') else ['pregrasp','approach','close','lift','hold']
        record_errors=[]
        if environment.get('ARX_LILO_POLICY_URL') and environment.get('ARX_LILO_VERIFY_CLOSURE')=='1':
            labels.insert(-1,'close')
        if (out/'pregrasp_via_command.json').exists() and 'pregrasp_via' not in labels:labels.insert(0,'pregrasp_via')
        for name in labels:
            if (out/(name+'_command.json')).exists():commands.append(read(out/(name+'_command.json')))
            if (out/(name+'_execution.json')).exists():
                execution=read(out/(name+'_execution.json'));executions.append(execution);contacts+=execution['contacts']
        save('all_execution_contacts.json',contacts)
        snap=read(out/'capture_snapshot.json') if (out/'capture_snapshot.json').exists() else {}
        target=snap.get('objects',{}).get('test_object',{}).get('path','/World/Task/assets/asset_bbq_sauce_bottle')
        bad=[];fingers=set()
        for event in contacts:
            if event['num_contacts']<=0:continue
            actors=event['a']+event['b']
            if '/World/Robot/' in event['a'] and '/World/Robot/' in event['b']:continue
            if target in actors and any(name in actors for name in ('right_link27','right_link28')):
                fingers.update(name for name in ('right_link27','right_link28') if name in actors);continue
            bad.append(event)
        violation=0.
        for command in commands:
            bounds=np.array([snap['joint_limits'][snap['joint_names'].index(name)] for name in command['joint_names']]);q=np.array(command['position']);violation=max(violation,float(np.maximum(bounds[:,0]-q,q-bounds[:,1]).clip(0).max()))
        result=read(out/'grasp_report.json') if (out/'grasp_report.json').exists() else {}
        clear=not any(p.startswith(('/World/StaticPickPole','/World/StaticPickRack','/World/DynamicPickObstacle')) for p in final['invariant']['usd']) if final else None
        review={'physical_grasp_success':result.get('success',False),'historical_physics_matched_before_execution':read(out/'setup_verification.json')['historical_physical_properties_equal'],'configuration_audit_enabled':args.audit,'current_configuration_unchanged':unchanged,'changed_protected_sources':changed,'no_added_obstacles':clear,'forbidden_contact_events':len(bad),'contacting_fingers':sorted(fingers),'command_max_joint_limit_violation_rad':violation,'fresh_zerograsp_inference':(out/'network_candidates.json').exists(),'command_trajectory_count':len(commands),'executed_trajectory_count':len(executions),'held_min_rise_m':result.get('held_object_min_z_m',0)-result.get('initial_object_z_m',0),'error':error}
        expected_count=len(labels)
        review['controller']='lilo_vla' if environment.get('ARX_LILO_POLICY_URL') else 'curobo_teacher'
        review['execution_pass']=bool(review['physical_grasp_success'] and not bad and violation<=1e-6 and len(fingers)==2 and len(commands)==expected_count and len(executions)==expected_count)
        review['configuration_audit_pass']=bool(unchanged and not changed and clear) if args.audit else None
        review['pass']=bool(review['execution_pass'] and (review['configuration_audit_pass'] if args.audit else True))
        review['pass_scope']='execution and configuration audit' if args.audit else 'execution; runtime configuration audit disabled'
        save('verification.json',review);print(json.dumps(review,indent=2),flush=True)
    if not args.skip_encoding and (out/'frames/frames.json').exists():
        stage('encode_video',sys.executable,Path(__file__).with_name('encode_reproduction.py'),[out],environment)
    return 0 if exit_code==0 and review['pass'] else 1


if __name__=='__main__':raise SystemExit(main())
