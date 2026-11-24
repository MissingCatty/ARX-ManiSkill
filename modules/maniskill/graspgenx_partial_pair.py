"""Five matched ARX trials per method: GT partial-cloud GraspGenX vs ZeroGrasp."""
from arx_project.layout import canonical_path
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import graspgenx_partial_geometry as geom
from graspgenx_partial_geometry import save, sha

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
DEFAULT = CODE/'outputs/arx_graspgenx_partial_topdown_pair_20261120_01'
ARMS = ['zerograsp', 'graspgenx']
GPUS = {'zerograsp':'GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169',
        'graspgenx':'GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b',
        'curobo':'GPU-3164d5f3-6a9a-17b4-53f9-a70e3a82b834'}


def read(path):return json.loads(Path(path).read_text())
def now():return datetime.now().astimezone().isoformat()


def large_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024), b''):digest.update(data)
    return digest.hexdigest()


def guard(reg):
    for name, expected in reg['source_sha256'].items():
        assert sha(name) == expected, 'Frozen source changed: '+name


def prepare(root):
    assert not (root/'registration.json').exists()
    root.mkdir(parents=True,exist_ok=True)
    for folder in ['entries','claims','source','shared_inputs','resident','results','preparation']:
        (root/folder).mkdir(exist_ok=True)
    choices = [
        ('arx_native_post_20261117_01','B0031_S1_open_008_pudding_box'),
        ('arx_native_post_20261117_01','B0025_S1_open_007_tuna_fish_can'),
        ('arx_native_post_20261117_01','B0109_S1_open_025_mug'),
        ('arx_independent_20261117_01','I0066_S1_open_036_wood_block'),
        ('arx_native_post_20261117_01','B0199_S1_open_043_phillips_screwdriver')]
    entries, evidence = [], {}
    for inputs, key in choices:
        path = CODE/'benchmark_inputs'/inputs/'admission'/key/'admission.json'
        admission = read(path);assert admission['passed']
        entry = deepcopy(admission['entry'])
        entry.pop('admission',None)
        entry.update(source_admission=str(path), campaign_registration=str(root/'registration.json'))
        entries.append(entry);save(root/'entries'/(key+'.json'),entry)
        evidence[str(path)] = sha(path)
    warmup = CODE/'benchmark_inputs/arx_native_post_20261117_01/warmup/B0069_S3_sparse_014_lemon'
    assert all((warmup/p).exists() for p in ['state.json','closed_state.json','camera/depth.png'])
    # Verify the generator-reference / EEF / TCP conversion before loading models.
    from scipy.spatial.transform import Rotation
    for rotation in Rotation.random(32,random_state=0).as_matrix():
        eef=np.eye(4);eef[:3,:3]=rotation;eef[:3,3]=[.1,-.2,.3]
        contact=geom.generator_to_contact(eef@geom.T_EEF_GENERATOR)
        restored=contact.copy();restored[:3,3]-=.043*restored[:3,2]
        assert np.allclose(restored,eef,atol=1.e-12)
    down=np.diag([1.,-1.,-1.,1.]);assert geom.angle_from_down(down)<1.e-10
    assert geom.angle_from_down(np.eye(4))>179.9
    save(root/'preparation/geometry_contract.json',{'passed':True,'random_frame_roundtrips':32,
         'upward_pose_rejected':True,'physical_attempts':0})
    descriptor = root/'preparation/gripper_descriptor.json'
    save(descriptor,geom.gripper_descriptor())
    sources = dict(evidence)
    paths = [Path(__file__),CODE/'graspgenx_partial_geometry.py',CODE/'graspgenx_partial_services.py',
        CODE/'arx_pipeline/run.py',CODE/'arx_pipeline/baseline.py',CODE/'arx_pipeline/environment.py',
        CODE/'arx_pipeline/planner.py',CODE/'resident_backend.py',CODE/'resident_planner.py',
        CODE/'fast_resident_planner.py',CODE/'short_lift_planner.py',CODE/'short_lift_protocol.py',
        CODE/'table_truncation_model.py',CODE/'postprocess_backend.py',CODE/'postprocess_pipeline.py',
        CODE/'checked_native_services.py',CODE/'tmp/physical_table_pair_20261118_01.py',
        CODE/'benchmark_lifecycle.py',CODE/'grasp_visuals.py',descriptor,
        ROOT/'arx_core/arx-sim/config/controllers.yaml',ROOT/'arx_core/arx-sim/config/runtime.yaml']
    for p in paths:sources[str(p)] = sha(p)
    sources.update(read(descriptor)['source_sha256'])
    for p in (ROOT/'research_dependencies/GraspGenX/graspgenx').rglob('*.py'):
        sources[str(p)] = sha(p)
    for name in ['graspgenx_partial_pair.py','graspgenx_partial_services.py','graspgenx_partial_geometry.py']:
        shutil.copy2(CODE/name,root/'source'/name)
    weights = [ROOT/'research_dependencies/graspgenx_checkpoints/release/gen/epoch_736.pth',
        ROOT/'research_dependencies/graspgenx_checkpoints/release/dis/epoch_1056.pth',
        ROOT/'arx_core/servers/zerograsp/vendor/zerograsp_mainline_minimal/checkpoints/zerograsp_cvpr2025/zerograsp_synthetic.ckpt']
    model_sha = {str(p):large_sha(p) for p in weights}
    reg = {'created_at':now(),'protocol':geom.PROTOCOL,'total_slots':10,'scenes_per_arm':5,
        'arms':ARMS,'schedule':entries,'required_components':[*ARMS,'curobo'],
        'warmup_trial':str(warmup),'descriptor':str(descriptor),'gpu_by_component':GPUS,
        'simulation_gpu':GPUS['curobo'],'source_sha256':sources,'model_sha256':model_sha,
        'input_policy':{'zerograsp':'Rendered RGB + noiseless GT depth (1mm stored units) + GT instance mask + K',
                        'graspgenx':'Visible GT object points backprojected from the same depth and mask; world meters; no reconstructed or hidden surface points',
                        'paired_cache':'Identical frozen observation inputs; live measured state for planning; reset discrepancy checked'},
        'topdown_max_angle_deg':geom.TOPDOWN_DEGREES,'yaw':'Unrestricted among generated candidates',
        'graspgenx_planner':'Official GraspMoE, diffusion + dense top-face OBB, native discriminator ranking',
        'diffusion_samples':512,'graspgenx_topk_cap':None,
        'zerograsp_policy':'Native corrected/filtered/NMS candidates, no cap or custom symmetry',
        'candidate_pose_changes':'Original fixed rotations; 2mm axial insertion retraction with measured open-gripper/table/observed-closing-sweep gates',
        'ggx_retraction_budget':'TCP43 minus measured distal contact-zone near Z; explicitly not a predicted ZeroGrasp insertion depth',
        'robot_height':'Existing joint4 ~0.22m, unchanged','tcp_m':.043,
        'planned_lift_m':.03,'minimum_hold_lift_m':.01,'required_hold_seconds':3.,
        'collision_policy':'Unchanged original robot/scene; actual table; no nominal closed precheck; measured closed lift checked',
        'mpc':False,'graspcl':False,'shape_completion':False,'test_feedback_updates':False,
        'one_attempt_per_arm_scene':True,'old_results_carried':0,
        'selection_policy':'Five predeclared open-table target types; no selection using success outcomes',
        'scope':'Small paired engineering test; report all 5 scenes per arm including no-candidate and planning rejection; no population-level efficacy claim'}
    save(root/'registration.json',reg);save(root/'progress.json',{'phase':'prepared','completed':0,'registered':10})
    print('PREPARED',root,flush=True)


def display_pool(trial, arm):
    import graspcl_pipeline as common
    import postprocess_pipeline
    path = trial/'graspcl';pool = read(path/'candidate_pool.json')
    camera = np.asarray(read(trial/'state.json')['camera_world_optical'])
    for name in ['state.json','camera/params.json','camera/color.png','camera/depth.png','camera/mask.png']:
        destination = path/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(trial/name,destination)
    rows = []
    for c in pool['candidates']:
        view = deepcopy(c)
        if arm == 'graspgenx':view['metadata']['source_network_depth_m'] = 0. # plotting convention only
        rows.append(common.display_row(view,camera))
    np.save(path/'raw_grasps.npy',np.asarray(rows).reshape(-1,17),allow_pickle=False)
    name = 'GraspGenX' if arm=='graspgenx' else 'ZeroGrasp native post'
    save(path/'network_candidates.json',{'model_name':name,'score_label':name+' score',
        'candidate_protocol':geom.PROTOCOL,'count':len(rows),'physical_attempt':False})
    postprocess_pipeline.render(path,trial/'zg_outputs')
    if arm == 'graspgenx':shutil.copytree(trial/'zg_outputs',trial/'gg_outputs')


def candidates(trial,arm,descriptor):
    from postprocess_pipeline import processed_rows,canonical_candidates
    camera = np.asarray(read(trial/'state.json')['camera_world_optical'])
    original = []
    if arm=='zerograsp':
        rows,_ = processed_rows(trial);original = canonical_candidates(rows,camera)
        for c in original:c['score'] = c['metadata']['source_network_score']
    else:
        source = trial/'graspgenx_raw'
        poses = np.load(source/'grasps_generator_world.npy',allow_pickle=False)
        scores = np.load(source/'scores.npy',allow_pickle=False)
        tags = read(source/'inference.json')['branch_tags']
        params = descriptor['parameters']
        budget = max(0.,geom.TCP_M-(params['offset_open'][2]-params['extents_open'][2]/2))
        for index,(pose,score,tag) in enumerate(zip(poses,scores,tags)):
            contact = geom.generator_to_contact(pose)
            original.append({'pose':contact,'width':.088,'score':float(score),
                'point1':contact[:3,3]-.044*contact[:3,1],
                'point2':contact[:3,3]+.044*contact[:3,1],
                'metadata':{'origin':'graspgenx','source_generator_index':index,
                    'source_branch':tag,'source_model_score':float(score),
                    'source_generator_pose_world':pose,'retraction_budget_m':budget,
                    'width_policy':'Actual full open ARX aperture; closure uses original measured controller',
                    'generator_to_eef_transform':np.linalg.inv(geom.T_EEF_GENERATOR)}})
        # Compatibility tensor only: original generation arrays remain under graspgenx_raw.
        np.save(trial/'raw_grasps.npy',np.empty((0,17)),allow_pickle=False)
    kept, rejected = [], []
    for i,c in enumerate(original):
        angle = geom.angle_from_down(c['pose'])
        if angle > geom.TOPDOWN_DEGREES+1.e-5:
            rejected.append({'index':i,'reason':'outside_registered_topdown_cone','angle_deg':angle});continue
        if not 0 < c['width'] <= .088+1.e-6:
            rejected.append({'index':i,'reason':'outside_actual_aperture'});continue
        c['metadata']['topdown_angle_deg'] = angle;kept.append(c)
    kept.sort(key=lambda c:-c['score'])
    path = trial/'candidates';path.mkdir()
    (trial/'graspcl').symlink_to(path,target_is_directory=True)
    save(path/'candidate_pool.json',{'candidates':kept,'rejected':rejected,
        'model':arm,'candidate_protocol':geom.PROTOCOL,'graspcl_invoked':False,
        'raw_candidate_count':len(original),'topdown_candidate_count':len(kept),
        'compatibility_path_note':'graspcl is only the old planner data-format path; no GraspCL model is loaded'})


def shared_input(root,trial,entry,arm):
    cache = root/'shared_inputs'/entry['id']
    if arm=='zerograsp':
        cache.mkdir();shutil.copytree(trial/'camera',cache/'camera')
        shutil.copy2(trial/'state.json',cache/'state.json')
        _, cloud = geom.visible_points(trial)
        np.save(cache/'gt_partial_points_world.npy',cloud,allow_pickle=False)
        save(cache/'manifest.json',{'kind':'GT visible partial cloud','complete_mesh_used_for_inference':False,
            'points':len(cloud),'unit':'meter','world_frame':True,
            'sha256':{str(p.relative_to(cache)):sha(p) for p in cache.rglob('*') if p.is_file()}})
    else:
        saved = read(cache/'state.json');live = read(trial/'state.json')
        qa,qb = np.asarray(saved['qpos']),np.asarray(live['qpos'])
        a,b = np.asarray(saved['camera_world_optical']),np.asarray(live['camera_world_optical'])
        errors = {'joint_rad':float(np.max(np.abs(qa-qb))),
                  'camera_translation_m':float(np.linalg.norm(a[:3,3]-b[:3,3]))}
        assert errors['joint_rad'] <= .005 and errors['camera_translation_m'] <= .002,errors
        shutil.copytree(trial/'camera',trial/'fresh_camera_before_shared_cache')
        for p in (cache/'camera').iterdir():shutil.copy2(p,trial/'camera'/p.name)
        save(trial/'paired_observation_reset_check.json',{'passed':True,**errors})
    for name,expected in read(cache/'manifest.json')['sha256'].items():assert sha(cache/name)==expected
    shutil.copy2(cache/'gt_partial_points_world.npy',trial/'gt_partial_points_world.npy')
    save(trial/'shared_input_receipt.json',{'cache':str(cache),'manifest_sha256':sha(cache/'manifest.json'),
         'actual_camera_sha256':{p.name:sha(p) for p in (trial/'camera').iterdir()},
         'model':arm,'visible_only':True,'shape_completion':False})


class Services:
    def __init__(self,root,reg):self.root=root;self.reg=reg;self.children={};self.streams=[]

    def start(self):
        from arx_service_readiness import wait_ready
        from benchmark_lifecycle import mark_ready
        config = {'owner_pid':os.getpid(),'required_components':self.reg['required_components'],
                  'sockets':{k:str(CODE/'tmp'/('ggx20_'+k+'.sock')) for k in self.reg['required_components']}}
        for value in config['sockets'].values():assert not Path(value).exists(),value
        save(self.root/'resident/backend_config.json',config);self.config=config
        for kind in self.reg['required_components']:
            executable = {'zerograsp':canonical_path(ROOT/'tmp/arx-fullchain/env/bin/python'),
                'graspgenx':CODE/'.venv-graspgenx/bin/python','curobo':ROOT/'curobo/.venv/bin/python'}[kind]
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=GPUS[kind],OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                ARX_NEURAL_GPU=GPUS['zerograsp'],ARX_PLANNER_GPU=GPUS['curobo'],
                ARX_SIM_GPU=GPUS['curobo'],LILO_GPU=GPUS['curobo'],
                GRASP_TRAJOPT_POLICY='interactive',GRASPGENX_CHECKPOINT_DIR=str(ROOT/'research_dependencies/graspgenx_checkpoints'),
                GRASPGENX_GRIPPER_CFG_DIR=str(ROOT/'research_dependencies/GraspGenX/ext/gripper_descriptions'))
            env.pop('PYTHONPATH',None)
            command = ['bash','-c','source "$1"; shift; exec "$@"','ggx-paired-service',
                str(ROOT/'arx_core/scripts/curobo_arx/environment.sh'),str(executable),
                str(CODE/'graspgenx_partial_services.py'),'--kind',kind,'--root',str(self.root/'resident'),
                '--socket',config['sockets'][kind],'--warmup-trial',self.reg['warmup_trial'],
                '--descriptor',self.reg['descriptor']]
            stream = (self.root/'resident'/(kind+'.log')).open('x');self.streams.append(stream)
            self.children[kind]=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            health = wait_ready(Path(config['sockets'][kind]),self.root/'resident'/(kind+'_ready.json'),
                                self.children[kind],kind,timeout=1800)
            print('READY',kind,health['pid'],health['startup_seconds'],flush=True)
        mark_ready(self.root/'resident',self.reg['required_components'],config,self.children)

    def request(self,kind,trial,**flags):
        from resident_backend import rpc
        response=rpc(Path(self.config['sockets'][kind]),{'directory':str(trial),**flags},timeout=1800)
        if response['status']!='ok':raise RuntimeError(response.get('error',str(response)))
        return response

    def stop(self):
        from resident_backend import rpc
        for kind,process in reversed(list(self.children.items())):
            if process.poll() is None:
                try:rpc(Path(self.config['sockets'][kind]),{'command':'shutdown'},timeout=15)
                except Exception:process.terminate()
                try:process.wait(timeout=45)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        for stream in self.streams:stream.close()


def aggregate(root,phase):
    groups = {};cases=[]
    for arm in ARMS:
        rows=[]
        for path in sorted((root/'results'/arm).glob('*/result.json')):
            row=read(path);trial=path.parent
            if not (trial/'independent_evaluation.json').exists():continue
            plan=read(trial/'plan_result.json') if (trial/'plan_result.json').exists() else {}
            item={'id':row['id'],'success':row['success'],'reason':row['reason'],
                  'grasp_endpoint_reachable':plan.get('grasp_endpoint_reachable',False),
                  'planning_success':row['planning_success'],'physical_attempt':row['physical_attempt'],
                  'wall_seconds':row['wall_seconds']}
            rows.append(item);cases.append({'arm':arm,**item})
        groups[arm]={'registered':5,'completed':len(rows),'grasp_endpoint_reachable':sum(r['grasp_endpoint_reachable'] for r in rows),
            'planning_passes':sum(r['planning_success'] for r in rows),
            'physical_attempts':sum(r['physical_attempt'] for r in rows),'successes':sum(r['success'] for r in rows),
            'overall_success_rate':sum(r['success'] for r in rows)/5,
            'failure_reasons':dict(Counter(r['reason'] for r in rows if not r['success']))}
    result={'phase':phase,'registered':10,'completed':sum(g['completed'] for g in groups.values()),
            'updated_at':now(),'groups':groups,'cases':cases}
    save(root/'progress.json',result);return result


def run(root):
    import arx_runtime_imports
    arx_runtime_imports.install()
    from arx_quiet_diagnostics_20261119 import install as quiet
    quiet()
    import graspcl_batch
    from benchmark_lifecycle import require_ready
    from short_lift_protocol import install_execution_policy
    from resident_backend import rpc
    reg=read(root/'registration.json');guard(reg)
    assert not list((root/'claims').glob('*.json')),'Existing claimed slots require explicit continuation accounting'
    run_module,baseline=graspcl_batch.original_modules()
    install_execution_policy(run_module,baseline)
    original_make=run_module.make;keeper=[]
    def make(*args,**kwargs):
        env=original_make(*args,**kwargs)
        if not keeper:
            import sapien
            keeper.append(sapien.render.SapienRenderer(env._render_device))
        return env
    run_module.make=make
    service=Services(root,reg);started=time.perf_counter();phase='halted';error=None
    descriptor=read(reg['descriptor'])
    try:
        aggregate(root,'loading_components');service.start()
        startup=time.perf_counter()-started;save(root/'test_started.json',{'started_at':now(),'startup_seconds':startup,'claimed':0})
        for entry in reg['schedule']:
            for arm in ARMS:
                guard(reg);require_ready(root/'resident',reg['required_components'])
                assert shutil.disk_usage(root).free >= 1073741824
                slot=arm+'__'+entry['id'];claim=root/'claims'/(slot+'.json')
                assert not claim.exists();save(claim,{'state':'claimed','claimed_at':now(),'arm':arm,'id':entry['id']})
                trial=root/'results'/arm/entry['id'];trial.parent.mkdir(parents=True,exist_ok=True)
                events=[]
                def perform_worker(script,directory,gpu,log,**flags):
                    begin=time.perf_counter()
                    if script=='neural.py':
                        shared_input(root,directory,entry,arm)
                        response=service.request(arm,directory)
                        candidates(directory,arm,descriptor)
                        service.request('curobo',directory,refine=True,orientation_mode='fixed_orientation')
                        display_pool(directory,arm)
                    else:response=service.request('curobo',directory,selector='graspcl',**flags)
                    save(log,response)
                    events.append({'component':arm if script=='neural.py' else 'curobo','flags':flags,'wall_seconds':time.perf_counter()-begin})
                    return response['returncode']
                def worker(script,directory,gpu,log,**flags):
                    try:return perform_worker(script,directory,gpu,log,**flags)
                    except BaseException:
                        save(directory/'resident_request_error.json',{'component':script,'flags':flags,
                             'error':traceback.format_exc(),'infrastructure_outcome_unknown':True})
                        raise
                run_module.worker=worker
                print('BEGIN',arm,entry['id'],flush=True)
                result=run_module.trial(entry,trial,GPUS[arm],GPUS['curobo'],'right_wrist')
                save(trial/'component_timing.json',events)
                if (trial/'resident_request_error.json').exists() or result['reason']=='zerograsp_failed':
                    raise RuntimeError('Infrastructure failure preserved: '+slot)
                baseline.verify_trial(trial)
                if (trial/'plan_result.json').exists() and read(trial/'plan_result.json').get('selected'):
                    from short_lift_protocol import validate_selected
                    validate_selected(read(trial/'plan_result.json')['selected'])
                claim_value=read(claim);claim_value.update(state='complete',finished_at=now(),result_sha256=sha(trial/'result.json'))
                save(claim,claim_value);aggregate(root,'running')
                print('RESULT',arm,entry['id'],result['success'],result['reason'],flush=True)
        phase='complete'
    except BaseException:
        error=traceback.format_exc();save(root/'halt.json',{'at':now(),'error':error});print(error,flush=True)
    finally:
        service.stop()
        summary=aggregate(root,phase)
        save(root/'suite_timing.json',{'wall_seconds':time.perf_counter()-started,'startup_seconds':locals().get('startup'),
            'all_owned_services_exited':all(p.poll() is not None for p in service.children.values())})
        if phase=='complete':save(root/'final_summary.json',summary)
    if error:raise RuntimeError(error)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run'])
    p.add_argument('--output',type=Path,default=DEFAULT);args=p.parse_args()
    if args.stage=='prepare':prepare(args.output.resolve())
    else:run(args.output.resolve())


if __name__=='__main__':main()
