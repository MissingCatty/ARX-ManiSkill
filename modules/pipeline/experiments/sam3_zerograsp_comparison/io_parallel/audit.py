"""Independent readback of both methods, including native candidate identity."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import sys

from arx_project.artifacts import atomic_json
from arx_project.json_io import read_json
from arx_project.layout import ROOT,now,sha256
from ..cohort import check_pair
from .contracts import METHODS,verify_io
from .audit_helpers import native_trial_origin,rebuild_generator_candidates


def audit(root):
    from modules.curobo.fingertip_contact import load_profile
    sys.path[:0]=[str(ROOT/'modules/maniskill'),str(ROOT/'modules/maniskill/planning_common'),
        str(ROOT/'modules/pipeline/experiments/repaired_grasp_benchmark'),
        str(ROOT/'modules/pipeline/experiments/rayst3r_graspgenx')]
    from planning_common.contact_execution import verify_trial
    from planning_common.candidates import load_candidates
    from modules.zerograsp.arx_grasp_adapter import common_candidates
    reg=read_json(root/'registration.json');issues=[];results=[];videos=[]
    for method in METHODS:
        folder=root/'methods'/method
        for pair in reg['cohort']['pairs']:
            claim=folder/'claims'/(pair['id']+'.json');out=folder/'trials'/pair['id']
            if not claim.exists():results.append({'method':method,'id':pair['id'],'state':'not_started'});continue
            receipt=read_json(claim)
            if receipt.get('state')!='complete':results.append({'method':method,'id':pair['id'],'state':'interrupted'});continue
            try:
                check_pair(pair)
                if (sha256(out/'result.json')!=receipt['result_sha256']
                        or sha256(out/'latency.json')!=receipt['latency_sha256']):raise RuntimeError('Claim hash mismatch')
                result=read_json(out/'result.json')
                work=root/'independent_readback'/method/pair['id'];work.mkdir(parents=True,exist_ok=True)
                if not pair.get('common_precondition_failure'):
                    if not read_json(out/'paired_restore_verification.json')['passed']:raise RuntimeError('State restore not verified')
                    for name in ['state.json','camera/color.png','camera/mask.png','camera/depth_m.npy']:
                        if sha256(out/name)!=pair['input_hashes'][name]:raise RuntimeError('Paired input changed')
                    if not (out/'candidates/candidate_pool.json').exists() and result['reason']!='rayst3r_empty_pointcloud':
                        raise RuntimeError('Missing executed/generated candidate snapshot')
                    if (out/'candidates/candidate_pool.json').exists():
                        pool=read_json(out/'candidates/candidate_pool.json')
                        verify_io(out,require_decode=bool(pool['candidates']))
                        if method=='zerograsp':
                            origin=native_trial_origin(root,method,pair['id'])
                            if sha256(origin/'network_candidates.json')!=sha256(out/'network_candidates.json'):
                                raise RuntimeError('Copied native provenance receipt differs from its origin')
                            if sha256(origin/'postprocessed_grasps.npy')!=sha256(out/'postprocessed_grasps.npy'):
                                raise RuntimeError('Copied native candidate array differs from its origin')
                            native,_=load_candidates(origin)
                            expected=common_candidates(native,read_json(out/'state.json')['camera_world_optical'])
                        else:
                            replay=work/'candidate_rebuild'
                            if replay.exists():shutil.rmtree(replay)
                            replay.mkdir()
                            shutil.copytree(out/'graspgenx_raw',replay/'graspgenx_raw')
                            shutil.copy2(out/'state.json',replay/'state.json')
                            rebuild_generator_candidates(replay)
                            expected=read_json(replay/'candidates/candidate_pool.json')['candidates']
                        if pool['candidates']!=expected:raise RuntimeError('Generator candidates changed or extra postprocessing applied')
                if result.get('planning_success'):
                    gate=read_json(out/'prepared_plan_execution_gate.json')
                    if not gate['passed'] or gate['selected_arm']!=result['selected_arm']:raise RuntimeError('Execution handoff failed')
                    for name,digest in gate['artifact_hashes'].items():
                        if sha256(out/name)!=digest:raise RuntimeError('Prepared plan changed')
                for name in ['result.json','execution.jsonl','contact_entities.json','execution_origin.json']:
                    if (out/name).is_file():shutil.copy2(out/name,work/name)
                verify_trial(work,load_profile());evidence=read_json(work/'independent_evaluation.json')
                if not evidence['passed'] or evidence.get('independently_successful')!=result['success']:
                    raise RuntimeError('Independent physical result differs')
                videos.extend(out.glob('*.mp4'))
                results.append({'method':method,'id':pair['id'],'state':'complete','success':result['success']})
            except Exception as error:issues.append(method+'/'+pair['id']+': '+str(error))
    import imageio_ffmpeg
    binary=imageio_ffmpeg.get_ffmpeg_exe()
    def decode(path):
        r=subprocess.run([binary,'-v','error','-threads','1','-i',str(path),'-threads','1','-f','null','-'],capture_output=True,timeout=300)
        return {'file':str(path.relative_to(root)),'passed':r.returncode==0}
    with ThreadPoolExecutor(max_workers=4) as pool:decoded=list(pool.map(decode,videos))
    if any(not row['passed'] for row in decoded):issues.append('Video decode failed')
    if not read_json(root/'services_stopped.json')['all_owned_exited']:issues.append('Owned services did not exit')
    from .continuation import verify_inheritance
    try:verify_inheritance(root,reg,full=True)
    except Exception as error:issues.append('Continuation inheritance: '+str(error))
    record={'at':now(),'passed':not issues,'record_integrity':not issues,'issues':issues,
        'results':results,'videos':decoded,
        'task_complete':sum(r['state']=='complete' for r in results)==148,
        'authorized_schedule_complete':all((root/'methods'/m/'complete.json').exists() for m in METHODS)}
    atomic_json(root/'acceptance.json',record);return record
