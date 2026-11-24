"""Clean continuation changing only authored-space mesh serialization."""
import argparse
import copy
import fcntl
from pathlib import Path
import shutil
import time

from .common import ROOT,read,write,sha,digest,verify_files,space
from .resume import exit_proof,copy_and_recover
from .watchdog_recovery import inspect_records,probe_storage
from .storage import register_bulk_storage,container_storage_mounts


RELATIVE='scripts/curobo_arx/sdg_stage1/scene_transport.py'


def prepare(parent,out,verification):
    parent,out=Path(parent).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Never replace an existing registration')
    evidence=read(verification)
    if evidence.get('pass') is not True or evidence.get('changed_source_sha256')!=sha(ROOT/RELATIVE):
        raise ValueError('Cache source has no matching passing offline ARX verification')
    lock=(parent/'coordinator.lock').open('r')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        state=read(parent/'state.json');old=read(parent/'registration.json');launch=read(parent/'launch.json')
        if state['status']!='stopped' or state.get('error') or state.get('unresolved_groups'):
            raise ValueError('Require a clean stopped parent with no unresolved execution')
        if time.time()>=launch['stop_unix']:raise ValueError('Original deadline expired')
        proof=exit_proof(parent)
        verify_files({**old['execution_source_files'],**old['dependencies']})
        if sha(Path(old['source_snapshot'])/RELATIVE)!=evidence['baseline_source_sha256']:
            raise ValueError('Offline verification baseline is not the running production source')
        rows=inspect_records(parent)
        if any(r['state'] in ('claimed','executing','writing','interrupted') for r in rows):
            raise ValueError('Unresolved parent attempt cannot be hidden by an optimization cutover')
        space(ROOT/'runs');space('/coolas/data');out.mkdir()
        previous=Path(old['source_snapshot']);source=out/'source_snapshot/COOLAS-ARX'
        source.parent.mkdir();shutil.copytree(previous,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
        (source.parent/'curobo').symlink_to(ROOT.parent/'curobo',target_is_directory=True)
        shutil.copy2(ROOT/RELATIVE,source/RELATIVE)
        source_files=copy.deepcopy(old['source_files']);source_files[str(ROOT/RELATIVE)]=sha(ROOT/RELATIVE)
        execution={str(source/Path(p).relative_to(previous)) if Path(p).is_relative_to(previous) else p:v
                   for p,v in old['execution_source_files'].items()}
        execution[str(source/RELATIVE)]=sha(source/RELATIVE);verify_files(execution)
        inherited=copy_and_recover(parent/'ledger.sqlite3',out/'ledger.sqlite3',proof)
        if inherited['recovered']:raise ValueError('Clean cutover must not recover unknown attempts')
        write(out/'exit_proof.json',proof);write(out/'recovery_receipt.json',inherited)
        shutil.copy2(verification,out/'offline_verification.json')
        reg=copy.deepcopy(old)
        reg.update(created_unix=time.time(),source_snapshot=str(source),source_files=source_files,
            execution_source_files=execution,source_identity=digest(source_files),
            deadline_unix=launch['stop_unix'],campaign_started_unix=old.get('campaign_started_unix',launch['started_unix']),
            lineage=old.get('lineage',[])+[{'campaign':str(parent),'registration_sha256':sha(parent/'registration.json'),
                'state_sha256':sha(parent/'state.json'),'status':'stopped'}],
            geometry_cache_optimization={'version':'authored_actor_relative_chain_v2',
                'offline_verification_sha256':sha(out/'offline_verification.json'),
                'no_rounding_or_geometry_simplification':True,'original_precision_thresholds_unchanged':True},
            resume={'parent':str(parent),'proof_sha256':sha(out/'exit_proof.json'),
                'recovery_receipt_sha256':sha(out/'recovery_receipt.json'),**inherited,
                'runtime_change':'Use authored child transform chains for mesh cache identity; fresh live actor poses and original vertex equivalence checks remain unchanged',
                'frozen_unchanged':'Robot, geometry, physics, cameras, renderers, placements, model weights, planning algorithm, success and admission criteria'},
            bulk_storage=register_bulk_storage(out,Path(old['bulk_storage']['root']).parent/out.name))
        write(out/'registration.json',reg);write(out/'catalog.json',reg['catalog'])
        for name in ('storage_policy.json','method_receipt.json'):shutil.copy2(parent/name,out/name)
        probes=probe_storage(Path(reg['bulk_storage']['root']),container_storage_mounts(out,reg))
        write(out/'storage_preflight_probes.json',probes)
        if not probes['pass']:raise RuntimeError('Runtime-user storage probe failed')
        write(out/'source_change_receipt.json',{'changed_files':[RELATIVE],
            'parent_sha256':sha(previous/RELATIVE),'new_sha256':sha(source/RELATIVE),
            'all_other_execution_sources_identical':True,'old_results_and_STOP_retained':True,
            'deadline_and_cumulative_attempt_budget_retained':True,
            'frozen_configuration_equal':all(reg[k]==old[k] for k in ('catalog','quotas','render_configs',
                'render_camera_clipping','visual_geometry_policy','graspness','frozen','devices','model_reference'))})
        write(out/'state.json',{'status':'prepared','successes':len(inherited['inherited_success_ids']),
            'new_successes':0,'physical_attempts':inherited['inherited_physical_attempts'],'quota_complete':False})
        return reg
    finally:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from',dest='parent',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--verification',required=True);args=parser.parse_args()
    result=prepare(args.parent,args.out,args.verification)
    print({'campaign':args.out,'inherited_successes':len(result['resume']['inherited_success_ids']),
           'inherited_physical_attempts':result['resume']['inherited_physical_attempts']})
