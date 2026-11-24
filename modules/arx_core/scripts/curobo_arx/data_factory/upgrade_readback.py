"""New immutable collection registration for readback-only optimization."""
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


CHANGED = (
    'arx-sim/arx_sim/diagnostics/research_fingerprint.py',
    'scripts/curobo_arx/data_factory/fast_fingerprint.py',
    'scripts/curobo_arx/data_factory/frozen.py',
    'scripts/curobo_arx/data_factory/runtime.py',
)


def prepare(parent,out):
    parent,out=Path(parent).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Never overwrite an old optimization registration')
    lock=(parent/'coordinator.lock').open('r')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        proof=exit_proof(parent)
        state=read(parent/'state.json')
        if state['status']!='stopped' or state.get('error') or state.get('unresolved_groups'):
            raise ValueError('Optimization requires a clean stopped parent, not an unresolved fault')
        old=read(parent/'registration.json');launch=read(parent/'launch.json')
        if time.time()>=launch['stop_unix']:raise ValueError('Original collection deadline expired')
        verify_files({**old['execution_source_files'],**old['dependencies']})
        rows=inspect_records(parent)
        if any(r['state'] in ('claimed','executing','writing','interrupted') for r in rows):
            raise ValueError('Do not upgrade while any attempt is unresolved')
        space(ROOT/'runs');space('/coolas/data')
        out.mkdir()
        previous=Path(old['source_snapshot'])
        source=out/'source_snapshot/COOLAS-ARX'
        source.parent.mkdir(parents=True)
        shutil.copytree(previous,source,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
        (source.parent/'curobo').symlink_to(ROOT.parent/'curobo',target_is_directory=True)
        source_files=copy.deepcopy(old['source_files']);changes=[]
        for relative in CHANGED:
            original=previous/relative;updated=ROOT/relative;target=source/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(updated,target)
            changes.append({'path':relative,'parent_sha256':sha(original) if original.exists() else None,
                            'new_sha256':sha(target)})
            source_files[str(updated)]=sha(updated)
        execution={}
        for path,value in old['execution_source_files'].items():
            p=Path(path)
            execution[str(source/p.relative_to(previous)) if p.is_relative_to(previous) else path]=value
        for relative in CHANGED:execution[str(source/relative)]=sha(source/relative)
        verify_files(execution)
        oracle=previous/'arx-sim/arx_sim/diagnostics/research_fingerprint.py'
        inherited=copy_and_recover(parent/'ledger.sqlite3',out/'ledger.sqlite3',proof)
        if inherited['recovered']:raise ValueError('A clean optimization cutover must not recover unknown executions')
        write(out/'exit_proof.json',proof);write(out/'recovery_receipt.json',inherited)
        reg=copy.deepcopy(old)
        reg.update(created_unix=time.time(),source_snapshot=str(source),source_files=source_files,
            execution_source_files=execution,source_identity=digest(source_files),
            deadline_unix=launch['stop_unix'],campaign_started_unix=old.get('campaign_started_unix',launch['started_unix']),
            lineage=old.get('lineage',[])+[{'campaign':str(parent),'registration_sha256':sha(parent/'registration.json'),
                'state_sha256':sha(parent/'state.json'),'status':'stopped'}],
            fingerprint_optimization={'version':'batched_live_properties_v1','verify_scalar_equivalence':True,
                'progress_interval_s':2.,'scalar_oracle':{'path':str(oracle),'sha256':sha(oracle)},
                'same_USD_traversal_and_all_robot_and_object_properties':True},
            resume={'parent':str(parent),'proof_sha256':sha(out/'exit_proof.json'),
                'recovery_receipt_sha256':sha(out/'recovery_receipt.json'),**inherited,
                'runtime_change':'Batch read identical live object properties and throttle progress-only writes; first use compared against immutable original scalar capture',
                'frozen_unchanged':'Robot, physics, cameras, placements, neural proposals/weights, original success and admission criteria'},
            bulk_storage=register_bulk_storage(out,Path(old['bulk_storage']['root']).parent/out.name))
        reg['dependencies'][str(oracle)]=sha(oracle)
        # New optimized files are execution source, not robot or scene changes.
        write(out/'registration.json',reg);write(out/'catalog.json',reg['catalog'])
        for name in ['storage_policy.json','method_receipt.json']:shutil.copy2(parent/name,out/name)
        probe=probe_storage(Path(reg['bulk_storage']['root']),container_storage_mounts(out,reg))
        write(out/'storage_preflight_probes.json',probe)
        if not probe['pass']:raise RuntimeError('Runtime-user storage probe failed')
        write(out/'source_change_receipt.json',{'changed_files':changes,'protected_inputs_identical':True,
            'catalog_and_quotas_identical':reg['catalog']==old['catalog'] and reg['quotas']==old['quotas'],
            'model_render_camera_visual_and_success_configuration_identical':all(reg.get(k)==old.get(k) for k in
                ['graspness','render_configs','render_camera_clipping','visual_geometry_policy','attempt_seconds','model_reference']),
            'old_STOP_and_all_results_retained':True})
        write(out/'state.json',{'status':'prepared','successes':len(inherited['inherited_success_ids']),
            'new_successes':0,'physical_attempts':inherited['inherited_physical_attempts'],'quota_complete':False})
        return reg
    finally:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--from',dest='parent',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();r=prepare(args.parent,args.out)
    print({'campaign':args.out,'inherited_successes':len(r['resume']['inherited_success_ids']),
           'deadline_unix':r['deadline_unix'],'changed_files':list(CHANGED)})
