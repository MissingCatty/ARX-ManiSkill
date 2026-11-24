"""Preserve fixed benchmark inputs, invalidate obsolete results, register 2x200."""
from copy import deepcopy
from pathlib import Path
from datetime import datetime
import argparse
import json
import os
import shutil
import subprocess

import graspcl_pipeline as common
from graspcl_execute import guard

CODE = Path(__file__).resolve().parent
OLD_NAMES = ['arx_baseline_20261116_01', 'arx_baseline_20261116_parallel_01',
    'arx_zerograsp_graspcl_200_20261117_01', 'arx_zerograsp_graspcl_200_resident_20261117_02',
    'arx_zerograsp_native_post_20261117_01', 'arx_zerograsp_native_post_20261117_02']


def now():
    return datetime.now().astimezone().isoformat()


def snapshot(inputs):
    source = CODE / 'outputs/arx_baseline_20261116_parallel_01'
    reg = common.read(source / 'registration.json')
    guard(reg)
    previous = CODE / 'outputs/arx_zerograsp_native_post_20261117_02'
    if common.read(previous / 'progress.json')['phase'] != 'stopped':
        raise RuntimeError('Previous benchmark must drain before snapshot/cleanup')
    for kind in ['zerograsp', 'graspcl', 'curobo']:
        stopped = common.read(previous / 'resident' / (kind+'_stopped.json'))
        try:
            os.kill(stopped['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Old resident is still alive')
    inputs.mkdir(parents=True, exist_ok=False)
    sources = dict(reg['source_hashes'])
    reset_count, admission_failed = 0, 0
    for scheduled in reg['schedule']:
        slot = scheduled['id']
        admission = common.read(source / 'admission' / slot / 'admission.json')
        assert all(admission['entry'][k] == scheduled[k] for k in ['id','model_id','layout','seed'])
        # Preserve scene/viewpoint/reset input, never a historical outcome.
        keys = ['id','model_id','family','layout','seed','scene','observation','admission']
        bound = {'passed': admission['passed'],
                 'entry': {k: admission['entry'][k] for k in keys if k in admission['entry']}}
        common.save(inputs / 'admission' / slot / 'admission.json', bound)
        admission_failed += not admission['passed']
        reset = source / 'trials' / slot / 'initial_conditions.json'
        if reset.is_file():
            target = inputs / 'resets' / slot / 'initial_conditions.json'
            target.parent.mkdir(parents=True)
            shutil.copy2(reset, target)
            reset_count += 1
    prior = CODE / 'outputs/arx_zerograsp_graspcl_200_resident_20261117_02'
    shutil.copy2(prior / 'reference/common_initial_conditions.json', inputs / 'common_initial_conditions.json')
    shutil.copy2(prior / 'storage_policy.json', inputs / 'storage_policy.json')
    bank = CODE / 'outputs/arx_zerograsp_graspcl_200_20261117_01/frozen_bank'
    complete = common.read(bank / 'bank_complete.json')
    for name, digest in complete['model_sha256'].items():
        # Old manifests can refer to the original bank, which is byte-identical.
        relative = Path(name).relative_to(Path(name.split('/model/')[0]) / 'model')
        assert common.sha(bank / 'model' / relative) == digest
    shutil.copytree(bank / 'model', inputs / 'frozen_bank/model')
    complete['model_sha256'] = {str(p): common.sha(p) for p in sorted((inputs / 'frozen_bank/model').rglob('*')) if p.is_file()}
    common.save(inputs / 'frozen_bank/bank_complete.json', complete)
    common.save(inputs / 'frozen_bank/provenance.json', {
        'source_bank_manifest_sha256': common.sha(bank / 'bank_complete.json'),
        'positive_demonstrations': complete['positive_demonstrations'],
        'negative_feedback': complete['negative_feedback'],
        'source_ids': complete['source_ids'], 'test_outcomes_reused': False,
        'memory_frozen_during_test': True, 'same_case_feedback_and_recall_excluded': True})
    warmup = CODE / 'outputs/arx_zerograsp_graspcl_physical_20261117_01/trials/B0069_S3_sparse_014_lemon'
    target = inputs / 'warmup' / warmup.name
    target.mkdir(parents=True)
    for name in ['state.json','closed_state.json','pregrasp_state.json','raw_grasps.npy']:
        shutil.copy2(warmup / name, target / name)
    shutil.copytree(warmup / 'camera', target / 'camera')
    shutil.copytree(warmup / 'network/raw_outputs', target / 'network/raw_outputs')
    manifest = {'created_at': now(), 'schedule': reg['schedule'], 'layouts': reg['layouts'],
        'success': reg['success'], 'table_height_m': reg['table_height_m'], 'camera': reg['camera'],
        'recording_camera': reg['recording_camera'], 'source_hashes': sources,
        'source_registration_sha256': common.sha(source / 'registration.json'),
        'warmup_trial': str(target), 'bank': str(inputs / 'frozen_bank'),
        'source_feedback_ids_in_memory_order': complete['source_ids'],
        'slots': 200, 'models': 40, 'reset_receipts': reset_count,
        'admission_failed': admission_failed, 'old_results_carried': 0,
        'observation_protocol': 'Replay fixed historical wrist-view joint goals; fresh RGB-D after motion. Original simulator target instance mask and bbox-derived historical view remain the registered protocol.'}
    manifest['input_sha256'] = {str(p): common.sha(p) for p in sorted(inputs.rglob('*')) if p.is_file()}
    common.save(inputs / 'manifest.json', manifest)
    print('INPUT SNAPSHOT', inputs, reset_count, 'resets;', admission_failed, 'admission failures', flush=True)


def clean(inputs, replacement):
    manifest = common.read(inputs / 'manifest.json')
    guard({'source_hashes': {**manifest['source_hashes'], **manifest['input_sha256']}})
    units = ['maniskill-arx-native-post-20261117-02.service',
        'maniskill-arx-native-post-report-20261117-02.service',
        'maniskill-arx-grasp-visuals-20261116-01.service']
    for unit in units:
        state = subprocess.check_output(['systemctl','--user','show',unit,'--property=ActiveState','--value'],text=True).strip()
        if state not in ['inactive','failed']:
            raise RuntimeError('Old owned service must stop: '+unit)
    receipt = {'requested_by_user': True, 'invalidated_at': now(),
        'reason': 'Original handwritten-rule results discarded; both native-post/resident pipelines restart with independent fresh 200-slot allocations.',
        'preserved_inputs': str(inputs), 'replacement': str(replacement), 'directories': []}
    common.save(inputs / 'cleanup_manifest.json', receipt)
    for name in OLD_NAMES:
        root = CODE / 'outputs' / name
        if root.is_symlink() or root.parent != CODE / 'outputs':
            raise RuntimeError('Cleanup target is not an owned experiment directory')
        registration = (root / 'registration.json').read_bytes()
        stop = (root / 'STOP').read_bytes() if (root / 'STOP').is_file() else b''
        receipt['directories'].append({'path': str(root), 'registration_sha256': common.sha(root / 'registration.json'),
            'results_deleted': False})
        common.save(inputs / 'cleanup_manifest.json', receipt)
        shutil.rmtree(root)
        root.mkdir()
        (root / 'registration.json').write_bytes(registration)
        (root / 'STOP').write_bytes(stop + b'\nInvalidated by user; results removed. Do not resume.\n')
        common.save(root / 'invalidated.json', {'status':'invalidated','results_deleted':True,
            'reason':receipt['reason'],'replacement':str(replacement),'cleanup_manifest':str(inputs / 'cleanup_manifest.json')})
        receipt['directories'][-1]['results_deleted'] = True
        common.save(inputs / 'cleanup_manifest.json', receipt)
    receipt['removed_logs']=[]
    for name in ['arx_baseline_20261116_01.log','arx_baseline_20261116_parallel_01.log',
        'arx_grasp_visuals_20261116_01.log','arx_graspcl_200_20261117_01.log',
        'arx_graspcl_200_resident_20261117_02.log','arx_native_post_20261117_02.log',
        'arx_native_post_report_20261117_02.log']:
        path=CODE / 'logs' / name
        if path.is_file():
            path.unlink()
            receipt['removed_logs'].append(str(path))
    common.save(inputs / 'cleanup_manifest.json',receipt)
    print('CLEANED', len(receipt['directories']), 'obsolete result directories', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['snapshot','clean'])
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--replacement',type=Path)
    args=parser.parse_args()
    if args.stage=='snapshot': snapshot(args.inputs.resolve())
    else: clean(args.inputs.resolve(),args.replacement.resolve())
