"""Resume already terminalized slots, archiving the previous startup marker."""
from pathlib import Path
import argparse
import shutil
import subprocess
import tempfile

import graspcl_unseen_resume_tests_20261119 as controller
from short_lift_services import InputGuard

common=controller.common
SELF=Path(__file__).resolve()
EPOCH='continuation_epoch09'


def prepare(root):
    state=subprocess.check_output(['systemctl','--user','show',
        'maniskill-arx-graspcl-unseen-50-100-20261119-08.service','-p','MainPID','--value'],text=True).strip()
    assert state=='0','Previous owned controller must exit before explicit resumption'
    receipt=common.read(root/'continuation_epoch08/registration.json')
    for key in ['retained_terminal_evidence_sha256','frozen_bank_sha256','frozen_input_manifest_sha256','source_sha256']:
        controller.verify_hashes(receipt[key])
    arm=root/'A_zerograsp/fixed_zerograsp'
    claims=list((arm/'claims').glob('*.json'));assert len(claims)==22
    assert all(common.read(p)['state']=='complete' for p in claims)
    interrupted=common.read(arm/'trials'/controller.INTERRUPTED/'result.json')
    assert interrupted['infrastructure_interruption'] and interrupted['grasp_outcome']=='unknown'
    assert not any((root/n).exists() for n in controller.bench.PHASES[2:])
    epoch=root/EPOCH;epoch.mkdir(exist_ok=False)
    # Exercise the exact original bootstrap gate without inference or execution.
    with tempfile.TemporaryDirectory(dir=epoch,prefix='marker-check-') as temporary:
        campaign=Path(temporary);common.save(campaign/'registration.json',common.read(root/'A_zerograsp/registration.json'))
        guard=InputGuard(campaign/'resident');warmup=campaign/'resident/neural_warmup'
        marker=campaign/'test_started.json';marker.write_text('historical')
        try:guard.check(warmup,'zerograsp')
        except RuntimeError as error:assert 'after test claims' in str(error)
        else:raise AssertionError('Historical marker must reproduce the prestart refusal')
        marker.rename(campaign/'historical_test_started.json')
        assert guard.check(warmup,'zerograsp')['bootstrap_only']
        marker.write_text('current')
        try:guard.check(warmup,'zerograsp')
        except RuntimeError:pass
        else:raise AssertionError('Current test phase must still prohibit bootstrap warmup')
    receipt.update(runtime_epoch=9,created_at=controller.bench.core.now(),
        prestart_epoch08='Zero new test claims; the old test_started marker blocked fixed training-only warmup.',
        old_A_resident_archive=str(epoch/'A_resident_epoch08'),
        bootstrap_gate_change='Archive the previous epoch marker before warmup. Original InputGuard is unchanged; new test_started is written after all components are ready.',
        reused_interruption_terminal_record=str(root/'continuation_epoch08/interruption_terminal_record.json'))
    receipt['source_sha256'][str(SELF)]=common.sha(SELF)
    common.save(epoch/'registration.json',receipt)
    shutil.copy2(SELF,epoch/SELF.name)
    for name in ['suite_timing.json','progress.json','test_started.json']:
        p=root/'A_zerograsp'/name
        if p.exists():shutil.copy2(p,epoch/('A_previous_'+name))
    common.save(epoch/'preflight.json',{'passed':True,'historical_marker_refusal_reproduced':True,
        'archived_marker_allows_bootstrap':True,'current_marker_still_blocks_bootstrap':True,
        'original_input_guard_unchanged':True,'new_test_claims_in_epoch08':0,
        'retained_A_complete_results':21,'retained_A_unknown_interruption':1,
        'remaining_test_executions':278,'actual_grasps_or_models_loaded':0})


def retain_terminal_interruption(root):
    epoch=root/EPOCH;receipt=common.read(epoch/'registration.json')
    controller.verify_hashes(receipt['retained_terminal_evidence_sha256'])
    trial=root/'A_zerograsp/fixed_zerograsp/trials'/controller.INTERRUPTED
    result=common.read(trial/'result.json')
    assert result['infrastructure_interruption'] and result['grasp_outcome']=='unknown'
    assert common.read(trial/'independent_evaluation.json')['passed']
    marker=root/'A_zerograsp/test_started.json'
    if marker.exists():marker.rename(epoch/'A_test_started_previous.json')
    (root/'A_zerograsp/resident').rename(epoch/'A_resident_epoch08')
    common.save(epoch/'interruption_terminal_record.json',{'passed':True,'reused_existing_terminal_record':True,
        'id':controller.INTERRUPTED,'outcome':'unknown','replayed':False})


def run(root):
    controller.EPOCH=EPOCH
    controller.terminalize_interruption=retain_terminal_interruption
    controller.run(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    (prepare if args.stage=='prepare' else run)(args.output.resolve())
