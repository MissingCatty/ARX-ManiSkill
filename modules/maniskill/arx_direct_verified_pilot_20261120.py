"""Retain six complete executions and run only the four unclaimed cases."""
import argparse
from pathlib import Path
import shutil

import arx_direct_terminal_pilot_20261120 as previous
import arx_direct_verifier_20261120 as verifier

pilot = previous.pilot
SELF = Path(__file__).resolve()
PRIOR = SELF.parent/'outputs/arx_direct_grasp_pilot_20261120_04'


def prepare(root):
    assert pilot.common.read(PRIOR/'suite_timing.json')['all_owned_services_exited']
    previous.prepare(root)
    dependencies = [SELF, Path(previous.__file__).resolve(), Path(verifier.__file__).resolve(),
                    SELF.with_name('arx_direct_verified_simulator_20261120.py')]
    retained = []
    for group in pilot.GROUPS:
        arm = root/group/'fixed_graspcl';old = PRIOR/group/'fixed_graspcl'
        for claim_path in sorted((old/'claims').glob('*.json')):
            source = old/'trials'/claim_path.stem
            assert (source/'result.json').exists()
            target = arm/'trials'/source.name
            shutil.copytree(source, target)
            original_evaluation = target/'independent_evaluation.json'
            shutil.copy2(original_evaluation, target/'independent_evaluation_before_stage_contract_fix.json')
            verifier.verify_trial(target)
            assert pilot.common.sha(target/'result.json') == pilot.common.sha(source/'result.json')
            claim = pilot.common.read(claim_path)
            claim.update(state='complete', result_sha256=pilot.common.sha(target/'result.json'),
                finished_at=pilot.bench.core.now(), inherited_from=str(source), replayed=False,
                retained_complete_execution=True, verifier_stage_contract_corrected=True)
            pilot.common.save(arm/'claims'/claim_path.name, claim)
            retained.append({'group': group, 'id': source.name, 'source': str(source),
                'original_result_sha256': pilot.common.sha(source/'result.json'),
                'original_trace_sha256': pilot.common.sha(source/'execution.jsonl'),
                'new_independent_evaluation': pilot.common.read(target/'independent_evaluation.json')})
    assert len(retained) == 6
    for directory in [root, *[root/n for n in pilot.GROUPS], *[root/n/'fixed_graspcl' for n in pilot.GROUPS]]:
        reg = pilot.common.read(directory/'registration.json')
        reg['source_hashes'].update({str(p): pilot.common.sha(p) for p in dependencies})
        reg.update(actual_run_entry=str(SELF), continuation_of=str(PRIOR), inherited_executions=retained,
            remaining_physical_execution_slots=4, physical_execution_budget_total=10,
            algorithm_changes_since_prior=0, verifier_change='Require pregrasp only in the fixed-pregrasp arm; preserve all physical success gates.')
        pilot.common.save(directory/'registration.json', reg)
    receipt = pilot.common.read(root/pilot.EPOCH/'registration.json')
    receipt['root_registration_sha256'] = pilot.common.sha(root/'registration.json')
    receipt.update(inherited_executions=6, remaining_executions=4, actual_replays=0)
    pilot.common.save(root/pilot.EPOCH/'registration.json', receipt)
    for p in dependencies:shutil.copy2(p, root/'source'/p.name)
    pilot.common.save(root/'retained_execution_verification.json', {'passed': True,
        'retained_executions': retained, 'physical_executions_replayed': 0,
        'original_batch_files_changed': 0, 'remaining_physical_executions': 4})
    for group in pilot.GROUPS:
        pilot.bench.base.existing.summarize(root/group/'fixed_graspcl', 'prepared', {})
        pilot.common.save(root/group/'progress.json', pilot.common.read(root/group/'fixed_graspcl/progress.json'))
    pilot.aggregate(root, 'prepared_continuation')
    pilot.parent.guard(pilot.common.read(root/'registration.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','run'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args();root = args.output.resolve()
    if args.stage == 'prepare':prepare(root)
    else:
        _, baseline = pilot.bench.core.batch.original_modules();verifier.install(baseline)
        pilot.BACKEND_GPU = pilot.common.read(root/'registration.json')['runtime_backend_gpu']
        pilot.Backends.training_plan_check = previous.fixture.training
        pilot.manager_class = previous.manager
        pilot.parent.CODE = pilot.CODE
        # The base controller assigns its original simulator filename. Replace
        # that assignment in memory; all source files stay frozen on disk.
        import inspect
        source = inspect.getsource(pilot.run)
        source = source.replace("CODE/'arx_direct_simulator_20261120.py'", "CODE/'arx_direct_verified_simulator_20261120.py'")
        ns = dict(pilot.run.__globals__)
        exec(compile(source, str(SELF)+':verified_simulator_entry', 'exec'), ns)
        ns['run'](root)
