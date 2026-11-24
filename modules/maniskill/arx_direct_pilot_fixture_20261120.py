"""Complete the training-only RPC fixture before sending its planning request."""
import argparse
from pathlib import Path
import shutil
from unittest.mock import patch

import arx_direct_pilot_placement_20261120 as placed

pilot = placed.pilot
SELF = Path(__file__).resolve()
original_training = pilot.Backends.training_plan_check


def training(self):
    rpc = pilot.parent.rpc
    warm = Path(self.reg['warmup_trial'])
    assert (warm/'raw_grasps.npy').is_file()
    def complete(address, message, **kwargs):
        directory = Path(message['directory'])
        assert self.campaign/'startup_validation' in directory.parents
        shutil.copy2(warm/'raw_grasps.npy', directory/'raw_grasps.npy')
        pilot.common.save(directory/'bootstrap_fixture.json', {'training_only': True,
            'raw_grasps_source': str(warm/'raw_grasps.npy'),
            'raw_grasps_sha256': pilot.common.sha(warm/'raw_grasps.npy'),
            'physical_executions': 0, 'test_feedback_added': 0})
        return rpc(address, message, **kwargs)
    with patch.object(pilot.parent, 'rpc', complete):
        return original_training(self)


def seal(root):
    assert pilot.common.read(root/'progress.json')['phase'] == 'prepared'
    assert not (root/pilot.EPOCH/'started.json').exists()
    directories = [root, *[root/n for n in pilot.GROUPS], *[root/n/'fixed_graspcl' for n in pilot.GROUPS]]
    evidence = {}
    for i, directory in enumerate(directories):
        path = directory/'registration.json';evidence[str(path)] = pilot.common.sha(path)
        shutil.copy2(path, root/pilot.EPOCH/('registration_before_fixture_'+str(i)+'.json'))
        reg = pilot.common.read(path)
        assert (Path(reg['warmup_trial'])/'raw_grasps.npy').is_file()
        reg['source_hashes'][str(SELF)] = pilot.common.sha(SELF)
        reg.update(actual_run_entry=str(SELF), training_rpc_fixture_schema_checked=True)
        pilot.common.save(path, reg)
    receipt = pilot.common.read(root/pilot.EPOCH/'registration.json')
    receipt['root_registration_sha256'] = pilot.common.sha(root/'registration.json')
    pilot.common.save(root/pilot.EPOCH/'registration.json', receipt)
    pilot.common.save(root/pilot.EPOCH/'fixture_setup_revision.json', {
        'before_registration_sha256': evidence, 'source': str(SELF), 'source_sha256': pilot.common.sha(SELF),
        'models_loaded': 0, 'actual_claims': 0, 'algorithm_changes': 0,
        'change': 'Supply the original raw candidate array required by the generic planning RPC for its training-only fixture.'})
    shutil.copy2(SELF, root/'source'/SELF.name)
    pilot.parent.guard(pilot.common.read(root/'registration.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['seal', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args();root = args.output.resolve()
    if args.stage == 'seal':
        seal(root)
    else:
        pilot.BACKEND_GPU = pilot.common.read(root/'registration.json')['runtime_backend_gpu']
        pilot.Backends.training_plan_check = training
        pilot.run(root)
