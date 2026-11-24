"""Freeze and run the five paired cases with enforced terminal geometry."""
import argparse
import inspect
from pathlib import Path
import shutil
import textwrap
import types

import arx_direct_pilot_fixture_20261120 as fixture
import arx_direct_terminal_20261120 as terminal

pilot = fixture.pilot
SELF = Path(__file__).resolve()
original_manager = pilot.manager_class


def manager():
    cls = original_manager()
    source = textwrap.dedent(inspect.getsource(pilot.parent.Manager.launch))
    source = source.replace('graspcl_unseen_services_quiet_20261119.py', 'arx_direct_terminal_services_20261120.py')
    ns = dict(pilot.parent.Manager.launch.__globals__)
    exec(compile(source, str(SELF)+':services', 'exec'), ns)
    cls.launch = types.FunctionType(ns['launch'].__code__, pilot.parent.Manager.launch.__globals__, 'launch')
    return cls


def prepare(root):
    fixture.placed.prepare(root, pilot.SIM_GPU)
    dependencies = [SELF, Path(fixture.__file__).resolve(), Path(terminal.__file__).resolve(),
                    SELF.with_name('arx_direct_terminal_services_20261120.py')]
    for directory in [root, *[root/n for n in pilot.GROUPS], *[root/n/'fixed_graspcl' for n in pilot.GROUPS]]:
        reg = pilot.common.read(directory/'registration.json')
        reg['source_hashes'].update({str(p): pilot.common.sha(p) for p in dependencies})
        reg.update(actual_run_entry=str(SELF), training_rpc_fixture_schema_checked=True,
            terminal_constraint_projection={'blend_radius_m': .050, 'full_constraint_radius_m': .025,
                'fixed_pregrasp_goal': False, 'dense_collision_and_derivative_checks': True,
                'reason': 'Joint-space optimization alone does not preserve the terminal approach corridor.'})
        pilot.common.save(directory/'registration.json', reg)
    receipt = pilot.common.read(root/pilot.EPOCH/'registration.json')
    receipt['root_registration_sha256'] = pilot.common.sha(root/'registration.json')
    pilot.common.save(root/pilot.EPOCH/'registration.json', receipt)
    for path in dependencies:shutil.copy2(path, root/'source'/path.name)
    pilot.parent.guard(pilot.common.read(root/'registration.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args();root = args.output.resolve()
    if args.stage == 'prepare':prepare(root)
    else:
        pilot.BACKEND_GPU = pilot.common.read(root/'registration.json')['runtime_backend_gpu']
        pilot.Backends.training_plan_check = fixture.training
        pilot.manager_class = manager
        pilot.run(root)
