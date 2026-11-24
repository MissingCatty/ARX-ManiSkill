"""Choose the pilot's actual planner GPU before freezing its registration."""
import argparse
from pathlib import Path
import shutil
import subprocess

import arx_direct_pilot_ready_20261120 as ready

SELF = Path(__file__).resolve()
pilot = ready.pilot


def prepare(root, gpu):
    available = int(subprocess.check_output(['nvidia-smi', '--id='+gpu,
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    assert available >= 10000, ('Insufficient space before preloading pilot components', available)
    pilot.BACKEND_GPU = gpu
    ready.prepare(root)
    for directory in [root, *[root/n for n in pilot.GROUPS], *[root/n/'fixed_graspcl' for n in pilot.GROUPS]]:
        path = directory/'registration.json';reg = pilot.common.read(path)
        reg['source_hashes'][str(SELF)] = pilot.common.sha(SELF)
        reg.update(actual_run_entry=str(SELF), runtime_backend_gpu=gpu,
                   initial_backend_free_mib=available,
                   prior_prestart_output=str(pilot.CODE/'outputs/arx_direct_grasp_pilot_20261120_02'),
                   prior_prestart_actual_claims=0,
                   placement_reason='GPU0 transient external allocation exhausted memory before readiness; use GPU2 without altering the algorithm.')
        pilot.common.save(path, reg)
    path = root/pilot.EPOCH/'registration.json';r = pilot.common.read(path)
    r['root_registration_sha256'] = pilot.common.sha(root/'registration.json')
    pilot.common.save(path, r)
    shutil.copy2(SELF, root/'source'/SELF.name)
    pilot.parent.guard(pilot.common.read(root/'registration.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend-gpu')
    args = parser.parse_args();root = args.output.resolve()
    if args.stage == 'prepare':
        assert args.backend_gpu
        prepare(root, args.backend_gpu)
    else:
        pilot.BACKEND_GPU = pilot.common.read(root/'registration.json')['runtime_backend_gpu']
        pilot.run(root)
