"""Place the registered 200-scene run's resident inference and planning on separate GPUs."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import arx_graspcl_mpc_200_bench as full

SELF = Path(__file__).resolve()
configure_benchmark = full.wrist.benchmark.configure


def configure(registration):
    base = configure_benchmark(registration)
    original_backends = base.Backends
    original_campaign = base.Campaign
    neural_gpu = registration['runtime_neural_gpu']

    class SplitBackends(original_backends):
        def start(self):
            popen = base.core.subprocess.Popen

            def launch(command, *args, **kwargs):
                if '--kind' in command and command[command.index('--kind') + 1] == 'zerograsp':
                    kwargs['env'] = dict(kwargs['env'], CUDA_VISIBLE_DEVICES=neural_gpu,
                                         ARX_NEURAL_GPU=neural_gpu)
                return popen(command, *args, **kwargs)

            with patch.object(base.core.subprocess, 'Popen', launch):
                super().start()
            root = self.campaign / 'resident'
            placement = {component: full.read(root / (component + '_ready.json'))['gpu']
                         for component in ('zerograsp', 'curobo')}
            assert placement == {'zerograsp': neural_gpu, 'curobo': registration['runtime_backend_gpu']}
            full.save(root / 'gpu_placement.json', placement)

    class SplitCampaign(original_campaign):
        def resources(self, worker, gpu):
            while super().resources(worker, gpu):
                result = subprocess.check_output(['nvidia-smi', '--id=' + neural_gpu,
                    '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=15)
                free = int(result.strip())
                if free >= self.reg['gpu_backend_free_mib_floor']:
                    return True
                self.progress(worker, phase='waiting_for_resources', gpu=gpu,
                              neural_gpu=neural_gpu, neural_free_mib=free)
                self.failed.wait(15)
            return False

    base.Backends = SplitBackends
    base.Campaign = SplitCampaign
    return base


def prepare(output, neural_gpu, backend_gpu, scene_gpu):
    original = Path(full.__file__).resolve()
    with patch.object(full, 'SELF', SELF):
        full.prepare(output, backend_gpu, scene_gpu)
    shutil.copy2(original, output / 'source' / original.name)
    for path in (output / 'registration.json', output / full.ARM / 'registration.json'):
        reg = full.read(path)
        reg['source_hashes'][str(original)] = full.sha(original)
        reg['runtime_neural_gpu'] = neural_gpu
        full.save(path, reg)


def main():
    if sys.argv[1] == 'prepare':
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--output', type=Path, required=True)
        parser.add_argument('--neural-gpu', required=True)
        parser.add_argument('--backend-gpu', required=True)
        parser.add_argument('--scene-gpu', required=True)
        args = parser.parse_args(sys.argv[2:])
        prepare(args.output.resolve(), args.neural_gpu, args.backend_gpu, args.scene_gpu)
        return
    with patch.object(full, 'SELF', SELF), patch.object(full.wrist.benchmark, 'configure', configure):
        full.main()


if __name__ == '__main__':
    main()
