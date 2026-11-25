"""Resident ZeroGrasp and the baseline's exact native dual-arm planner binding."""
import os
import subprocess
import sys

from arx_project.layout import ROOT, write_json
from arx_project.registry import module_records
from modules.pipeline.experiments.dual_arm_grasp.run import Services as BaseServices, gpu_inventory


class Services(BaseServices):
    def start(self):
        from arx_service_readiness import wait_ready
        for kind in self.components:
            module_kind = self.bindings[kind]
            folder = self.root/'resident'/kind
            folder.mkdir()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.reg['gpu_roles'][kind],
                ARX_PLANNER_GPU=self.reg['gpu_roles']['curobo_dual_arm'],
                ARX_SIM_GPU=self.reg['gpu_roles']['simulation'],
                LILO_GPU=self.reg['gpu_roles']['curobo_dual_arm'],
                GRASP_TRAJOPT_POLICY='interactive', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                CUBLAS_WORKSPACE_CONFIG=':4096:8')
            env.pop('PYTHONPATH', None)
            python = ROOT/module_records()[module_kind]['environment']/'bin/python'
            entrypoint = (['-m', 'modules.pipeline.experiments.sam3_zerograsp_comparison.service']
                if kind == 'zerograsp' else ['-m', 'arx_project', 'serve'])
            command = ['bash', '-c', 'source "$1" || exit; shift; exec "$@"', 'v6-worker',
                str(ROOT/'modules/arx_core/scripts/curobo_arx/environment.sh'), str(python),
                *entrypoint, '--kind', module_kind, '--root', str(folder), '--socket', str(self.sockets[kind]),
                '--warmup-trial', str(self.root/'preflight/head_warmup'),
                '--descriptor', str(self.root/'gripper_descriptor.json')]
            stream = (folder/'service.log').open('x')
            self.streams.append(stream)
            child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            self.children[kind] = child
            wait_ready(self.sockets[kind], folder/(module_kind+'_ready.json'), child, module_kind, timeout=1800)
            print('READY_V6', kind, flush=True)
        free = gpu_inventory()
        write_json(self.root/'preflight/warm_resources.json', {'free_mib': free,
            'minimum_remaining_mib': 1536, 'bindings': self.bindings})
        if any(free.get(gpu, 0) < 1536 for gpu in set(self.reg['gpu_roles'].values())):
            raise RuntimeError('Insufficient GPU memory after V6 warmup')
