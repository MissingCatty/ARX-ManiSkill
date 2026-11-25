"""Resident workers, with unchanged per-component admission floors."""
import os
import subprocess
import time

from arx_project.layout import ROOT,write_json,now
from arx_project.registry import module_records
from modules.pipeline.experiments.sam3_zerograsp_comparison.services import Services as OriginalServices
from modules.pipeline.experiments.dual_arm_grasp.run import gpu_inventory


class Services(OriginalServices):
    def start(self):
        from arx_service_readiness import wait_ready
        floors=self.reg['configuration']['resource_floors_mib'];startup=[]
        for kind in self.components:
            module_kind=self.bindings[kind];gpu=self.reg['gpu_roles'][kind]
            minimum=floors['curobo_before_loading'] if kind=='curobo_dual_arm' else floors.get(kind+'_before_loading',floors.get(kind))
            if gpu_inventory()[gpu] < minimum:raise RuntimeError('Registered startup floor not met for '+kind)
            folder=self.root/'resident'/kind;folder.mkdir()
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,ARX_PLANNER_GPU=self.reg['gpu_roles']['curobo_dual_arm'],
                ARX_SIM_GPU=self.reg['gpu_roles']['simulation'],LILO_GPU=self.reg['gpu_roles']['curobo_dual_arm'],
                GRASP_TRAJOPT_POLICY='interactive',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            env.pop('PYTHONPATH',None)
            python=ROOT/module_records()[module_kind]['environment']/'bin/python'
            if kind == 'zerograsp':
                entry=['-m','modules.pipeline.experiments.sam3_zerograsp_comparison.service']
            elif kind == 'curobo_dual_arm' and self.reg.get('bounded_scene_mesh_cache'):
                entry=['-m','modules.pipeline.experiments.sam3_zerograsp_comparison.io_parallel.planner_service']
            else:
                entry=['-m','arx_project','serve']
            command=['bash','-c','source "$1" || exit; shift; exec "$@"','io74-worker',
                str(ROOT/'modules/arx_core/scripts/curobo_arx/environment.sh'),str(python),*entry,
                '--kind',module_kind,'--root',str(folder),'--socket',str(self.sockets[kind]),
                '--warmup-trial',str(self.root/'preflight/head_warmup'),'--descriptor',str(self.root/'gripper_descriptor.json')]
            stream=(folder/'service.log').open('x');self.streams.append(stream);begun=time.perf_counter()
            child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT);self.children[kind]=child
            wait_ready(self.sockets[kind],folder/(module_kind+'_ready.json'),child,module_kind,timeout=1800)
            startup.append({'kind':kind,'gpu':gpu,'seconds':time.perf_counter()-begun,'pid':child.pid})
            write_json(self.root/'preflight/startup.json',{'at':now(),'components':startup})
            print('READY_IO74',kind,flush=True)
