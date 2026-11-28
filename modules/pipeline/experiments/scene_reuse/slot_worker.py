"""Compare shared GPU geometry with and without fixed obstacle slots."""
import argparse
from pathlib import Path
import time

from arx_project.layout import now, read_json, write_json
from modules.curobo.scene_cache import mesh_stores
from modules.curobo.scene_reuse.gpu_runtime import GPUGeometryReuse
from modules.curobo.scene_reuse.slot_scenes import SlotSceneUpdates
from .worker import Worker as OriginalWorker
from .preflight import run as component_preflight
from .slot_preflight import hot_switches


class ComponentModes:
    def __init__(self,planner,legacy,slots,share_both=True):
        self.planner,self.legacy,self.slots,self.share_both=planner,legacy,slots,share_both
    def begin(self,cache):
        self.planner.pool.memory_bindings.scenes=self.slots if cache is not None else self.legacy
        self.planner._cpu_geometry.begin(None); self.planner.gpu_geometry_reuse.begin(self.share_both or cache is not None)
    def finish(self):
        result=self.planner._cpu_geometry.finish(); result['GPU_geometry']=self.planner.gpu_geometry_reuse.finish()
        self.planner.pool.memory_bindings.scenes=self.legacy
        return result


class Worker(OriginalWorker):
    def __init__(self,planner,campaign,legacy,slots):
        super().__init__(planner,campaign); self.legacy,self.slots=legacy,slots
        self.reg=read_json(Path(campaign)/'registration.json')
        self.cases={c['id']:c for c in self.reg['cases']}
    def request(self,directory,*,variant,**flags):
        if variant not in ['control','cached']: raise ValueError('Unknown slot variant')
        case=self.cases[Path(directory).parent.name]
        if case.get('forced_arm') is not None: flags['arm']=case['forced_arm']
        enabled=variant=='cached' and self.reg['config'].get('slot_updates_enabled',True)
        self.planner.pool.memory_bindings.scenes=self.slots if enabled else self.legacy
        # Both groups use the accepted GPU resource sharing.
        self.planner.gpu_geometry_reuse.begin(self.reg['config'].get('GPU_sharing_enabled_both_groups',True) or variant=='cached')
        try: result=super().request(directory,variant='control',**flags)
        finally:
            sharing=self.planner.gpu_geometry_reuse.finish()
            self.planner.pool.memory_bindings.scenes=self.legacy
            write_json(Path(directory)/'GPU_geometry.json',sharing)
        result.update(mode=variant,GPU_geometry=sharing,scene_slots_enabled=enabled)
        write_json(Path(directory)/'planning_metrics.json',result)
        return result


def main():
    parser=argparse.ArgumentParser()
    for name in ['root','socket','warmup-trial','descriptor']: parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--kind',default='curobo_dual_arm'); args=parser.parse_args()
    import torch
    from curobo.types import Pose
    from arx_project.service import serve
    from modules.curobo.scene_reuse.runtime import create
    torch.set_num_threads(4)
    if torch.cuda.mem_get_info()[0]/1024**2<10000: raise RuntimeError('Original startup reserve unavailable')
    probe=torch.arange(1024,device='cuda',dtype=torch.float32)
    if float((probe*2).sum())!=1047552.: raise RuntimeError('CUDA arithmetic failed')
    campaign=args.root.parent; reg=read_json(campaign/'registration.json'); began=time.perf_counter()
    planner=create(args); planner._cpu_geometry=planner.cpu_geometry_reuse
    planner.gpu_geometry_reuse=GPUGeometryReuse(mesh_stores(planner))
    memory=planner.pool.memory_bindings; legacy=memory.scenes
    slots=SlotSceneUpdates(memory.native.SceneCfg.create,Pose.from_list,lambda path:memory.files.digest(path))
    planner.cpu_geometry_reuse=ComponentModes(planner,legacy,slots,reg['config'].get('GPU_sharing_enabled_both_groups',True))
    pilot_ids=['M0001_S1_open_apple','M0010_S1_open_boxed_food','M0062_S4_clutter_keychain','M0066_S4_clutter_mug','M0078_S4_clutter_teapot']
    pilot=[c for c in reg['cases'] if c['id'] in pilot_ids]
    try: checked=component_preflight(planner,args.root/'slot_geometry_preflight',args.warmup_trial,pilot)
    finally: planner.cpu_geometry_reuse=planner._cpu_geometry
    hot=hot_switches(planner,pilot,args.root/'slot_switch_preflight',legacy,slots)
    write_json(args.root/'startup.json',{'at':now(),'passed':checked['passed'] and hot['passed'],
        'seconds':time.perf_counter()-began,'component_pairs':checked['pairs'],'model_loads':planner.pool.loads})
    serve(args,worker=Worker(planner,campaign,legacy,slots))


if __name__=='__main__':main()
