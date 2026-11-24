"""Native GraspGenX candidates with the approved ARX conditioning frame."""
from pathlib import Path
import sys
import time
from arx_project.layout import ROOT, module_dir, weights_dir, read_json as read, write_json as save, sha256 as sha
class GraspGenX:
    def __init__(self, args):
        global np, Image
        import numpy as np
        from PIL import Image
        sys.path.insert(0, str(module_dir('graspgenx')))
        import torch
        from graspgenx.grasp_server import GraspGenXSampler
        from graspgenx.utils.checkpoint_io import load_model_cfg
        from graspgenx.samplers.planner import run_planner_on_object
        from arx_frames import VERSION, T_EEF_GENERATOR, TCP_M
        desc=read(args.descriptor)
        if desc['repair_contract']!=VERSION or not np.array_equal(desc['T_eef_generator'],T_EEF_GENERATOR) or desc['tcp_m']!=TCP_M:
            raise ValueError('GraspGenX descriptor does not match current ARX frame')
        self.descriptor_path=args.descriptor
        self.torch, self.run_planner = torch, run_planner_on_object
        checkpoint = weights_dir('graspgenx')
        cfg = load_model_cfg(str(checkpoint / 'gen'), str(checkpoint / 'dis'))
        cfg.diffusion.ptv3vanilla.enable_flash = False
        cfg.discriminator.ptv3vanilla.enable_flash = False
        self.sampler = GraspGenXSampler.from_sweep_volume(cfg, read(args.descriptor)['parameters'])

    def request(self, directory, seed=20261122, **unused):
        points = np.load(directory / 'completed_points_world.npy', allow_pickle=False)
        assert np.isfinite(points).all() and len(points) >= 20
        _, indices = np.unique(np.floor(points / .002).astype(np.int64), axis=0, return_index=True)
        points = points[np.sort(indices)]
        if len(points) > 3500: points = points[np.linspace(0, len(points) - 1, 3500, dtype=int)]
        self.torch.manual_seed(seed); np.random.seed(seed % 2**32)
        self.torch.cuda.synchronize(); start = time.perf_counter()
        grasps, scores, tags, obb = self.run_planner(points.astype(np.float32), self.sampler,
            planner='graspmoe', grasp_threshold=-1., num_grasps=512,
            topk_num_grasps=-1, moe_obb_density='dense')
        self.torch.cuda.synchronize()
        assert grasps.shape == (len(scores), 4, 4) and len(tags) == len(scores)
        assert np.isfinite(grasps).all() and np.isfinite(scores).all()
        output = directory / 'graspgenx_raw'; output.mkdir()
        np.save(output / 'model_input_points_world.npy', points, allow_pickle=False)
        np.save(output / 'grasps_generator_world.npy', grasps, allow_pickle=False)
        np.save(output / 'scores.npy', scores, allow_pickle=False)
        result = {'success': True, 'count': len(scores), 'branch_tags': tags,
            'planner': 'graspmoe', 'num_diffusion': tags.count('diff'), 'num_obb': tags.count('obb'),
            'obb': obb, 'seed': seed, 'model_load_count': 1, 'point_count': len(points),
            'input': 'RaySt3R completion from fresh head RGB-D and SAM3 mask',
            'ground_truth_used': False, 'posthoc_topdown_filter': False,
            'wall_seconds': time.perf_counter() - start}
        from arx_frames import VERSION, T_EEF_GENERATOR, ORIGIN_BACK_M, TCP_M
        result.update(repair_contract=VERSION,T_eef_generator=T_EEF_GENERATOR.tolist(),
            model_origin_behind_eef_m=ORIGIN_BACK_M,physical_tcp_m=TCP_M,
            generator_coordinate_frame='virtual_gripper_origin_100mm_behind_EEF',
            descriptor_sha256=sha(self.descriptor_path))
        save(output / 'inference.json', result)
        return result


def create(args):return GraspGenX(args)
