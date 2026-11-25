"""RaySt3R metric camera-point reconstruction and world transform."""
from pathlib import Path
import sys
import time
from arx_project.layout import ROOT, module_dir, weights_dir, read_json as read, write_json as save, sha256 as sha
class RaySt3R:
    def __init__(self, args):
        global np, Image
        import numpy as np
        from PIL import Image
        import torch
        sys.path.insert(0,str(module_dir('rayst3r')))
        from modules.rayst3r.arx_reconstruction import RaySt3RAdapter
        self.torch = torch
        self.model = RaySt3RAdapter()

    def request(self, directory, **unused):
        camera = directory / 'camera'
        rgb = np.asarray(Image.open(camera / 'color.png').convert('RGB'))
        depth = np.load(camera / 'depth_m.npy', allow_pickle=False)
        mask = np.asarray(Image.open(camera / 'mask.png')) > 0
        k = np.asarray(read(directory / 'state.json')['intrinsics'])
        self.torch.cuda.synchronize(); start = time.perf_counter()
        output = self.model.predict(rgb, depth, mask, k, directory / 'rayst3r_native')
        points = np.asarray(output['points'], dtype=np.float32)
        assert points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all()
        np.save(directory / 'completed_points_camera.npy', points, allow_pickle=False)
        transform = np.asarray(read(directory / 'state.json')['camera_world_optical'])
        world = points @ transform[:3, :3].T + transform[:3, 3]
        np.save(directory / 'completed_points_world.npy', world.astype(np.float32), allow_pickle=False)
        self.torch.cuda.synchronize()
        result = {'success': True, 'point_count': len(points), 'metric': True,
            'coordinate_frame': 'camera_optical_m_to_world_m', 'model_load_count': 1,
            'configuration': self.model.configuration, 'actual_novel_views': 21,
            'input_mask_sha256': sha(camera / 'mask.png'), 'wall_seconds': time.perf_counter() - start}
        save(directory / 'rayst3r.json', result)
        return result



def create(args):return RaySt3R(args)
