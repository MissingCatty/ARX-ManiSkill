"""SAM3 text segmentation adapter; no reconstruction or planning imports."""
from pathlib import Path
import sys
import time
from arx_project.layout import ROOT, module_dir, weights_dir, read_json as read, write_json as save, sha256 as sha
class SAM3:
    def __init__(self, args):
        global np, Image
        import numpy as np
        from PIL import Image
        sys.path.insert(0, str(module_dir('sam3')))
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        sys.path.insert(0, str(module_dir('camera')))
        from camera_imaging import camera_profiles, lens_maps, sensor_and_model_frames
        self.torch = torch
        self.profile = camera_profiles()['head']
        self.maps = lens_maps(self.profile)
        self.convert = sensor_and_model_frames
        model = build_sam3_image_model(device='cuda',
            checkpoint_path=str(weights_dir('sam3') / 'sam3.pt'),
            load_from_HF=False, enable_inst_interactivity=False)
        self.processor = Sam3Processor(model, device='cuda', confidence_threshold=.5)

    def request(self, directory, **unused):
        camera = directory / 'camera'
        rgb = np.asarray(Image.open(camera / 'color.png').convert('RGB'))
        depth = np.asarray(Image.open(camera / 'depth.png'), dtype=np.float32) * read(camera / 'params.json')['depth_scale']
        frames = self.convert(rgb, depth, self.profile, self.maps)
        sensor = directory / 'raw_head'
        sensor.mkdir()
        Image.fromarray(frames['raw_rgb']).save(sensor / 'rgb.png')
        Image.fromarray(frames['raw_depth']).save(sensor / 'depth.png')
        Image.fromarray(frames['rgb']).save(camera / 'color.png')
        np.save(camera / 'depth_m.npy', frames['depth_m'], allow_pickle=False)
        Image.fromarray(np.rint(frames['depth_m'] / .0001).astype(np.uint16)).save(camera / 'depth.png')
        params = read(camera / 'params.json'); params['depth_scale'] = .0001
        params['projection'] = 'rectified_color_optical'; save(camera / 'params.json', params)
        if (camera / 'mask.png').exists():
            evaluation = directory / 'evaluation_only'
            evaluation.mkdir()
            (camera / 'mask.png').rename(evaluation / 'visible_target_gt.png')
        text = read(directory / 'entry.json')['target_text']
        start = time.perf_counter()
        with self.torch.inference_mode(), self.torch.autocast('cuda', dtype=self.torch.bfloat16):
            state = self.processor.set_image(Image.fromarray(frames['rgb']))
            state = self.processor.set_text_prompt(prompt=text, state=state)
            scores = state['scores'].detach().float().cpu().numpy()
            masks = state['masks'].detach().cpu().numpy().astype(bool).reshape(-1, 480, 640)
        self.torch.cuda.synchronize()
        index = int(np.argmax(scores)) if len(scores) else None
        mask = masks[index] if index is not None else np.zeros((480, 640), dtype=bool)
        Image.fromarray(mask.astype(np.uint8) * 255).save(camera / 'mask.png')
        rgba = np.concatenate([frames['rgb'], mask[..., None].astype(np.uint8) * 255], axis=-1)
        Image.fromarray(rgba).save(directory / 'target_rgba.png')
        overlay = frames['rgb'].copy()
        overlay[mask] = (.6 * overlay[mask] + .4 * np.array([50, 220, 80])).astype(np.uint8)
        Image.fromarray(overlay).save(directory / 'sam3_overlay.jpg', quality=88)
        result = {'success': True, 'target_found': bool(mask.any()), 'mask_pixels': int(mask.sum()),
            'prompt': text, 'scores': scores.tolist(), 'selected_index': index,
            'selection': 'highest SAM3 score for the registered text; no GT matching',
            'ground_truth_used': False, 'model_load_count': 1,
            'rgb_sha256': sha(camera / 'color.png'), 'mask_sha256': sha(camera / 'mask.png'),
            'wall_seconds': time.perf_counter() - start}
        save(directory / 'sam3.json', result)
        return result



def create(args):return SAM3(args)
