"""The current official-vendored ZeroGrasp inference with a resident model."""
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from scipy import ndimage
from .common import ROOT, digest, read, sha, write
from .geometry import grasp_pool


class ZeroGrasp:
    def __init__(self, registration):
        self.registration = registration
        package = ROOT/'servers/zerograsp/vendor/zerograsp_mainline_minimal'
        sys.path.insert(0,str(package))
        os.chdir(package)
        import demo
        self.demo, self.package, self.models = demo,package,{}
        original = demo.BaseTrainer.load_from_checkpoint
        def cached(checkpoint, **kwargs):
            key = (str(checkpoint),kwargs['config'].img_height,kwargs['config'].img_width)
            if key not in self.models:
                self.models[key] = original(checkpoint,**kwargs)
            return self.models[key]
        demo.BaseTrainer.load_from_checkpoint = staticmethod(cached)

    def handle(self, request):
        directory = Path(request['directory'])
        row = read(directory/'head_observation.json')
        asset = request['asset']
        if row['object_id'] != asset['object_id']:
            raise ValueError('Perception object identity mismatch')
        rgb = np.array(Image.open(directory/'head.png').convert('RGB'))
        depth = np.load(directory/'head_depth.npy')
        cam = np.asarray(row['camera_world_optical'])
        obj = np.asarray(row['snapshot']['objects']['test_object']['world'])
        K = np.asarray(row['intrinsics'])
        yy,xx = np.indices(depth.shape)
        points = np.stack([(xx-K[0,2])/K[0,0]*depth,(yy-K[1,2])/K[1,1]*depth,depth,np.ones_like(depth)],axis=-1)
        local = points @ (np.linalg.inv(obj) @ cam).T
        size = np.asarray(asset['size_m'])
        mask = (np.abs(local[:,:,:3])<=size/2+.004).all(axis=2) & (local[:,:,2]>-size[2]/2+.004) & np.isfinite(depth) & (depth>0)
        center = np.linalg.inv(cam) @ obj[:,3]
        if center[2] <= 0:
            return {'rejected':'target_behind_head_camera','candidates':[]}
        uv = (K @ center[:3])[:2]/center[2]
        labels,n = ndimage.label(mask)
        options = []
        for index in range(1,n+1):
            ys,xs = np.where(labels==index)
            if len(xs)>50:
                options.append((float(np.linalg.norm(np.array([xs.mean(),ys.mean()])-uv)),index))
        if not options or min(options)[0]>=30:
            return {'rejected':'no_registered_target_depth_mask','candidates':[]}
        mask = (labels==min(options)[1]).astype(np.uint8)
        Image.fromarray(mask).save(directory/'mask.png')
        Image.fromarray(np.nan_to_num(np.clip(depth*1000,0,65535)).astype(np.uint16)).save(directory/'depth.png')
        h,w = depth.shape
        camera = {'width':w,'height':h,'depth_scale':.001,'intrinsics':{'fx':float(K[0,0]),'fy':float(K[1,1]),'ppx':float(K[0,2]),'ppy':float(K[1,2]),'width':w,'height':h}}
        write(directory/'params.json',camera)
        # Reuse current actual input preparation (depth units, camera JSON, RGB), not a surrogate.
        sys.path.insert(0,str(ROOT/'shared'))
        sys.path.insert(0,str(ROOT/'servers/zerograsp'))
        from coolas_zerograsp.grasp import prepare_zerograsp_input
        from arx_shared.schemas import ZeroGraspInputRequest
        Image.fromarray(rgb).save(directory/'color.png')
        prepared = prepare_zerograsp_input(ZeroGraspInputRequest(camera_dir=str(directory),mask_path=str(directory/'mask.png'),output_dir=str(directory/'zg_input'),label=asset['object_id']))
        output = directory/'network'
        argv = sys.argv
        try:
            sys.argv = ['demo.py','--img_path',prepared.rgb_path,'--depth_path',prepared.depth_path,
                '--mask_path',prepared.mask_path,'--camera_info_path',prepared.camera_info_path,
                '--output_dir',str(output),'--checkpoint',self.registration['zerograsp_checkpoint'],
                '--config',str(self.package/'configs/demo_640x480.yaml')]
            self.demo.main()
        finally:
            sys.argv = argv
        paths = sorted(output.glob('*.raw_grasps.npy'))
        if not paths:
            return {'rejected':'no_raw_network_grasps','candidates':[]}
        rows = np.concatenate([np.load(path) for path in paths])
        np.save(directory/'raw_grasps.npy',rows)
        pool = grasp_pool(rows,cam,obj,size,self.registration['quotas']['candidate_reserve'])
        pool.update(cache_key=digest({'asset':asset['asset_sha256'],'placement':asset['placement'],
            'head_pixels':sha(directory/'head.png'),'head_depth':sha(directory/'head_depth.npy'),
            'head_camera':cam.tolist(),'intrinsics':K.tolist(),'checkpoint':self.registration['zerograsp_checkpoint_sha256'],
            'source_identity':self.registration['source_identity']}),models_loaded=len(self.models),
            observation=str(directory/'head_observation.json'),mask_method='Historical target bounds/projection isolation; real rendered RGB-D network input')
        write(directory/'pool.json',pool)
        return pool
