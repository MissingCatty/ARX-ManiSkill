"""Seal shared inputs before independent mask/shape evaluation."""
import json
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
from prepare_scenes import RUN,PROJECT,sha,write


def main():
    target=RUN/'input_freeze.json'
    if target.exists():
        d=json.loads(target.read_text())
        for p,h in d['files'].items():assert sha(p)==h,p
        print('Frozen inputs unchanged');return
    manifest=json.loads((RUN/'scene_selection.json').read_text())
    segmentation=json.loads((RUN/'segmentation_v03_complete.json').read_text())
    assert len(manifest['cases'])==50
    assert all(x['success'] for x in segmentation['receipts'])
    files={};checks=[]
    for case in manifest['cases']+[manifest['warmup']]:
        cid=case['id'];source=RUN/'observations'/cid;gt=RUN/'ground_truth'/cid;seg=RUN/'segmentation_v03'/cid
        # Keep object truth out of the camera record read by model adapters.
        camera=json.loads((source/'camera.json').read_text())
        if 'object_world' in camera:
            backup=RUN/'preparation/camera_metadata_before_truth_split'/f'{cid}.json';backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source/'camera.json',backup)
            write(gt/'object_pose.json',{'object_world':camera.pop('object_world')})
            write(source/'camera.json',camera)
        receipt=json.loads((seg/'receipt.json').read_text())
        assert receipt['ground_truth_used'] is False and receipt['rgb_sha256']==sha(source/'rgb.png')
        rgb=np.asarray(Image.open(source/'rgb.png'));depth=np.load(source/'depth_m.npy');mask=np.asarray(Image.open(seg/'mask.png'))>0
        assert rgb.shape==(480,640,3) and depth.shape==mask.shape==(480,640)
        assert np.isfinite(depth).all() and mask.any()
        checks.append({'id':cid,'predicted_mask_pixels':int(mask.sum()),'valid_mask_depth_pixels':int((mask&(depth>0)).sum())})
        for p in [source/'rgb.png',source/'depth_m.npy',source/'camera.json',seg/'mask.png',seg/'receipt.json',gt/'surface_world.ply',gt/'mask.png']:
            files[str(p)]=sha(p)
    write(target,{'created_at':time.time(),'segmentation_directory':'segmentation_v03','formal_scenes':50,'warmup_scenes':1,
                  'files':files,'checks':checks,'formal_model_attempts':0,'physical_trials':0,
                  'truth_boundary':'GT stored separately; adapters receive only RGB, metric depth, shared SAM3 mask and K'})
    print('Frozen 50 common-object inputs and one excluded warmup')


if __name__=='__main__':main()
