"""Direct RGB-D geometry reference and independent SAM3 mask evaluation."""
import json
import time
import numpy as np
from PIL import Image
from prepare_scenes import RUN,write
from geometry_evaluation import unproject,subsample,surface_metrics
from run_method import reference_data


def main():
    freeze=json.loads((RUN/'input_freeze.json').read_text())
    cases=json.loads((RUN/'scene_selection.json').read_text())['cases']
    results=[]
    for case in cases:
        cid=case['id'];source=RUN/'observations'/cid;seg=RUN/freeze['segmentation_directory']/cid;truth=RUN/'ground_truth'/cid
        camera=json.loads((source/'camera.json').read_text());k=np.array(camera['intrinsics'])
        depth=np.load(source/'depth_m.npy');mask=np.asarray(Image.open(seg/'mask.png'))>0
        gt=np.asarray(Image.open(truth/'mask.png'))>0
        start=time.perf_counter();points=subsample(unproject(depth,mask,k),20000);generation_s=time.perf_counter()-start
        reference=reference_data(cid,camera);visible=unproject(depth,gt,k)
        out=RUN/'baseline'/cid;out.mkdir(parents=True,exist_ok=True);np.save(out/'metric_points.npy',points)
        entry={'id':cid,'category':case['category'],'style_id':case['style_id'],'generation_s':generation_s,
               'point_count':len(points),'geometry':surface_metrics(points,reference,visible),
               'mask_iou':float((mask&gt).sum()/(mask|gt).sum()),
               'mask_visible_target_recall':float((mask&gt).sum()/gt.sum()),'model_inferences':0,'physical_trials':0}
        write(out/'result.json',entry);results.append(entry)
    write(RUN/'baseline_complete.json',{'results':results,'formal_scenes':50,'model_inferences':0,'physical_trials':0})
    print('Baseline complete; median mask IoU',float(np.median([x['mask_iou'] for x in results])))


if __name__=='__main__':main()
