"""Interface-only warmup on an observation excluded from the 50 formal cases."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from prepare_scenes import RUN,write
from model_adapters import load_adapter, SUPPORTED_METHODS


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',required=True,choices=SUPPORTED_METHODS);args=parser.parse_args()
    output=RUN/'preparation/model_warmups'/args.method;output.mkdir(parents=True,exist_ok=True)
    source=RUN/'observations/warmup';seg=RUN/'segmentation_v02/warmup'
    if not (seg/'mask.png').exists():seg=RUN/'segmentation/warmup'
    rgb=np.array(Image.open(source/'rgb.png').convert('RGB'));depth=np.load(source/'depth_m.npy');mask=np.array(Image.open(seg/'mask.png'))>0
    k=np.array(json.loads((source/'camera.json').read_text())['intrinsics'])
    torch.set_num_threads(6);random.seed(12345);np.random.seed(12345);torch.manual_seed(12345)
    t=time.perf_counter();model=load_adapter(args.method);torch.cuda.synchronize();load_s=time.perf_counter()-t
    torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
    result=model.predict(rgb,depth,mask,k,output/'native_input');torch.cuda.synchronize();seconds=time.perf_counter()-t
    if 'mesh' in result and result['mesh'] is None:
        write(output/'failure.json',{'method':args.method,'reason':'native_generator_returned_no_surface','load_s':load_s,'warmup_s':seconds,
                                    'formal_model_trials':0,'physical_trials':0})
        raise RuntimeError('Native generator returned no isosurface')
    if 'mesh' in result:
        result['mesh'].export(output/'mesh.ply');count=len(result['mesh'].vertices)
    else:
        np.save(output/'points.npy',result['points']);count=len(result['points'])
    assert count>0,'Empty warmup reconstruction'
    record={'method':args.method,'success':True,'load_s':load_s,'warmup_s':seconds,'output_vertices_or_points':count,
            'peak_gpu_gb':torch.cuda.max_memory_allocated()/1e9,'configuration':model.configuration,
            'formal_model_trials':0,'physical_trials':0}
    write(output/'receipt.json',record);print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
