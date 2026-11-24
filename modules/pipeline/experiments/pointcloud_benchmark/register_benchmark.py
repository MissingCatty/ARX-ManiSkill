"""Register fixed observations, evaluation rules and retained model budgets."""
import json
from pathlib import Path
import subprocess
import time
from prepare_scenes import RUN,PROJECT,sha,write
from model_adapters import SUPPORTED_METHODS

METHODS=list(SUPPORTED_METHODS)


def main():
    target=RUN/'registration.json'
    if target.exists():raise RuntimeError('Registration already exists; retain it unchanged')
    inputs=json.loads((RUN/'input_freeze.json').read_text())
    for p,digest in inputs['files'].items():assert sha(p)==digest,p
    scenes=json.loads((RUN/'scene_selection.json').read_text())
    preflights={}
    for method in METHODS:
        path=RUN/'preparation/model_warmups'/method/'receipt.json'
        d=json.loads(path.read_text());assert d['success'];preflights[method]={'receipt':str(path),'sha256':sha(path)}
    frozen=dict(inputs['files'])
    for p in Path(__file__).parent.glob('*.py'):
        if p.name not in ['summarize_results.py','visualize_results.py','publish_progress.py']:
            frozen[str(p)]=sha(p)
    for p in (RUN/'input_freeze.json',RUN/'scene_selection.json',PROJECT/'configs/camera_profiles.json',PROJECT/'configs/head_camera_mount.json'):
        frozen[str(p)]=sha(p)
    sources={}
    for repo in ['rayst3r','stable-point-aware-3d','dinov2']:
        folder=PROJECT/'third_party'/repo
        revision=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
        diff=subprocess.check_output(['git','-C',str(folder),'diff'],text=True)
        diff_file=RUN/'source_snapshots'/(repo+'.patch');diff_file.parent.mkdir(exist_ok=True);diff_file.write_text(diff)
        sources[repo]={'revision':revision,'patch':str(diff_file),'patch_sha256':sha(diff_file)}
        for p in folder.rglob('*.py'):
            if '.git' not in p.parts and 'build' not in p.parts:frozen[str(p)]=sha(p)
    zero=PROJECT.parent/'arx_core/servers/zerograsp/vendor/zerograsp_mainline_minimal'
    for p in [zero/'demo.py',zero/'main.py',zero/'configs/demo_640x480.yaml']+list((zero/'zerograsp').rglob('*.py')):
        frozen[str(p)]=sha(p)
    payloads=[]
    for p in (PROJECT/'weights').rglob('*'):
        if p.is_file() and p.suffix in ('.pth','.safetensors') and '.cache' not in p.parts:payloads.append(p)
    payloads.append(zero/'checkpoints/zerograsp_cvpr2025/zerograsp_synthetic.ckpt')
    weights={}
    for p in payloads:
        stat=p.stat();weights[str(p)]={'sha256':sha(p),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'inode':stat.st_ino}
        print('Registered weight',p.name,stat.st_size,flush=True)
    cases=[]
    for case in scenes['cases']:
        cid=case['id'];prefixes=[str(RUN/'observations'/cid),str(RUN/'segmentation_v03'/cid)]
        cases.append({k:case[k] for k in ['id','category','category_cn','style_id','style_index']})
        cases[-1]['input_hashes']={p:h for p,h in inputs['files'].items() if any(p.startswith(prefix+'/') for prefix in prefixes)}
    record={'id':'retained_methods_50','registered_at':time.time(),'methods':METHODS,
            'method_scene_budget':len(METHODS)*len(cases),'cases':cases,'point_budget':20000,'reference_surface_points':50000,
            'segmentation_directory':'segmentation_v03','input_freeze_sha256':sha(RUN/'input_freeze.json'),
            'frozen_files':frozen,'weights':weights,'sources':sources,'preflights':preflights,
            'resources':{'gpu_uuid':'GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b','gpu_model':'RTX 4090','physical_gpu_index_at_registration':3},
            'modality':{'zerograsp':'RGBD','rayst3r':'RGBD','spar3d':'RGB'},
            'evaluation':{'RGBD_placement':'native metric optical camera coordinates',
                          'RGB_placement':'observed masked depth only; 24 proper-rotation starts, 15 ICP iterations, 85% trimmed, scale bounded to 0.5–2 times observed extent initialization',
                          'shape_score':'separate GT-aligned normalized shape score, never used to place metric predictions',
                          'distances':'unsquared nearest-point distances; complete surface sampled uniformly',
                          'failures':'empty or invalid reconstructions retained; unexecuted/resource failures counted separately'},
            'warmup':'one excluded shared plate observation; no-surface outcome retained',
            'no_automatic_formal_retries':True,'physical_trials':0,'planning_trials':0,'training':False}
    write(target,record);print(target)


if __name__=='__main__':main()
