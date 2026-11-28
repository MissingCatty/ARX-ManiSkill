"""Register fixed slots after shared GPU geometry was accepted and stopped."""
import argparse
from pathlib import Path
import shutil

from arx_project.layout import ROOT,ensure_run_path,now,read_json,sha256,write_json
from .prepare import prepare as base_prepare

CONFIG=ROOT/'configs/experiments/scene_reuse_slots.json'


def prepare(root):
    previous=ROOT/'tmp/state_updates/scene_reuse_gpu5_20261128_01'; accepted=read_json(previous/'acceptance.json')
    if not accepted['optional_implementation_accepted'] or not read_json(previous/'services_stopped.json')['all_owned_exited']:
        raise RuntimeError('GPU sharing is not accepted and stopped')
    base_prepare(root); reg=read_json(root/'registration.json'); cfg=read_json(CONFIG)
    reg.update(config=cfg,experiment='V8_fixed_scene_slots',preceding_GPU_sharing_adopted=True)
    for name in ['registration.json','acceptance.json','analysis.json','independent_review.json','services_stopped.json']:
        p=previous/name; reg['input_sha256'][str(p)]=sha256(p)
    dest=root/'source_registered'/CONFIG.relative_to(ROOT); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(CONFIG,dest)
    reg['source_sha256'][str(CONFIG)]=sha256(CONFIG); write_json(root/'registration.json',reg)
    write_json(ROOT/'tmp/state_updates/scene_reuse_active.json',{'at':now(),'run':str(root.relative_to(ROOT)),
        'stage':'fixed_obstacle_slots','phase':'registered'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    prepare(ensure_run_path(parser.parse_args().root))
