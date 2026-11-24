"""Archived ARX compatibility checks and native-encoded full-stale donors."""
import argparse
import copy
from pathlib import Path

import numpy as np
from PIL import Image
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,verify_files
from training.nearfield_arx import native_mask
from training.nearfield_arx.mask_repair_data import saved_row
from .capture import validate_saved_capture
from .advisor33_geometry import observe,first_plane


def run(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    previous=ROOT/'runs/nearfield_advisor33_20261111_04'
    reg=read(previous/'registration.json');projector=native_mask.NativeProjector(reg['learned_policy']['robot_usd'])
    review=ROOT/'runs/nearfield_advisor32_20261111_01/diagnosis_complete/review.json'
    rows=[];pins={str(review):sha(review)};cache={}
    for item in read(review)['camera_views']:
        rgb=Path(item['depth']).with_suffix('.png');ep,raw=saved_row(rgb,cache)
        files=validate_saved_capture(ep,raw);pins.update({str(ep/p):h for p,h in files.items()})
        row=native_mask.bind(read(item['observation']),rgb)
        asset=next(a for a in reg['catalog']['objects'] if a['object_id']==row['object_id'])
        depth=np.load(item['depth'],allow_pickle=False)
        gate,masks,points=observe(projector,asset['usd_path'],row,depth)
        trial=read(ep.parent.parent/'trial_result.json')
        positive=trial['success'] and item['stage']=='policy_3'
        first=first_plane(row['snapshot']['link_world']['right_eef_link'],'REFINE 0 0 0 0 0 0',points,gate['box'])
        rows.append({'trial_id':item['trial_id'],'stage':item['stage'],'historical_success':trial['success'],
            'required_closure_positive':positive,'gate':gate,'first_plane':first,
            'source_observation':item['observation'],'target_points':len(points)})
        for p in (item['observation'],item['depth'],str(rgb),str(ep/'samples.jsonl')):pins[p]=sha(p)
    positives=[r for r in rows if r['required_closure_positive']]
    entries=[r for r in rows if r['stage']=='H' and r['historical_success']]
    passed=len(rows)==20 and len(positives)==4 and all(r['gate']['pass'] for r in positives) and all(r['first_plane']['pass'] for r in entries)
    result={'pass':passed,'observations':len(rows),'successful_preclose_states':len(positives),
        'false_rejections':sum(not r['gate']['pass'] for r in positives),'rows':rows,
        'physical_trials':0,'model_inferences':0,'files':pins}
    write(out/'geometry_preflight.json',result)
    if not passed:raise RuntimeError('Native mask historical compatibility failed')
    donors=copy.deepcopy(reg['full_stale_donors']);folder=out/'stale_donors';folder.mkdir()
    for obj,donor in donors.items():
        for record in donor['records']:
            rgb=Path(record['raw_rgb']);ep,raw=saved_row(rgb,cache)
            files=validate_saved_capture(ep,raw)
            row=native_mask.bind(raw,rgb);depth=np.load(record['raw_depth'],allow_pickle=False)
            masks,_=native_mask.load(row,depth)
            images=native_mask.model_views(Image.open(rgb),depth,masks,reg['learned_policy']['visual_contract']['depth_scale'])
            dest=folder/(obj+'_plane'+str(record['plane'])+'_depth.png');images[1].save(dest)
            record.update(previous_model_images=record['model_images'],model_images=[str(rgb),str(dest)],
                mask_version=native_mask.VERSION,native_mask_binding=row['native_mask_binding'])
            record['files'].update({str(ep/p):h for p,h in files.items()})
            record['files'].update({str(dest):sha(dest),str(ep/'samples.jsonl'):sha(ep/'samples.jsonl')})
            donor['images_by_plane'][str(record['plane'])]=record['model_images']
            pins.update(record['files'])
    write(folder/'manifest.json',donors);pins[str(folder/'manifest.json')]=sha(folder/'manifest.json')
    pins[str(out/'geometry_preflight.json')]=sha(out/'geometry_preflight.json')
    verify_files(pins);write(out/'verification.json',{'pass':True,'physical_trials':0,'model_inferences':0,
        'four_stale_views_reencoded_from_original_frames':True,'historical_camera_pose_unchanged':True,'files':pins})
    return {'pass':True,'historical_views':20,'preclose_positives':4,'stale_views':4}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');print(run(p.parse_args().out))
