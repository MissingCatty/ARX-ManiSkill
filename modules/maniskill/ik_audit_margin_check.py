"""Read-only attribution of the pre-existing nominal-closed 8 mm table offset."""
from pathlib import Path
import numpy as np
import torch

from graspcl_pipeline import read,save,sha
from resident_planner import collision_scene_config


def run(pool,root):
    root=Path(root);previous=read(root / 'blocker_attribution.json');output={}
    for profile,group in previous.items():
        rows=[]
        for entry in group['closed_extra_cases']:
            path=root / 'cases' / profile / entry['id'];audit=read(path / 'audit.json')
            source=Path(audit['source_trial']);state=read(source / 'state.json')
            context=pool.bind(root / 'margin_countercheck_models' / profile / entry['id'],state)
            data=np.load(path / 'solutions.npz');mask=data['full_pose_success']&data['open_world_clear']
            q=torch.as_tensor(data['full_pose_q'][mask],device='cuda')
            if not len(q):
                rows.append({'id':entry['id'],'tested_solutions':0,
                    'margin_excludes_all_found_open_clear_branches':False,
                    'note':'Already blocked by other meshes at the open-gripper gate; do not count again as a table-caused failure.'})
                continue
            original=context.closed_checker.validate(q[:,None]).reshape(-1)
            scene=read(context.out / 'scene.json')
            context.closed_checker.clear_scene_cache();context.closed_checker.update_world(collision_scene_config(scene))
            physical_table=context.closed_checker.validate(q[:,None]).reshape(-1)
            rows.append({'id':entry['id'],'tested_solutions':len(q),
                'original_nominal_closed_world_valid':int(original.sum()),
                'same_closed_model_actual_table_world_valid':int(physical_table.sum()),
                'margin_excludes_all_found_open_clear_branches':bool(not original.any() and physical_table.any())})
        output[profile]={'cases':rows,'case_count':sum(r['margin_excludes_all_found_open_clear_branches'] for r in rows)}
    save(root / 'closed_margin_fullworld_countercheck.json',{'source_sha256':sha(Path(__file__)),
        'physical_attempts':0,'baseline_or_model_files_modified':False,
        'scope':'Analytical attribution only. Same nominal-closed robot and all object meshes, actual table surface vs existing +8 mm collision clearance. Does not imply pregrasp/lift/path or physical grasp would succeed.',
        'results':output})
    print('MARGIN COUNTERCHECK',{k:v['case_count'] for k,v in output.items()},flush=True)
