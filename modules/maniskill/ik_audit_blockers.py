"""Attribute saved model rejections without executing or changing baseline data."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch

from graspcl_pipeline import read,save,sha
from ik_workspace_audit import collision_details,box_penetration
from resident_planner import collision_scene_config


def inspect(pool,root):
    root=Path(root);source=Path(read(root / 'registration.json')['source_campaign'])
    save(root / 'blockers_job_registration.json',{'source_sha256':sha(Path(__file__)),
        'physical_attempts':0,'scope':'Attribution only: original saved poses/configuration; per-mesh collision isolation and existing 8 mm nominal-closed table clearance compared geometrically.'})
    results={}
    for profile in ['zerograsp_200','graspcl_200']:
        other=[];initial=[];closed_extra=[]
        for path in sorted((root / 'cases' / profile).glob('*/audit.json')):
            row=read(path);trial=Path(row['source_trial'])
            first_failed=row['original_replay']['rejected'].get('grasp_ik',0)==row['candidate_count']
            other_case=first_failed and row['any_pure_ik'] and row['any_table_clear_pure_solution'] and not row['any_full_scene_clear_pure_solution']
            extra=row['all_found_open_closed_solutions_table_blocked'] and not row['all_found_ik_solutions_table_blocked']
            if not other_case and not extra:continue
            state=read(trial / 'state.json');context=pool.bind(root / 'blocker_models' / profile / trial.name,state)
            data=np.load(path.parent / 'solutions.npz')
            mask=data['full_pose_success'] & (data['open_table_clear'] if other_case else data['open_world_clear'])
            q=torch.as_tensor(data['full_pose_q'][mask],device='cuda')
            if other_case:
                info=collision_details(context,state,q)
                scene=read(context.out / 'scene.json');objects={};union=torch.zeros(len(q),device='cuda',dtype=torch.bool)
                for name,geometry in scene.get('mesh',{}).items():
                    context.checker.clear_scene_cache();context.checker.update_world(collision_scene_config({'mesh':{name:geometry}}))
                    hit=~context.checker.validate(q[:,None]).reshape(-1)
                    if hit.any():objects[name]={'model_id':state['scene_meshes'][name]['model_id'],'solutions_rejected':int(hit.sum())}
                    union|=hit
                other.append({'id':trial.name,'tested_table_clear_solutions':len(q),'ground_collision_solutions':int((info['ground_penetration']>0).sum()),
                    'joint_bound_invalid_solutions':int((~info['joint_bounds_valid']).sum()),
                    'mesh_collision_solutions':int(union.sum()),'all_covered_by_mesh_collision':bool(union.all()),'objects':objects})
            if extra:
                if len(q):
                    info=collision_details(context,state,q,context.closed_checker,raise_z=.008)
                    actual=box_penetration(info['spheres'],context.base,state['scene_cuboids']['table']).max(-1).values
                    closed_extra.append({'id':trial.name,'open_world_clear_solutions':len(q),
                        'closed_clear_against_actual_table_solutions':int((actual<=0).sum()),
                        'closed_clear_against_8mm_clearance_solutions':int((info['table_penetration']<=0).sum()),
                        'has_margin_only_table_rejection':bool((actual<=0).any()),
                        'min_nominal_closed_penetration_actual_table_m':float(actual.min())})
        for claim in (source / profile / 'claims').glob('*.json'):
            trial=source / profile / 'trials' / claim.stem;result=read(trial / 'result.json')
            classified=(trial / 'constraint_rejection_classification.json').exists()
            if result['physical_attempt'] or not (classified or result['reason']=='measured_start_state_in_collision'):continue
            state=read(trial / 'state.json')
            try:context=pool.bind(root / 'initial_state_models' / profile / trial.name,state,include_target=True)
            except AssertionError as error:
                if str(error)!='Measured state collides in unchanged ARX model':raise
                context=pool.contexts['open']
            q=context.measured(state)[None];info=collision_details(context,state,q)
            scene=read(context.out / 'scene.json');objects=[]
            for name,geometry in scene.get('mesh',{}).items():
                context.checker.clear_scene_cache();context.checker.update_world(collision_scene_config({'mesh':{name:geometry}}))
                if not bool(context.checker.validate(q[:,None]).all()):objects.append({'name':name,'model_id':state['scene_meshes'][name]['model_id']})
            initial.append({'id':trial.name,'table_collision':bool((info['table_penetration']>0).any()),
                'ground_collision':bool((info['ground_penetration']>0).any()),'mesh_collisions':objects})
        results[profile]={'other_constraint_cases':other,'initial_state_collision_cases':initial,'closed_extra_cases':closed_extra,
            'other_cases_count':len(other),'initial_table_collision_cases':sum(r['table_collision'] for r in initial),
            'closed_extra_margin_only_cases':sum(r.get('has_margin_only_table_rejection',False) for r in closed_extra)}
        save(root / 'blocker_attribution.json',results)
    print('BLOCKER ATTRIBUTION COMPLETE', {k:{'other':v['other_cases_count'],'initial_table':v['initial_table_collision_cases'],
        'closed_extra_margin_only':v['closed_extra_margin_only_cases']} for k,v in results.items()},flush=True)
