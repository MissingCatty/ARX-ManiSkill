"""Register and compare V7 candidate filters without simulation or model calls."""
import argparse
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from PIL import Image

from arx_project.layout import ROOT,ensure_run_path,now,read_json,sha256,write_json
from modules.curobo.dual_arm.model import ArmModel
from modules.graspgenx.postprocessing.geometry import GripperGeometry,RobotPointFilter
from modules.graspgenx.postprocessing.filters import CandidatePostprocessor
from modules.graspgenx.postprocessing.nms import native_backend
from modules.graspgenx.postprocessing.native_geometry import backend as geometry_backend


def raw(path):return json.loads(Path(path).read_text())


def verify(reg):
    for name,digest in {**reg['source_sha256'],**reg['input_sha256']}.items():
        if sha256(name)!=digest:raise RuntimeError('Frozen V7 source or input changed: '+name)


def prepare(root,config_path):
    if root.exists():raise ValueError('Use a fresh offline registration')
    from modules.pipeline.experiments.current_configuration.configuration import load_current
    current=load_current()
    cfg=read_json(config_path)
    component=ROOT/cfg['offline']['component_preflight'];checked=raw(component/'acceptance.json')
    if not checked['passed']:raise ValueError('CPU geometry preflight did not pass')
    for name,digest in checked['source_sha256'].items():
        if sha256(name)!=digest:raise RuntimeError('Geometry changed after component equivalence checks')
    source=ROOT/cfg['offline']['source_run']
    if not raw(source/'independent_review.json')['passed']:raise ValueError('Original B inputs are not independently accepted')
    cases=read_json(source/'registration.json')['cases']
    if len(cases)!=69:raise ValueError('The V7 offline cohort must keep all 69 saved observations')
    robot=RobotPointFilter();grippers={s:GripperGeometry(s) for s in ['right','left']}
    inputs={};counts=[]
    for case in cases:
        directory=Path(case['source_directory'])
        files=[directory/name for name in ['state.json','camera/depth_m.npy','camera/mask.png',
            'camera/params.json','completed_points_world.npy','candidates/candidate_pool.json',
            'graspgenx_raw/grasps_generator_world.npy','graspgenx_raw/scores.npy','graspgenx_raw/inference.json',
            'sam3.json','rayst3r.json','plan_result.json','result.json']]
        inputs.update({str(p):sha256(p) for p in files})
        counts.append(len(np.load(directory/'graspgenx_raw/scores.npy',allow_pickle=False)))
    sources=[Path(__file__).resolve(),component/'acceptance.json',ROOT/'modules/curobo/dual_arm/model.py',
        ROOT/'modules/curobo/fingertip_contact/geometry.py',ROOT/'modules/graspgenx/arx_frames.py',
        ROOT/'simulation/robots/arx/finger_pad_geometry.json',ROOT/'simulation/robots/arx/collision/arx_refined_871.json',
        ROOT/'configs/policies/fingertip_table.json',ROOT/'configs/modules.json',config_path,
        ROOT/'configs/experiments/active_experiment.json',native_backend()[1],geometry_backend()[1]]
    sources+=list((ROOT/'modules/graspgenx/postprocessing').glob('*.py'))
    sources+=list((ROOT/'modules/graspgenx/postprocessing').glob('*.cpp'))
    sources+=list(Path(__file__).parent.glob('*.py'))
    sources+=list(set(robot.source_paths+[p for g in grippers.values() for p in g.source_paths]))
    sources=sorted(set(p.resolve() for p in sources))
    root.mkdir(parents=True)
    for p in sources:
        target=root/'source_registered'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    import scipy,trimesh
    write_json(root/'registration.json',{'at':now(),'version':'V7','experiment':'offline_candidate_postprocessing',
        'authorization':'2026-11-26：开始吧；先完成69份冻结观测的粗筛和去重离线对照。',
        'cases':cases,'source_run':str(source),'candidate_counts':counts,'total_candidates':sum(counts),
        'thresholds':cfg['offline']['thresholds'],'NMS_timing_repeats':3,
        'geometry_scope':'authored_ARX_wrist_and_finger_convex_parts_both_physical_arms',
        'point_collision':'at_least_1mm_strict_interior_at_pre_mid_grasp',
        'observed_table_surface_owned_by_V2_analytic_vertex_check':True,
        'occluded_and_unobserved_geometry':'unknown_then_original_planner',
        'completion_used_for':'cavity_support_diagnostic_and_separate_empty0_variant',
        'finger_opening_per_joint_m':.044,'candidate_width_m':.088,
        'candidate_pose_score_width_and_source_indices':'unchanged',
        'NMS_deferred_candidates':'retained_separately_for_later_planner_fallback',
        'geometry_rejected_raw_candidates_are_not_reinserted':True,
        'environment':'environments/geometric_ik','versions':{'numpy':np.__version__,'scipy':scipy.__version__,'trimesh':trimesh.__version__},
        'new_perception_calls':0,'new_IK_calls':0,'planning_requests':0,'simulation_instances':0,'physical_grasps':0,
        'automatic_retries':False,'current_configuration':current['collision_profile_id'],
        'source_sha256':{str(p):sha256(p) for p in sources},'input_sha256':inputs})
    write_json(root/'progress.json',{'at':now(),'phase':'registered','completed_cases':0,'total_cases':69,
        'planning_requests':0,'physical_grasps':0})


def selected_retention(result,selected,physical_success):
    index=int(selected['index']);side=selected['arm'];output={}
    for name,value in result['variants'].items():
        incoming=value['input_indices'];where={n:i for i,n in enumerate(incoming)}
        kept=index in where;representative=index in value['representatives']
        output[name]={'selected_retained_including_deferred':kept,
            'selected_is_priority_representative':representative,
            'representative_for_selected':value['representative_for_input'][where[index]] if kept else None,
            'selected_physical_arm_geometry_clear':bool(result['geometry'][side]['kept'][index]),
            'original_selection_same_arm_available':kept and (not name.startswith('geometry') or bool(
                result['geometry'][side]['kept_nonempty' if name.startswith('geometry_empty') else 'kept'][index])),
            'original_physical_grasp_success':bool(physical_success)}
    return output


def summarize(rows):
    names=list(rows[0]['variants']) if rows else [];summary={}
    for name in names:
        values=[row['variants'][name] for row in rows]
        times=[v['postprocessing_seconds'] for v in values]
        summary[name]={'cases':len(rows),'representatives_total':sum(v['representatives'] for v in values),
            'representatives_median':float(np.median([v['representatives'] for v in values])),
            'postprocessing_median_ms':float(np.median(times)*1000),
            'postprocessing_p90_ms':float(np.percentile(times,90)*1000),
            'original_selected_available':sum(v['retention']['selected_retained_including_deferred'] for v in values),
            'original_selected_is_representative':sum(v['retention']['selected_is_priority_representative'] for v in values),
            'original_selected_same_arm_available':sum(v['retention']['original_selection_same_arm_available'] for v in values),
            'original_successful_selected_available':sum(v['retention']['selected_retained_including_deferred']
                for v in values if v['retention']['original_physical_grasp_success']),
            'original_successful_same_arm_available':sum(v['retention']['original_selection_same_arm_available']
                for v in values if v['retention']['original_physical_grasp_success']),
            'original_successful_cases':sum(v['retention']['original_physical_grasp_success'] for v in values)}
    return summary


def run(root):
    if (root/'started.json').exists() or (root/'STOP').exists():raise ValueError('Never repeat a started offline registration')
    reg=raw(root/'registration.json');verify(reg);write_json(root/'started.json',{'at':now()})
    begin=time.perf_counter();model=ArmModel();robot=RobotPointFilter()
    grippers={s:GripperGeometry(s) for s in ['right','left']}
    engine=CandidatePostprocessor(grippers,robot,reg['thresholds'])
    native_backend()
    write_json(root/'startup.json',{'at':now(),'passed':True,'seconds':time.perf_counter()-begin,
        'arm_kinematic_proof':model.proof,'sphere_profile':model.sphere_proof,
        'gripper_cavities':{s:[g.cavity[0],g.cavity[1]] for s,g in grippers.items()},
        'planar_parts_kept_for_table_and_unknown_for_point_interior':sum(p.equations is None for g in grippers.values() for p in g.parts)})
    rows=[]
    for ordinal,case in enumerate(reg['cases']):
        directory=Path(case['source_directory']);out=root/'cases'/case['id'];out.mkdir(parents=True)
        write_json(root/'claims'/(case['id']+'.json'),{'at':now(),'state':'claimed','case':case['id']})
        io_begin=time.perf_counter();state=read_json(directory/'state.json')
        candidates=raw(directory/'candidates/candidate_pool.json')['candidates']
        selected=raw(directory/'plan_result.json')['selected'];original=raw(directory/'result.json')
        if len(candidates)!=reg['candidate_counts'][ordinal]:raise ValueError('Original candidate count mismatch')
        if not np.allclose(candidates[selected['index']]['pose'],selected['contact_pose_world'],atol=1e-12,rtol=0):
            raise ValueError('Historical selection no longer identifies its unchanged candidate')
        depth=np.load(directory/'camera/depth_m.npy',allow_pickle=False)
        mask=np.asarray(Image.open(directory/'camera/mask.png'))>0
        completed=np.load(directory/'completed_points_world.npy',allow_pickle=False)
        input_seconds=time.perf_counter()-io_begin
        points,target,observation=engine.observations(state,depth,mask,completed,model.frames(state))
        result=engine.process(candidates,state,points,target,reg['NMS_timing_repeats'])
        retention=selected_retention(result,selected,original['success'])
        variants={}
        for name,value in result['variants'].items():
            if name.startswith('geometry'):value['postprocessing_seconds']+=observation['seconds']
            write_json(out/(name+'.json'),value)
            variants[name]={'representatives':len(value['representatives']),'deferred':len(value['deferred']),
                'geometry_survivors':len(value['input_indices']),'postprocessing_seconds':value['postprocessing_seconds'],
                'retention':retention[name]}
        arrays={};arm_metrics={}
        for side,value in result['geometry'].items():
            for key in ['table_bad','point_bad','point_counts','cavity_counts','kept','kept_nonempty']:
                arrays[side+'_'+key]=value[key]
            arm_metrics[side]={'table_rejections':int(value['table_bad'].sum()),
                'observed_solid_rejections':int(value['point_bad'].sum()),'geometry_survivors':int(value['kept'].sum()),
                'empty_cavity_suspects':int((value['cavity_counts']==0).sum()),'timing':value['timing'],
                'selected_table_reason':value['table_reasons'][selected['index']],
                'selected_observed_collision':bool(value['point_bad'][selected['index']]),
                'selected_cavity_points':int(value['cavity_counts'][selected['index']])}
        np.savez_compressed(out/'geometry_masks.npz',**arrays)
        row={'id':case['id'],'raw_count':result['raw_count'],'original_selected':{'index':selected['index'],'arm':selected['arm']},
            'original_physical_grasp_success':original['success'],'input_read_seconds':input_seconds,
            'observation':observation,'coordinate_validation_seconds':result['coordinate_validation_seconds'],
            'geometry_seconds':result['geometry_seconds'],'arms':arm_metrics,'variants':variants,
            'poses_scores_widths_unchanged':True}
        write_json(out/'result.json',row);rows.append(row)
        write_json(root/'claims'/(case['id']+'.json'),{'at':now(),'state':'complete','result_sha256':sha256(out/'result.json')})
        write_json(root/'progress.json',{'at':now(),'phase':'processing','completed_cases':len(rows),'total_cases':69,
            'current':case['id'],'total_raw_candidates_processed':sum(r['raw_count'] for r in rows),
            'summaries':summarize(rows),'planning_requests':0,'physical_grasps':0})
        print('POSTPROCESS',len(rows),case['id'],'raw',result['raw_count'],
            'geometry_seconds',round(result['geometry_seconds'],3),flush=True)
    verify(reg)
    write_json(root/'acceptance.json',{'at':now(),'task_complete':len(rows)==69,'record_integrity':True,
        'cases':len(rows),'raw_candidates':sum(r['raw_count'] for r in rows),'summaries':summarize(rows),
        'all_poses_scores_widths_and_source_indices_unchanged':True,
        'new_IK_calls':0,'planning_requests':0,'physical_grasps':0,
        'retention_failures':[{'id':r['id'],'original_selected':r['original_selected'],'original_physical_success':r['original_physical_grasp_success'],
            'variants':[n for n,v in r['variants'].items() if not v['retention']['original_selection_same_arm_available']]}
            for r in rows if any(not v['retention']['original_selection_same_arm_available'] for v in r['variants'].values())]})
    write_json(root/'progress.json',{'at':now(),'phase':'complete','completed_cases':69,'total_cases':69,
        'summaries':summarize(rows),'planning_requests':0,'physical_grasps':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--config',type=Path)
    args=parser.parse_args();root=ensure_run_path(args.root)
    if args.stage=='prepare':prepare(root,args.config.resolve())
    else:
        try:run(root)
        except BaseException:
            write_json(root/'STOP',{'at':now(),'error':traceback.format_exc()});raise
