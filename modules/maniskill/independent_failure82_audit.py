"""Frozen first-82 ledger diagnosis and CPU gripper/table replay; no service calls."""
from collections import Counter
from datetime import datetime
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import yaml
import graspcl_pipeline as common

CODE=Path(__file__).resolve().parent
REFERENCE=CODE.parent/'arx_core/scripts/curobo_arx/static_pick/reference'


class GripperGeometry:
    def __init__(self):
        self.models={}
        for role in ['model','model_closed']:
            cfg=yaml.safe_load((REFERENCE/role/'arx.yml').read_text())['robot_cfg']['kinematics']
            joints={j.find('child').get('link'):j for j in ET.parse(REFERENCE/role/'arx.urdf').getroot().findall('joint')}
            self.models[role]=(cfg['collision_spheres'],joints)

    def transform(self,joints,child,values):
        pending=[]
        while child!='right_link26':
            joint=joints[child];pending.append(joint);child=joint.find('parent').get('link')
        out=np.eye(4)
        for joint in reversed(pending):
            origin=joint.find('origin');t=np.eye(4)
            t[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            out=out@t
            if joint.get('type')=='prismatic':
                delta=np.eye(4);delta[:3,3]=np.fromstring(joint.find('axis').get('xyz'),sep=' ')*values[joint.get('name')]
                out=out@delta
            elif joint.get('type')!='fixed':raise RuntimeError('Unexpected non-gripper joint in relative transform')
        return out

    def templates(self,state):
        measured=dict(zip(state['joint_names'],state['qpos']));result=[]
        for role,margin in [('model',0.),('model_closed',.008)]:
            spheres,joints=self.models[role]
            locks={k:measured[k] if role=='model' else .022 for k in ['right_joint27','right_joint28']}
            inverse=np.linalg.inv(self.transform(joints,'right_eef_link',locks));all_s=[];names=[]
            for name in ['right_link26','right_link27','right_link28']:
                transform=inverse@self.transform(joints,name,locks)
                for sphere in spheres[name]:
                    if sphere['radius']<=0:continue
                    center=transform[:3,:3]@sphere['center']+transform[:3,3]
                    all_s.append([*center,float(sphere['radius'])]);names.append(name)
            result.append((np.asarray(all_s),np.asarray(names),margin))
        return result


def points(directory,state):
    params=common.read(directory/'camera/params.json');k=params['intrinsics']
    d=np.asarray(Image.open(directory/'camera/depth.png'),dtype=float)*params['depth_scale']
    mask=np.asarray(Image.open(directory/'camera/mask.png'))>0;v,u=np.indices(d.shape)
    xyz=np.stack([(u-k['ppx'])*d/k['fx'],(v-k['ppy'])*d/k['fy'],d],-1)
    xyz=xyz[mask&np.isfinite(d)&(d>.12)&(d<2.)]
    camera=np.asarray(state['camera_world_optical']);xyz=xyz@camera[:3,:3].T+camera[:3,3]
    _,indices=np.unique(np.floor(xyz/.002).astype(np.int64),axis=0,return_index=True)
    return xyz[np.sort(indices)]


def clearance(spheres,pose,table,margin):
    xyz=spheres[:,:3]@pose[:3,:3].T+pose[:3,3]
    center=np.asarray(table['center']).copy();center[2]+=margin
    delta=np.abs(xyz-center)-np.asarray(table['dims'])/2
    sdf=np.linalg.norm(np.maximum(delta,0),axis=-1)+np.minimum(np.max(delta,axis=-1),0)
    return float(np.min(sdf-spheres[:,3]))


def sweep_support(xyz,grasp,width,template):
    spheres,names,_=template;bounds=[]
    for name in ['right_link27','right_link28']:
        subset=spheres[names==name]
        bounds.append((np.min(subset[:,:3]-subset[:,3,None],axis=0),np.max(subset[:,:3]+subset[:,3,None],axis=0)))
    low=np.maximum(bounds[0][0],bounds[1][0]);high=np.minimum(bounds[0][1],bounds[1][1])
    local=(xyz-grasp[:3,3])@grasp[:3,:3]
    keep=(local[:,0]>=low[0])&(local[:,0]<=high[0])&(local[:,2]>=low[2])&(local[:,2]<=high[2])&(np.abs(local[:,1])<=width/2)
    return int(np.count_nonzero(keep))


def audit(root,out):
    arm=root/'fixed_zerograsp';claims=[common.read(f) for f in (arm/'claims').glob('*.json')]
    claims=sorted((c for c in claims if c['state']=='complete'),key=lambda c:(c['finished_at'],c['slot']))[:82]
    assert len(claims)==82
    geometry=GripperGeometry();counts=Counter();rows=[];aggregate_rejections=Counter();max_error=0.;empty_diagnostics=[]
    for claim in claims:
        directory=arm/'trials'/claim['slot'];result=common.read(directory/'result.json')
        assert common.sha(directory/'result.json')==claim['result_sha256']
        assert common.read(directory/'independent_evaluation.json')['passed']
        row={'id':result['id'],'model_id':result['model_id'],'layout':result['layout'],
            'planning_success':result['planning_success'],'result_reason':result['reason'],'finished_at':claim['finished_at']}
        if result['planning_success']:category='passed'
        elif result['reason']=='independent_scene_admission_failed':
            category='admission_failed'
            admission=common.read(Path(common.read(arm/'registration.json')['inputs'])/'admission'/result['id']/'admission.json')
            row['admission_failure_reasons']=dict(Counter(e['error'] for e in admission['rejected_variants']))
        else:
            plan=common.read(directory/'plan_result.json');n=plan['count'];r=plan['rejected'];aggregate_rejections.update(r)
            truncation=common.read(directory/'truncation.json');pool=common.read(directory/'graspcl/candidate_pool.json')
            row.update(candidate_count_before_truncation=truncation['input_candidates'],candidate_count_after_truncation=n,
                candidate_rejections=r,source_pool_value_width_rejections=len(pool.get('rejected',[])))
            if n==0:category='empty_after_truncation'
            elif r['grasp_ik']==n:category='grasp_ik_or_full_scene_collision'
            elif r['grasp_ik']+r['closed_collision']==n:category='nominal_closed_collision'
            elif r['grasp_ik']+r['closed_collision']+r['pre_or_lift_ik']==n:category='pre_or_lift_ik'
            elif r['path']:category='path'
            else:category='other'
            state=common.read(directory/'state.json');templates=geometry.templates(state);table=state['scene_cuboids']['table']
            for candidate in pool['candidates']:
                pose=np.asarray(candidate['pose']);g=pose.copy();g[:3,3]-=.043*pose[:3,2]
                exact=min(clearance(s,g,table,m) for s,_,m in templates)
                error=abs(exact-candidate['metadata']['gripper_table_clearance_after_m']);max_error=max(max_error,error)
            if n==0:
                source=common.read(directory/'graspcl/unrefined_candidate_pool.json')['candidates'];xyz=points(directory,state)
                causes=Counter();items=[]
                for candidate in source:
                    contact=np.asarray(candidate['pose']);depth=candidate['metadata']['source_network_depth_m']
                    had_clear=False;accepted=False;best=-float('inf')
                    for s in np.unique(np.r_[np.arange(0,depth,.002),depth]):
                        grasp=contact.copy();grasp[:3,3]-=(.043+s)*contact[:3,2]
                        gap=min(clearance(spheres,grasp,table,margin) for spheres,_,margin in templates);best=max(best,gap)
                        if gap<0:continue
                        had_clear=True
                        if sweep_support(xyz,grasp,candidate['width'],templates[0]):accepted=True;break
                    cause='unexpected_cpu_retained' if accepted else 'no_observed_target_support_at_table_clear_depth' if had_clear else 'no_gripper_table_clear_depth_in_allowed_range'
                    causes[cause]+=1;items.append({'source_row':candidate['metadata']['postprocessed_row_index'],
                        'depth_m':depth,'best_gripper_table_clearance_m':best,'cause':cause})
                empty_diagnostics.append({'id':result['id'],'causes':dict(causes),'candidates':items})
        counts[category]+=1;row['exclusive_case_category']=category;rows.append(row)
    assert counts['passed']==45 and sum(counts.values())==82
    assert counts=={'passed':45,'admission_failed':3,'empty_after_truncation':4,'grasp_ik_or_full_scene_collision':15,'pre_or_lift_ik':15}
    assert max_error<.00015,('CPU gripper model differs from recorded original checker',max_error)
    assert not any(r['causes'].get('unexpected_cpu_retained') for r in empty_diagnostics)
    data={'created_at':datetime.now().astimezone().isoformat(),'read_only':True,'new_physical_attempts':0,
        'source':str(arm),'cohort':'First 82 terminal claims in chronological completion order',
        'cutoff_finished_at':claims[-1]['finished_at'],'case_counts':dict(counts),
        'failed_inferred_candidate_rejections':dict(aggregate_rejections),
        'at_least_grasp_pose_and_open_closed_valid_scenes':60,
        'gripper_cpu_vs_saved_checker_clearance_max_error_m':max_error,
        'empty_candidate_details':empty_diagnostics,'cases':rows,
        'limits':'grasp_ik combines full-pose solver failure and original scene collision. pre_or_lift_ik combines two endpoints. Existing logs do not separate those causes, so these are not counts of globally unreachable arm positions. No solver/physics/planner/table margin was changed or rerun.'}
    out.mkdir(parents=True,exist_ok=False);common.save(out/'audit.json',data)
    lines=['# 独立测试最先完成 82 例：规划拒绝核查','',
        '固定最先完成的 82 个终态槽：45 例通过初始完整方案，37 例未通过。主 benchmark 持续运行，未改参数、调用规划服务或新增抓取。','',
        '| 未通过阶段 | 场景数 | 说明 |','|---|---:|---|',
        '| 初始化/遮挡视图 | 3 | 12 个初始化变体均未通过；均为部分遮挡场景 |',
        '| 插入截断后无候选 | 4 | 3 个刀场景、1 个乐高场景；并非抓取 IK 拒绝 |',
        '| 抓取 IK/完整碰撞检查 | 15 | 剩余候选全部在该合并门槛被拒绝 |',
        '| 预抓取或抬升 IK | 15 | 至少有一个抓取位姿通过 IK 及开/名义闭爪碰撞；前后动作未通过 |','',
        '至少 60 个场景存在通过原抓取 IK、开爪碰撞与名义闭爪碰撞检查的候选。45 表示包含前后动作要求的初始方案通过数，不是只统计抓取终点的位置可达。路径搜索与优化没有单独导致本批任何一个场景失败。','',
        '失败的已推理场景中，候选级拒绝为：抓取 IK/碰撞 230、名义闭爪碰撞 1、预抓取或抬升 IK 25。候选数不能直接当作场景数。','',
        '4 个空池场景的几何回放（CPU、原 ARX 球模型和原 8 mm 余量）：']
    for row in empty_diagnostics:lines.append('- '+row['id']+'：'+str(row['causes']))
    lines+=['',f'CPU 夹爪几何与原检查器已保存净空值的最大差异为 {max_error*1000:.6f} mm。','',
        '当前日志把抓取求解与碰撞失败合并为 grasp_ik，把预抓取与抬升合并为 pre_or_lift_ik，不能直接将这些例子定性为机械臂本体不可达。若进一步区分，需要单独离线求解核查，不能修改或回灌当前评测。','',
        '[逐例数据与完整回放记录](audit.json)','']
    (out/'前82例规划失败核查.md').write_text('\n'.join(lines));return data


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();data=audit(args.root.resolve(),args.output.resolve())
    print(json.dumps({'counts':data['case_counts'],'empty_details':[{k:v for k,v in r.items() if k!='candidates'} for r in data['empty_candidate_details']],
        'cpu_geometry_max_error_m':data['gripper_cpu_vs_saved_checker_clearance_max_error_m']},ensure_ascii=False))
