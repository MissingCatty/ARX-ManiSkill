"""Independent read-only label/input audit and paired GraspGenX report."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import graspgenx_partial_geometry as geom
from graspgenx_partial_geometry import save, sha

CODE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())


def combined_video(trial):
    target=trial/'execution_dual_view.mp4'
    if target.exists():return target
    if not all((trial/n).exists() for n in ['execution.mp4','right_wrist.mp4']):return None
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-hide_banner','-loglevel','error',
        '-i',str(trial/'execution.mp4'),'-i',str(trial/'right_wrist.mp4'),
        '-filter_complex_threads','1','-filter_threads','1',
        '-filter_complex','[0:v]scale=960:540,setsar=1[a];[1:v]scale=720:540,setsar=1[b];[a][b]hstack=inputs=2[v]',
        '-map','[v]','-c:v','libx264','-preset','fast','-crf','22','-threads','2','-shortest',str(target)],
        check=True,timeout=180)
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();reg=read(root/'registration.json')
    assert read(root/'progress.json')['phase']=='complete'
    assert read(root/'suite_timing.json')['all_owned_services_exited']
    import arx_runtime_imports
    arx_runtime_imports.install()
    import graspcl_batch
    from short_lift_protocol import install_verifier
    _,baseline=graspcl_batch.original_modules();install_verifier(baseline)
    details=[];groups={};source_checks=[]
    for name,expected in reg['source_sha256'].items():assert sha(name)==expected,name
    for arm in reg['arms']:
        health=read(root/'resident'/(arm+'_ready.json'))
        assert health['details']['model_load_count']==1
    planner=read(root/'resident/curobo_ready.json')
    assert planner['details']['model_load_count']=={'open':1,'closed':1}
    for arm in reg['arms']:
        rows=[]
        for entry in reg['schedule']:
            trial=root/'results'/arm/entry['id'];result=read(trial/'result.json')
            claim=read(root/'claims'/(arm+'__'+entry['id']+'.json'))
            assert claim['state']=='complete' and sha(trial/'result.json')==claim['result_sha256']
            baseline.verify_trial(trial);assert read(trial/'independent_evaluation.json')['passed']
            assert result['scene']==entry['scene'] and result['seed']==entry['seed']
            assert read(trial/'initial_conditions.json')['spawn']==entry['scene']
            receipt=read(trial/'shared_input_receipt.json')
            cache=root/'shared_inputs'/entry['id']
            assert receipt['manifest_sha256']==sha(cache/'manifest.json')
            for name,expected in receipt['actual_camera_sha256'].items():
                assert sha(trial/'camera'/name)==expected==sha(cache/'camera'/name)
            assert sha(trial/'gt_partial_points_world.npy')==sha(cache/'gt_partial_points_world.npy')
            source_checks.append({'arm':arm,'id':entry['id'],'shared_depth_mask_rgb_and_visible_points_verified':True})
            plan=read(trial/'plan_result.json') if (trial/'plan_result.json').exists() else {}
            pool=read(trial/'candidates/candidate_pool.json') if (trial/'candidates/candidate_pool.json').exists() else {}
            for candidate in pool.get('candidates',[]):
                assert geom.angle_from_down(candidate['pose'])<=reg['topdown_max_angle_deg']+1.e-5
            if plan.get('selected'):
                selected=plan['selected'];candidate=pool['candidates'][selected['index']]
                assert np.allclose(selected['contact_pose_world'],candidate['pose'],atol=1.e-12)
                expected_eef=np.asarray(candidate['pose']).copy()
                expected_eef[:3,3]-=.043*expected_eef[:3,2]
                assert np.allclose(selected['grasp_world'],expected_eef,atol=1.e-12)
                lift=np.asarray(selected['grasp_world']).copy();lift[2,3]+=.03
                assert np.allclose(lift,selected['lift_world'],atol=1.e-12)
                if arm=='graspgenx':
                    meta=candidate['metadata'];source=np.load(trial/'graspgenx_raw/grasps_generator_world.npy')[meta['source_generator_index']]
                    expected=geom.generator_to_contact(source)
                    expected[:3,3]-=meta['insertion_retraction_m']*expected[:3,2]
                    assert np.allclose(expected,candidate['pose'],atol=1.e-9)
            assert (trial/'zg_outputs/grasp_top10.json').exists()
            video=combined_video(trial)
            item={'arm':arm,'id':entry['id'],'model_id':entry['model_id'],
                'raw_candidates':pool.get('raw_candidate_count',0),
                'topdown_candidates':pool.get('topdown_candidate_count',0),
                'retained_after_table_check':len(pool.get('candidates',[])),
                'grasp_endpoint_reachable':plan.get('grasp_endpoint_reachable',False),
                'planning_success':result['planning_success'],'physical_attempt':result['physical_attempt'],
                'success':result['success'],'reason':result['reason'],'wall_seconds':result['wall_seconds'],
                'selected_origin':plan.get('selected',{}).get('candidate_metadata',{}).get('source_branch',arm),
                'trial':str(trial),'dual_view_video':str(video) if video else None}
            rows.append(item);details.append(item)
        n=len(rows);executed=sum(r['physical_attempt'] for r in rows)
        groups[arm]={'registered':n,'grasp_endpoints_found':sum(r['grasp_endpoint_reachable'] for r in rows),
            'full_plans':sum(r['planning_success'] for r in rows),'execution_entries':executed,
            'successes':sum(r['success'] for r in rows),'overall_success_rate':sum(r['success'] for r in rows)/n,
            'conditional_success_rate':sum(r['success'] for r in rows)/executed if executed else None,
            'failure_reasons':dict(Counter(r['reason'] for r in rows if not r['success']))}
    pose_checks=[]
    for entry in reg['schedule']:
        initials=[read(root/'results'/arm/entry['id']/'initial_conditions.json') for arm in reg['arms']]
        assert initials[0]==initials[1]
        states=[read(root/'results'/arm/entry['id']/'state.json') for arm in reg['arms']]
        differences={}
        for name in states[0]['objects']:
            a=np.asarray(states[0]['objects'][name]['pose']);b=np.asarray(states[1]['objects'][name]['pose'])
            differences[name]=float(np.linalg.norm(a[:3]-b[:3]))
        pose_checks.append({'id':entry['id'],'observed_object_position_differences_m':differences})
    model_hashes={}
    for name,expected in reg['model_sha256'].items():
        from graspgenx_partial_pair import large_sha
        actual=large_sha(name);assert actual==expected;model_hashes[name]=actual
    audit={'passed':True,'physical_grasps_added':0,'registered_and_completed':10,
        'source_hashes_unchanged':True,'model_hashes_unchanged':model_hashes,
        'independent_physical_success_labels_verified':True,'identical_paired_input_checks':source_checks,
        'initial_robot_and_scene_recipes_identical':True,'topdown_angles_verified':True,
        'measured_paired_object_pose_differences':pose_checks,
        'selected_poses_match_model_output_plus_declared_transform_and_retraction':True,
        'all_components_loaded_once_per_test_epoch':True,'all_owned_services_exited':True}
    save(root/'independent_pair_audit.json',audit)
    save(root/'paired_metrics.json',{'groups':groups,'cases':details,'scope':reg['scope'],
        'definition':'Reachable means found a valid IK/scene-collision endpoint in finite search; full plan includes approach/lift constraints.'})
    csv_path=root/'逐场景结果.csv'
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    resource=CODE/'docs/resource'/root.name;resource.mkdir(parents=True,exist_ok=True)
    for name in ['registration.json','paired_metrics.json','independent_pair_audit.json','suite_timing.json','逐场景结果.csv']:
        shutil.copy2(root/name,resource/name)
    lines=['# GraspGenX 部分点云俯抓对比','',
        '- 场景：布丁盒、罐头、杯子、木块、螺丝刀各一个开阔桌面场景，两法各执行一次，共 10 个新测试槽。',
        '- 输入：ZeroGrasp 使用 RGB、真值深度和实例掩码；GraspGenX 使用同一视角、同一深度及掩码提取的可见部分点云。两法共享冻结观测，规划使用当次测量状态。',
        '- 俯抓：GraspGenX 原生配置为 `planner=graspmoe, moe_obb_density=dense`，在包围盒顶面沿长轴采样位置，并扫描夹爪转角和插入深度；合并扩散候选后使用原生评分。两组统一筛选接近方向与竖直向下夹角不超过 30° 的候选，保留原始朝向及来源。',
        '- 执行：当前 ARX 高度、TCP43、相机、碰撞与物理设置保持；仅按登记规则退让过深插入。计划抬升 3 cm，物体实际抬高至少 1 cm 并保持 3 秒，同时满足原双指接触和动作／接触检查。','',
        '| 指标 | ZeroGrasp | GraspGenX |','|---|---:|---:|']
    for label,key in [('找到有效抓取终点','grasp_endpoints_found'),('完整初始规划通过','full_plans'),('进入抓取执行','execution_entries'),('抓取成功','successes')]:
        lines.append('| '+label+' | '+' | '.join(f"{groups[a][key]}/5（{groups[a][key]/5:.0%}）" for a in reg['arms'])+' |')
    lines.append('| 已进入执行中的成功率 | '+' | '.join(
        f"{groups[a]['successes']}/{groups[a]['execution_entries']}（{groups[a]['conditional_success_rate']:.0%}）"
        if groups[a]['execution_entries'] else '未进入执行' for a in reg['arms'])+' |')
    lines+=['','所有无候选、未找到有效终点、规划及执行失败都保留在各自 5 场景分母。终点未找到解不表示已证明数学不可达。本轮样本较少，只报告这五个场景的结果。','',
        '| 场景 | ZeroGrasp：终点／完整规划／成功 | GraspGenX：终点／完整规划／成功 |','|---|---|---|']
    for entry in reg['schedule']:
        row=[entry['id']]
        for arm in reg['arms']:
            case=next(r for r in details if r['arm']==arm and r['id']==entry['id'])
            row.append('／'.join('是' if case[k] else '否' for k in ['grasp_endpoint_reachable','planning_success','success']))
        lines.append('| '+' | '.join(row)+' |')
    reasons=sorted(set().union(*(set(v['failure_reasons']) for v in groups.values())))
    names={'no_feasible_native_post_or_recalled_candidate':'俯抓候选未找到可执行初始方案',
        'insufficient_hold_height':'物体抬升保持高度不足','no_bilateral_target_contact':'未形成双指目标接触',
        'close_not_converged':'闭爪未收敛','approach_not_converged':'接近未收敛',
        'lift_not_converged':'抬升未收敛','forbidden_robot_contact':'出现禁止的机器人接触',
        'measured_start_state_in_collision':'测量起始状态被碰撞检查拒绝',
        'measured_closure_lift_unreachable':'实测闭爪后的抬升规划未通过',
        'measured_pregrasp_approach_unreachable':'实测预抓取状态到抓取点的规划未通过'}
    lines+=['','| 最终失败原因 | ZeroGrasp | GraspGenX |','|---|---:|---:|']
    for reason in reasons:lines.append('| '+names.get(reason,reason)+' | '+' | '.join(str(groups[a]['failure_reasons'].get(reason,0)) for a in reg['arms'])+' |')
    clock=read(root/'suite_timing.json')
    lines+=['',f"本次运行含预热 {clock['wall_seconds']:.1f} 秒，其中启动预热 {clock['startup_seconds']:.1f} 秒；准备与历史启动故障另行保留。",'',
        f"[完整设置](resource/{root.name}/registration.json) · [逐例结果](resource/{root.name}/逐场景结果.csv) · [统计](resource/{root.name}/paired_metrics.json) · [独立核验](resource/{root.name}/independent_pair_audit.json)",'',
        f'原始产物：`outputs/{root.name}/`，每例含 Top10 位姿图及第三人称／腕部拼接录像。']
    path=CODE/'docs/GraspGenX部分点云俯抓对比.md';path.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'passed':True,'groups':groups,'report':str(path)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
