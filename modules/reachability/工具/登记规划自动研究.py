"""Freeze development cases, held-out cases, constraints and source evidence."""
from 公共 import EXPERIMENTS
from pathlib import Path
import hashlib

import numpy as np
from 公共 import ROOT,PROJECTS,MS,MODEL,URDF,TCP,read,save,sha,now

CAMPAIGN=EXPERIMENTS / '自动研究_20261120_01'


def case_record(directory,group,kind,key=None):
    directory=Path(directory);state=read(directory/'state.json');pool=read(directory/'candidates/candidate_pool.json')
    assert pool['candidates']
    assert all(c['metadata'].get('origin')=='zerograsp' for c in pool['candidates'])
    assert all(abs(c['metadata'].get('action_tcp_offset_eef_z_m',.043)-.043)<1e-9 for c in pool['candidates'])
    for name,g in state.get('scene_meshes',{}).items():
        assert sha(g['file_path'])==g['file_sha256'],name
    meta=read(ROOT/'产物/首版_01/地图元数据.json')
    lift=state['qpos'][state['joint_names'].index('joint4')]
    assert abs(lift-meta['lift_joint_m'])<1e-5
    assert state['scene_cuboids']==meta['static_world_cuboids']
    return {'case':key or directory.name,'directory':str(directory),'group':group,'stratum':kind,
            'candidates':len(pool['candidates']),'source_state_sha256':sha(directory/'state.json'),
            'source_candidates_sha256':sha(directory/'candidates/candidate_pool.json')}


def main():
    CAMPAIGN.mkdir(parents=True,exist_ok=False)
    previous=PROJECTS/'claro/docs/03_夹走并放下挡物/规划失败核查/resources/20261119_01/核查登记.json'
    cases=[]
    for c in read(previous)['cases']:
        kind='planning_control' if c['control'] else 'entry_collision' if c['reason'].startswith('进场') else 'grasp_or_pregrasp_failure'
        cases.append(case_record(c['directory'],'development',kind,c['case']))
    for key in ['I0048','I0055']:
        d=PROJECTS/'claro/docs/06_做配对验证/resources/S5_固定挡物配对01/试次'/key/'01_抓挡'
        cases.append(case_record(d,'development','planning_control',key))
    excluded={c['case'].split('_')[0] for c in cases}|{'I0100'}
    buckets={'grasp_failure':[],'pregrasp_failure':[],'success_control':[]}
    bank=MS/'outputs/arx_physical_table_parallel_20261118_01/zerograsp/fixed_zerograsp/trials'
    for d in sorted(bank.iterdir()):
        if d.name.split('_')[0] in excluded or not (d/'plan_result.json').exists():continue
        result=read(d/'result.json');plan=read(d/'plan_result.json')
        if result.get('success'):kind='success_control'
        elif not plan.get('success') and plan.get('rejected',{}).get('pre_or_lift_ik',0):kind='pregrasp_failure'
        elif not plan.get('success') and plan.get('rejected',{}).get('grasp_ik',0):kind='grasp_failure'
        else:continue
        if not read(d/'candidates/candidate_pool.json')['candidates']:continue
        buckets[kind].append(d)
    for kind,count in [('grasp_failure',3),('pregrasp_failure',3),('success_control',6)]:
        ranked=sorted(buckets[kind],key=lambda d:hashlib.sha256(('20261120:'+d.name).encode()).hexdigest())
        chosen=[];models=set()
        for d in ranked:
            model=read(d/'result.json')['model_id']
            if model in models:continue
            chosen.append(d);models.add(model)
            if len(chosen)==count:break
        assert len(chosen)==count,(kind,len(chosen))
        cases.extend(case_record(d,'heldout',kind,'验证_'+d.name) for d in chosen)
    assert len(cases)==24
    save(CAMPAIGN/'场景清单.json',{'created_at':now(),'cases':cases,
         'selection':'12 development: 7 grasp/pre failures, 2 entry collision, 3 prior planning-pass controls. Heldout: 3 grasp failures +3 pre/lift failures +6 actual-success controls from old200, SHA256 seed order with unique target models per stratum. No new outcome used.',
         'heldout_results_not_used_for_iteration':True,'all_inputs_native_postprocessed_candidates':True})
    protected=[MODEL,URDF,TCP,previous,MS/'arx_pipeline/planner.py',MS/'resident_planner.py',
               MS/'fast_resident_planner.py',MS/'short_lift_planner.py',MS/'table_truncation_model.py',
               MS/'tmp/physical_table_pair_20261118_01.py',ROOT/'工具/公共.py',ROOT/'工具/能力索引.py',ROOT/'工具/机器人几何.py',
               PROJECTS/'arx_core/scripts/curobo_arx/optimize_path.py']
    for c in cases:
        d=Path(c['directory']);protected.extend([d/'state.json',d/'candidates/candidate_pool.json',d/'plan_result.json'])
        protected.extend(Path(g['file_path']) for g in read(d/'state.json').get('scene_meshes',{}).values())
    save(CAMPAIGN/'登记.json',{'created_at':now(),'source_sha256':{str(p):sha(p) for p in set(protected)},
        'map':str(ROOT/'产物/首版_01'),'map_completion_sha256':sha(ROOT/'产物/首版_01/完成.json'),
        'required_components':['frozen ARX cuRobo open/closed planner and original scene checker','frozen FK capability bank'],
        'runtime':'offline replay of original ARX/MANI benchmark measured states; no simulator commands',
        'planning_metric':'G/H/nominal3cm-lift IK + full-scene including target entry path + target-excluded approach path + original trajectory optimizer and dense checks',
        'per_scene_budget':{'ik_calls':96,'path_pairs':12,'wall_seconds':12},'max_iterations':6,
        'heldout_evaluations':'baseline, frozen best map variant and blind-distance ablation, once each; no tuning on these results',
        'repeats':'two development seeds for the frozen best and blind ablation; solver reset between arms',
        'physical_attempt_budget':0,'robot_configuration_changes_allowed':False,'candidate_G_pose_changes_allowed':False,
        'uncovered_is_rejection':False,'default_promotion_allowed':False})
    save(CAMPAIGN/'来源.json',{'sources':[
        {'title':'DLR: Reachability and Dexterity: Analysis and Applications for Space Robotics','url':'https://elib.dlr.de/97212/1/Astra-PUBLISHED.pdf',
         'evidence':'Section II describes FK/IK/hybrid reachability maps, rotational grids and map coverage limits.',
         'hypotheses':['H01 map seeds','H02 map-ranked approach distance','H04 local geometric refinement from independent FK seeds'],
         'reproduction_level':'Inspired engineering changes; not reproduction of DLR evaluation.'},
        {'title':'Dynamic Grasping with Reachability and Motion Awareness','url':'https://arxiv.org/html/2103.10562',
         'evidence':'Sections IV-C/IV-D rank native grasp candidates using reachability and use motion seeds.',
         'hypotheses':['H03 map grasp ranking'],
         'reproduction_level':'Only ranking/seed ideas; no moving-object/RNN reproduction or new learned model.'},
        {'title':'ROS-Industrial REACH official code','url':'https://github.com/ros-industrial/reach','hypotheses':['map evaluation and joint-quality metrics']},
        {'title':'DLR official map code','url':'https://github.com/DLR-MO/reachability_map','hypotheses':['FK witness index']}],
         'negative_evidence':{'prior_endpoint_queries':133,'original_pass':20,'map_seed_pass':20,
             'source':str(ROOT/'产物/首版_01/查询验证_01/查询对照结果.json'),
             'interpretation':'No existing proof that first-version seeds alone improve endpoint solve count.'}})
    save(CAMPAIGN/'状态.json',{'status':'registered','started_at':now(),'completed_iterations':0,'physical_attempts':0,'heldout_opened':False})
    (CAMPAIGN/'研究计划.md').write_text('''# 可达性地图辅助规划自动研究

本轮用户授权自动选择场景并迭代，提高当前 ARX 链路的规划通过率。研究目录独立，旧结果、旧脚本和默认执行器不变。文件名沿用用户的中文约定；本文件对应技能中的 research.md，其余日志、来源、状态和结果表同样使用中文名称。

## 目标与评测

目标是完整初始规划通过率，而非仅位置 IK 或实际抓取成功率。12 个开发场景用于迭代，12 个预先冻结的留出场景只在方案冻结后验证。开发场景至少新增通过 4 个、无原成功对照退步才达到主要开发目标；还需相对不使用地图的短距离对照有增益或等通过数下更少规划调用，才能把收益归因于地图。

评估器：`工具/评估规划研究.py <迭代目录>`，机械输出 pass、score、通过数、增益和退步。Keep policy 为 score_improvement，并要求对照无退步；没有提升的独立变体保留负结果，但不进入默认链路。

每场景统一上限 96 次 IK、12 对路径、12 秒计算；资源启动／地图加载计时单列，批内常驻。最多 6 次开发迭代，每轮执行上限 5 分钟。pause_every=never；本次在当前任务内持续运行，不启动无限后台循环。完成或预算用尽后输出最终报告。

## 冻结项

机器人模型、关节限位、TCP43、原碰撞球、自碰撞策略、桌地几何、物体初始状态、完整杂物世界、抓取候选 G、原优化有效性检查不变。只改变候选／H 的尝试顺序、正常 H 运动目标及 IK 初值或求解算法。H 可取沿原接近轴后退 2/4/6/8 cm；不改变 G 的位置、朝向、夹持点或候选总集合。地图未覆盖始终表示未知，不作为拒绝规则。

沿用当前 2026-11-18 实际桌面、开爪规划、3 cm 名义抬升的 ARX benchmark 协议。本轮不执行物理动作。旧技能中的 Isaac 实例、固定 GPU、6 cm 物理抓取阈值属于早期实验，当前任务采用用户后续指定的现有 ARX/ManiSkill 链路和“规划成功率”目标；不将离线计算宣称物理验证。

已有另一项 GraspCL 300 槽任务正在 GPU2/3 运行，本任务不接触其服务、场景、经验库或源文件。选择其他有足够空闲显存的设备，启动前复查；基础设施或源完整性故障停止，不自动吞掉失败。

## 搜索空间与消融

1. H01：固定原 G/H，地图提供 IK 初值，检验单独种子作用。
2. H02：保留原候选次序，地图对 2/4/6/8 cm H 排序，检验接近距离选择。
3. H03：在 H02 上同时按 G 与 H 的完整位姿邻近性排序候选；不剔除未知候选。
4. 若前三项不足，转向从地图关节见证局部数值细化精确目标，使用原 FK／限位及原 checker 重新验证，检验地图是否能补充困难 IK 解。
5. 对保留方案做一次局部调整与开发确认；另跑不使用地图的同距离集合／同预算对照，隔离多试距离的收益。
6. 最终方案冻结后，留出集只测原基线、冻结方案和不用地图的距离对照，不根据留出结果调参。

旧 133 点查询两组均 20 点通过；已有 5/7 短 H 离线恢复证据仅作为开发线索，不称新发现或独立验证。保留全部未解出、超时、回归与资源故障记录。成功控制与混合场景均纳入，不用失败精选集推断总体成功率。

## 历史

| 轮次 | 假设 | 规划通过 | 增益／退步 | 决定 |
|---|---|---|---|---|
''')
    (CAMPAIGN/'研究日志.md').write_text('# 研究日志\n\n已完成范围、来源和 24 个输入场景登记；尚无本轮规划结果。\n')
    (CAMPAIGN/'迭代结果.tsv').write_text('iteration\tmetric_value\tdelta\tdelta_pct\tstatus\tdescription\tevaluator_source\ttimestamp\n')
    lines=['# 场景选择','', '| 分组 | 场景 | 类型 | 原生候选数 |','|---|---|---|---|']
    lines.extend(f"| {c['group']} | {c['case']} | {c['stratum']} | {c['candidates']} |" for c in cases)
    (CAMPAIGN/'场景选择.md').write_text('\n'.join(lines)+'\n')
    print('REGISTERED',CAMPAIGN,'cases',len(cases))


if __name__=='__main__':main()
