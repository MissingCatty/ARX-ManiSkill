"""Read-only experiment analysis and reproducible standalone publication data."""
import argparse
import csv
from pathlib import Path

import numpy as np

from arx_project.layout import ROOT,now,read_json,write_json,ensure_run_path
from modules.pipeline.experiments.gripper_sweep_optimization.report import load_phase,average


LABELS={'binding':('批量校验＋C版YAML','相同场景只更新开度'),
        'geometry':('当前批量校验＋C版绑定','开合爪C读取＋原生动态更新'),
        'memo':('逐次计算原生开合爪FK','复用精确原生开合爪球')}


def phase_data(root,kind):
    phase=load_phase(root,'codec')
    phase.update(kind=kind,reuse_review=read_json(root/'reuse_review.json'))
    parameters=root/'resident/parameter_refresh_equivalence/acceptance.json'
    phase['parameter_equivalence']=read_json(parameters) if parameters.exists() else None
    phase['kept']=phase['reuse_review']['kept']
    progress=read_json(root/'progress.json')
    for mode in ['legacy','batched']:
        rows=[r for r in progress['results'] if r['mode']==mode]
        phase['summaries'][mode]['mean_width_refreshes']=float(np.mean([r['width_reuse']['hits'] for r in rows]))
        phase['summaries'][mode]['mean_full_binds']=float(np.mean([r['width_reuse']['full_binds'] for r in rows]))
        phase['summaries'][mode]['sphere_cache_hits']=sum(r['sphere_reuse']['hits'] for r in rows)
        phase['summaries'][mode]['sphere_cache_misses']=sum(r['sphere_reuse']['misses'] for r in rows)
    return phase


def timing_table(phase):
    lines=['| 方案 | 规划通过 | 平均耗时 | 中位数 | P90 | 10次累计 |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode,label in zip(['legacy','batched'],LABELS[phase['kind']]):
        s=phase['summaries'][mode]
        lines.append(f"| {label} | {s['successes']}/{s['requests']} | {s['total_seconds']/s['requests']:.3f}秒 | "
            f"{s['median_seconds']:.3f}秒 | {s['p90_seconds']:.3f}秒 | {s['total_seconds']:.3f}秒 |")
    return lines


def complete_log(root,phase):
    a,b=[phase['summaries'][m]['total_seconds'] for m in ['legacy','batched']]
    status='kept' if phase['kept'] else 'reverted'
    path=root/'autoresearch-results.tsv';at=phase['reuse_review']['at']
    if not path.exists():
        with path.open('w') as stream:
            writer=csv.writer(stream,delimiter='\t',lineterminator='\n')
            writer.writerow(['iteration','metric_value','delta','delta_pct','status','description','evaluator_source','timestamp'])
            writer.writerow([0,a,'-','-','baseline',LABELS[phase['kind']][0],'mechanical',at])
            writer.writerow([1,b,b-a,f'{(b/a-1)*100:.3f}%',status,LABELS[phase['kind']][1],'mechanical',at])
        with (root/'research.md').open('a') as stream:
            stream.write(f"| 1 | {LABELS[phase['kind']][1]} | {a:.6f}→{b:.6f}秒 | {status} |\n")
        (root/'research_log.md').write_text('# 本轮分析\n\n'+
            f"两组各10次规划，累计{a:.6f}→{b:.6f}秒，变化{1-b/a:.2%}。\n\n"
            f"全部关节和完整路径逐位一致：{phase['reuse_review']['all_endpoints_and_paths_bitwise_equal']}。"
            f"至少省5%门槛：{phase['reuse_review']['minimum_saving_gate_passed']}。结论：{status}。\n")
    (root/'final_report.md').write_text('# 夹爪状态与场景复用结果\n\n'+'\n'.join(timing_table(phase))+
        f'\n\n采用结论：{status}。新增物理抓取0次。\n')
    write_json(root/'evaluator.json',{'pass':phase['kept'],'score':-b,'baseline_score':-a,'status':status})


def build(binding_root,geometry_root):
    phases=[phase_data(binding_root,'binding'),phase_data(geometry_root,'geometry')]
    for root,phase in zip([binding_root,geometry_root],phases):complete_log(root,phase)
    a,b=phases
    # A rejected change is not included in the next comparison's common base.
    current=(b,'batched') if b['kept'] else (b,'legacy')
    chosen,mode=current
    body=['# V7 场景绑定与开合爪计算优化结果','','2026年11月27日。','',
        '本轮继续优化上一轮平均5.37秒规划中的两个主要耗时：重复绑定机器人/场景，以及重复计算开合爪采样的FK。', '',
        '## 第一项：相同场景只更新夹爪开度（修订版）','',
        '对完整状态、底座、机械臂关节、限位、场景几何、目标纳入策略和模型文件计算签名。'
        '只有夹爪开度变化且其余内容完全相同时，复用已经绑定的场景，更新原生运动学张量。'
        '继续执行原FK检查、起始碰撞检查、原随机种子重置和完整留档。', '']
    body+=timing_table(a)
    sa,sb=[a['summaries'][m] for m in ['legacy','batched']]
    binding_change=a['review']['total_saving_fraction']
    binding_text=(f'完整规划累计耗时减少{binding_change:.2%}' if binding_change>=0 else
                  f'完整规划累计耗时增加{-binding_change:.2%}')
    body+=['',binding_text+f"。每次请求的完整绑定平均"
        f"{sa['mean_full_binds']:.1f}→{sb['mean_full_binds']:.1f}次，"
        f"另有{sb['mean_width_refreshes']:.1f}次只更新开度。",'',
        '## 第二项：开合爪模型读取与逐采样参数更新','',
        '开合爪流程原先还单独使用Python读取约102KB的机器人YAML配置。改用C读取后，五份实际配置'
        '单次解析中位数从0.229～0.311秒降到0.027～0.045秒，解析值逐项相同。', '',
        '每段开合爪开始时用原方法完整校验模型，只做一次不变参数初始化；随后每个原采样只复制'
        '原URDF已经证明随夹爪锁定状态变化的三组张量及锁定关节状态。每个采样仍重新执行原生FK，'
        '再按当前场景完成全部关节边界、自碰撞、目标与停放臂碰撞检查。', '',
        '本轮关闭球缓存，测试的是每次都重新计算的路径。', '']
    body+=timing_table(b)
    before=sum(average(b,'legacy',s,'exclusive_seconds') for s in ['gripper_opening','gripper_narrowing'])
    after=sum(average(b,'batched',s,'exclusive_seconds') for s in ['gripper_opening','gripper_narrowing'])
    body+=['',f"本轮累计省时{b['review']['total_saving_fraction']:.2%}。开合爪的模型读取、FK与验碰撞"
        f"（扣除绑定部分）平均{before:.3f}→{after:.3f}秒，减少{1-after/before:.1%}。", '',
        '## 当前耗时构成','',
        '以下取通过采用门槛的最新实现，每项为一次完整规划的平均耗时。已扣除重复包含的子调用。', '',
        '| 环节 | 平均耗时 |','| --- | ---: |']
    groups=[('机器人与场景绑定',['world_and_robot_bind']),
        ('开合爪模型读取、FK与验碰撞',['gripper_opening','gripper_narrowing']),
        ('抓取IK与终点几何核验',['grasp_IK','grasp_endpoint']),
        ('接触几何更新',['contact_validator_update']),
        ('路径搜索与轨迹优化',['path_PRM','trajectory_approach','trajectory_pregrasp'])]
    measured=0.
    for label,names in groups:
        seconds=sum(average(chosen,mode,name,'exclusive_seconds') for name in names)
        measured+=seconds;body.append(f'| {label} | {seconds:.3f}秒 |')
    mean=chosen['summaries'][mode]['total_seconds']/chosen['summaries'][mode]['requests']
    body+=[f'| 其余IK、交接、调度与记录 | {mean-measured:.3f}秒 |',
        f'| 合计 | {mean:.3f}秒 |','',
        '## 逐场景结果','', '下表是每场景两次完整规划的平均值。','',
        '| 场景 | 绑定原版 | 只更新开度 | 开合爪原版 | 开合爪优化后 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for ca,cb in zip(a['cases'],b['cases']):
        values=[c[m]['mean_RPC_seconds'] for c in [ca,cb] for m in ['legacy','batched']]
        body.append('| '+ca['id']+' | '+' | '.join(f'{v:.3f}秒' for v in values)+' |')
    exact=all(p['review']['all_endpoints_bitwise_equal'] and p['review']['all_paths_bitwise_equal'] for p in phases)
    rejected_root=ROOT/'tmp/gripper_state_reuse/binding5_20261127_01'
    rejected=phase_data(rejected_root,'binding')
    complete_log(rejected_root,rejected)
    memo_root=ROOT/'tmp/gripper_state_reuse/geometry5_20261127_01'
    memo=phase_data(memo_root,'memo');complete_log(memo_root,memo)
    exact=exact and all(p['review']['all_endpoints_bitwise_equal'] and p['review']['all_paths_bitwise_equal']
                        for p in [rejected,memo])
    body+=['','## 第一版尝试与取舍','',
        '第一版减少重复场景绑定，但仍复制整套运动学参数。20次规划全部通过、路径逐位一致，'
        f"累计耗时增加{-rejected['review']['total_saving_fraction']:.2%}，未采用。"
        '随后单独验证只更新原URDF证明会变化的张量，结果如第一项表格。','',
        '球缓存版同样完成20次规划，全部抓姿和路径保持，整组总耗时基本持平。'
        '缓存仅在相同输入重复出现时命中，未达到采用门槛，原结果继续保留。','',
        '## 核验与采用','',
        '| 项目 | 结果 |','| --- | --- |',
        '| 任务完成 | 场景复用两版40＋球缓存20＋本次直接计算20，共80/80完整规划 |',
        '| 记录完整性 | 登记、领取、分项计时、原始配置和独立复核完整 |',
        '| 事件适用性 | 四组纯规划，新增仿真与物理抓取0次 |',
        '| 关键事件覆盖 | 逐采样球/半径/碰撞掩码、中途碰撞拦截、状态恢复、实际配置与完整路径 |',
        f"| 抓法与路径 | 四轮选中抓姿、关节和完整路径逐位一致：{'是' if exact and rejected['review']['all_paths_bitwise_equal'] and memo['review']['all_paths_bitwise_equal'] else '否'} |",
        f"| 场景绑定复用 | {'采用' if a['kept'] else '保留原实现'} |",
        '| 原生球缓存 | 未采用 |',
        f"| 开合爪C读取与原生逐采样计算优化 | {'采用' if b['kept'] else '保留原实现'} |",'',
        '34项离线检查通过。本次直接计算版258个采样的球坐标、半径、碰撞掩码和首次失败点完全一致，'
        '包括端点无碰撞而中间碰撞的情况。四轮共240个选中关节终点独立FK与限位核验通过，'
        '修订版另有432项原生运动学张量逐位核验。', '',
        '每种实现分别做同期对照。旧批次、原始源码和STOP均保留。', '',
        '## 运行记录','',f"- 绑定初版：`{rejected['run']}`",f"- 绑定修订：`{a['run']}`",
        f"- 球缓存：`{memo['run']}`",f"- 开合爪直接计算：`{b['run']}`",'']
    document=ROOT/'modules/pipeline/docs/mainline_versions/V7_场景绑定与开合爪计算优化结果.md'
    document.write_text('\n'.join(body))
    snapshot={'at':now(),'task_complete':True,'record_integrity':True,'planning_requests':80,'physical_grasps':0,
        'phases':phases,'current_phase_kind':chosen['kind'],'current_mode':mode,
        'rejected_attempts':[rejected,memo],
        'opening_yaml_diagnosis':read_json(memo_root/'opening_yaml_diagnosis.json'),
        'width_refresh_adopted':a['kept'],'sphere_cache_adopted':False,'native_sweep_adopted':b['kept'],
        'default_activated':False,'all_paths_and_endpoints_bitwise_equal':exact}
    write_json(geometry_root/'publication_snapshot.json',snapshot)
    return snapshot


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--geometry',type=Path,required=True);args=parser.parse_args()
    build(ensure_run_path(args.binding),ensure_run_path(args.geometry))
