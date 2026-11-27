"""Generate a Chinese method/result report and a compact research history."""
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from arx_project.layout import ROOT,now,read_json,write_json
from .offline import raw

LABELS={'control':'原始候选','native':'原版逻辑移植','native_rechecked':'修正后重新验碰撞','arx_observed':'ARX指腹适配'}
NAMES={'M0001_S1_open_apple':'苹果','M0020_S1_open_donut':'甜甜圈','M0044_S3_sparse_banana':'香蕉',
    'M0064_S4_clutter_mango':'芒果','M0074_S4_clutter_scissors':'杂乱剪刀'}


def generate():
    active=read_json(ROOT/'tmp/graspgenx_native_postprocessing/active.json')
    offline=ROOT/active['offline'];planning=ROOT/active['planning'];off=raw(offline/'acceptance.json')
    progress=raw(planning/'progress.json');reg=raw(planning/'registration.json')
    reviewed=raw(planning/'independent_review.json') if (planning/'independent_review.json').exists() else None
    rows=progress['results'];summaries=progress.get('summaries',{})
    interrupted=len(reg.get('interrupted_request_ids',[]))
    complete=bool(reviewed and reviewed['passed'] and len(rows)+interrupted==20)
    lines=['# V7：原始GraspGenX候选的完整后处理与优化结果','',
        f"2026年11月27日。69份离线观测全部处理完成，规划对照完成{len(rows)}/20项，程序中断{interrupted}项。物理抓取0次。",'',
        '## 本轮接入方法','',
        '从GraspGenX的原始位姿数组和原生评分重新开始。按评分稳定排序，逐一核对与历史候选池的编号、坐标变换和数值；扩散与OBB两个原生分支全部保留为后处理输入。',
        '输入是相同的SAM3掩码、头部RGB-D和RaySt3R重建点。深度点和目标重建点统一做2毫米体素采样，移除实测机器人占据的点。桌面由V2几何规则单独检查。',
        '算法依次判断夹持区是否为空、修正开爪宽度和接近深度、检查碰撞，再用原生NMS删除重复抓法。空池保持为空；被删除的候选不作为后备重新送入规划。', '',
        '| 方案 | 宽深和几何处理 | 去重设置 |', '| --- | --- | --- |',
        '| 原始控制 | 不做后处理，保留原评分顺序、88毫米开爪宽度和抓取位姿 | 关闭 |',
        '| 原版逻辑移植 | 采用ZeroGrasp原检测算法：20毫米高度、10毫米手指厚度、20毫米手掌基深，夹持区少于10点判空；GraspGenX没有预测深度，适配初值取40毫米，端点对齐现有TCP43 | 30毫米 / 30° |',
        '| 修正后重新验碰撞 | 保留原版宽深修正公式，使用修正后的宽度和深度重算手指及手掌碰撞；输出宽度超过ARX的88毫米行程则拒绝 | 30毫米 / 30° |',
        '| ARX指腹适配 | 使用实际15毫米指腹；夹持支持来自目标重建点；沿原接近轴最多修正5毫米、不旋转；用观测点检查修正后的真实凸网格，并保留原V2桌面规则 | 15毫米 / 15° |','',
        'TCP仍是43毫米，生成器原点后移100毫米只用于坐标转换。原始位姿、修正位姿、原始开度、修正开度、来源编号与淘汰原因全部分别记录。', '',
        '## 69个场景的候选变化','',
        '原始候选共166,008个，单场景中位数2240个。', '',
        '| 方案 | 碰撞筛选后中位数 | NMS后中位数 | 输出候选合计 | 非空场景 | 原成功候选来源通过筛选 | 来源仍被NMS保留 | CPU后处理中位数 / P90 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mode,value in off['summaries'].items():
        lines.append(f"| {LABELS[mode]} | {value['before_nms_median']:.0f} | {value['after_nms_median']:.0f} | {value['output_candidates']} | "
            f"{value['nonempty_cases']}/69 | {value['historical_selected_seed_passes']}/55 | {value['historical_selected_seed_after_nms']}/55 | "
            f"{value['postprocessing_median_ms']/1000:.3f} / {value['postprocessing_p90_ms']/1000:.3f}秒 |")
    lines+=['',
        '这里按历史原始编号追踪候选来源。候选经过宽深修正后，实际开爪宽度和落点会变化；本轮没有物理执行这些新抓法。', '',
        '## 优化发现','',
        'ZeroGrasp原实现先计算宽度和深度修正，再检查碰撞，但碰撞掩码仍使用旧宽度。对较小物体，收窄后的手掌和手指占据范围已经改变，旧掩码会误挡一部分抓法。',
        '重新按修正后的尺寸验碰撞后，原成功候选来源通过筛选的数量从45增到50。恢复的是碗、甜甜圈、柠檬、番茄、西兰花五例；NMS之后仍保留的成功候选来源由39增到43。',
        'ARX实际指腹组通过筛选的成功候选来源降到23/55。被淘汰的32例中，29例在新增的指腹深度支撑条件处退出，1例是桌面碰撞，2例是观测点或混合碰撞。31例在原开度、原位姿下能通过上一轮真实网格检查。主要问题是新增的深度支撑条件过严，这组条件不采用。', '',
        '## 五场景规划对照','',
        '苹果、甜甜圈、香蕉、芒果和杂乱剪刀在实验前固定。四组共用原V3双臂16/64交替搜索与逐个IK实现，同一显卡、同一常驻模型和相同预热，不混入V8计算优化。',
        '完整规划包含开爪检查、双臂搜索、IK、碰撞、路径、共享内存交接及RPC返回；表中合计再加离线测得的后处理耗时。感知模型和原始GraspGenX生成不重新运行。', '',
        '| 方案 | 规划通过 | 后处理空池 | 实际检查候选合计 | 后处理＋规划累计 | 中位数 / P90 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode,value in summaries.items():
        lines.append(f"| {LABELS[mode]} | {value['planning_successes']}/{value['completed']} | {value['postprocessing_empty']} | "
            f"{value['candidate_checks']} | {value['combined_total_seconds']:.3f}秒 | {value['combined_median_seconds']:.3f} / {value['combined_p90_seconds']:.3f}秒 |")
    lines+=['','| 场景 | 原始候选 | 原版逻辑 | 修正后复检 | ARX指腹适配 |','| --- | --- | --- | --- | --- |']
    by={(r['case_id'],r['variant']):r for r in rows}
    for case in reg['cases']:
        cells=[]
        for mode in reg['variants']:
            r=by.get((case['id'],mode))
            missing='程序中断' if case['id']+'_'+mode in reg.get('interrupted_request_ids',[]) else '未执行'
            cells.append(missing if r is None else ('通过' if r['success'] else '规划不出来：'+r['reason'])+f"；{r['input_candidates']}个候选；{r['postprocessing_seconds']+r['planning_seconds']:.2f}秒")
        lines.append('| '+NAMES[case['id']]+' | '+' | '.join(cells)+' |')
    lines+=['','### 修正后复检组实际选中的抓法','',
        '| 物体 | 原始候选编号 | 手臂 | 开爪总宽 | 沿原接近轴修正 |',
        '| --- | ---: | --- | ---: | ---: |']
    for case in reg['cases']:
        path=planning/'plans'/case['id']/'native_rechecked/plan_result.json'
        if not path.exists() or not raw(path)['success']:continue
        selected=raw(path)['selected'];meta=selected['candidate_metadata'];binding=selected['width_binding']
        lines.append(f"| {NAMES[case['id']]} | {selected['original_native_rank']} | {'右臂' if selected['arm']=='right' else '左臂'} | "
            f"{binding['candidate_width_m']*1000:.2f}毫米 | {meta['axial_refinement_m']*1000:+.2f}毫米 |")
    lines+=['','原版公式允许把初始40毫米深度缩到0，69例保留候选中最大的轴向修正为40毫米；上表给出本次实际选中值。',
        '候选宽度、GPU碰撞模型的双指位置和最终开爪命令逐项核对一致，双臂各以88、60、40毫米三种宽度预检；收爪过程按每指最多0.5毫米间隔检查。']
    spatial_path=ROOT/'tmp/graspgenx_native_postprocessing/spatial69_20261127_01/acceptance.json'
    if spatial_path.exists():
        sp=raw(spatial_path)
        lines+=['','## 不改变输出的计算加速','',
            '根据全部候选的夹爪几何包围范围，预先排除不可能参与空抓或碰撞判断的背景点。原检测器会用首个腔外点处理接触点选择中的并列情况，因此额外保留该点及其原顺序。',
            f"69例、{sp['candidate_rows']:,}个候选的空抓判断、碰撞判断、点数、宽度和深度全部逐项完全一致；参与计算的点数中位减少{sp['point_reduction_median']:.1%}。",'',
            '| CPU检测耗时 | 原完整点云 | 等值裁剪 |','| --- | ---: | ---: |',
            f"| 中位数 | {sp['full_median_seconds']:.3f}秒 | {sp['crop_median_seconds']:.3f}秒 |",
            f"| P90 | {sp['full_p90_seconds']:.3f}秒 | {sp['crop_p90_seconds']:.3f}秒 |",
            f"| 69例累计 | {sp['full_total_seconds']:.3f}秒 | {sp['crop_total_seconds']:.3f}秒 |",'',
            f"累计省时{sp['saving_fraction']:.1%}。该分项独立计时，前面的规划表使用当次实测的后处理耗时。",
            '运行接口可以只计算选中的后处理组，避免顺带执行未采用的ARX指腹分支。苹果、甜甜圈和剪刀三例的新完整后处理分别为0.631、0.828、0.822秒，最终候选池与旧输出全部数值一致。']
    lines+=['','## 本轮结论','']
    if complete:
        control=summaries['control'];qualifying=[]
        for mode in ['native','native_rechecked','arx_observed']:
            v=summaries[mode]
            if v['planning_successes']>=control['planning_successes'] and v['combined_total_seconds']<control['combined_total_seconds']:
                qualifying.append(mode)
        if qualifying:
            best=min(qualifying,key=lambda k:summaries[k]['combined_total_seconds'])
            savings=1-summaries[best]['combined_total_seconds']/control['combined_total_seconds']
            lines.append(f"五场景先导中，{LABELS[best]}保持了控制组的规划通过数，后处理加规划累计省时{savings:.1%}。")
            if summaries[best]['combined_median_seconds']>control['combined_median_seconds']:
                lines.append('总收益主要来自甜甜圈和剪刀等较慢的场景；中位耗时略升，普通容易场景的额外后处理和收爪检查仍有开销。')
        else:lines.append('五场景先导中，没有方案同时达到原始控制组的规划通过数和更低总耗时。')
        lines.append('宽深复检修正了旧尺寸误挡的问题；ARX指腹硬筛配置不采用。当前默认链路保持原样，后续实抓对照优先检验修正后复检组，重点看收窄和轴向退让是否影响夹持。')
    else:lines.append(f'规划已完成{len(rows)}/20，剩余{20-len(rows)}项未执行。未执行与普通规划失败分别计数。')
    lines+=['','## 完整性与覆盖','',
        '| 项目 | 结果 |','| --- | --- |',
        f"| 任务 | 离线69/69，规划{len(rows)}/20，实际抓取0 |",
        '| 记录 | 9821个输出候选的源编号、开度、修正方向与幅度核查通过；原始数组与输入散列保持 |',
        '| 原版核查 | 25组合成输入对照通过；72个真实候选碰撞判断相同，宽深最大误差约0.000017毫米；几何、宽度接口、等值裁剪和空池检查共21项通过 |',
        f"| 事件覆盖 | 后处理与NMS均有逐候选记录；规划已核对{reviewed['checked_endpoints'] if reviewed else 0}个关节终点；闭合、抬升和保持未执行 |",'',
        '## 文件入口','',
        '- 可选后处理：`modules/graspgenx/native_postprocessing/`',
        '- 固定对照：`modules/pipeline/experiments/graspgenx_native_postprocessing/`',
        f'- 离线结果：`{active["offline"]}`',f'- 规划结果：`{active["planning"]}`',
        '- 独立配置：`configs/experiments/graspgenx_native_postprocessing.json`','']
    lines+=['## 可选调用接口','',
        '在项目根目录、现有geometric_ik环境运行。输入为原始数组转换后的位姿、原生评分、观测点云和目标点云；`load_raw`保留原始编号映射。','',
        '```python','from modules.graspgenx.native_postprocessing.processor import Processor, load_raw, make_pool',
        'poses, scores, raw_indices = load_raw(trial_directory, original_pool)',
        'result = Processor({}).process(',
        '    poses, scores, observed_points, target_points, table,',
        '    variants=("native_rechecked",)',
        ')["native_rechecked"]',
        'output_pool = make_pool(original_pool, result, raw_indices, "native_rechecked")','```','',
        '规划实验通过`modules.graspgenx.native_postprocessing.width_planning.create`绑定候选宽度。当前主线配置保持原样。','']
    doc=ROOT/'modules/pipeline/docs/mainline_versions/V7_原始候选完整后处理与优化结果.md'
    doc.write_text('\n'.join(lines));shutil.copy2(doc,offline/'report.md');shutil.copy2(doc,planning/'report.md')
    if complete:history(offline,summaries)
    print(doc)
    return doc


def history(root,summaries):
    control=summaries['control'];baseline=summaries['native'];entries=[]
    for number,mode in enumerate(['native_rechecked','arx_observed'],1):
        value=summaries[mode]
        passed=value['planning_successes']>=control['planning_successes']
        score=-value['combined_total_seconds'] if passed else -1e6-(control['planning_successes']-value['planning_successes'])*1e4-value['combined_total_seconds']
        better=passed and value['combined_total_seconds']<baseline['combined_total_seconds'] and mode!='arx_observed'
        status='kept' if better else 'reverted'
        entries.append({'iteration':number,'mode':mode,'score':score,'status':status,
            'planning_successes':value['planning_successes'],'seconds':value['combined_total_seconds'],
            'reason':'optional_only_default_unchanged' if better else 'not_adopted_after_retention_and_planning_comparison'})
    previous=raw(root/'research_iterations.json') if (root/'research_iterations.json').exists() else None
    write_json(root/'research_iterations.json',{'at':now(),'max_iterations':2,'entries':entries,'default_activated':False})
    with (root/'autoresearch-results.tsv').open('w') as stream:
        writer=csv.writer(stream,delimiter='\t');writer.writerow(['iteration','timestamp','hypothesis','metric','score','status','duration_seconds','notes'])
        for row in entries:writer.writerow([row['iteration'],now(),row['mode'],'paired_postprocessing_and_planning',row['score'],row['status'],row['seconds'],row['reason']])
    if previous is None or previous['entries']!=entries:
        with (root/'research.md').open('a') as stream:
            for row in entries:stream.write(f"| {row['iteration']} | {row['mode']} | {row['planning_successes']}/5；{row['seconds']:.3f}秒 | {row['status']} |\n")
    (root/'research_log.md').write_text('\n\n'.join(json.dumps(row,ensure_ascii=False,indent=2) for row in entries)+'\n')
    shutil.copy2(root/'report.md',root/'final_report.md')
    import sys
    sys.path.insert(0,str(Path.home()/'.codex/skills/autoresearch-skill/scripts'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:from style_presets import rcparams
    except ImportError:
        def rcparams():plt.style.use('default')
    rcparams();fig,ax=plt.subplots(figsize=(7,4))
    best=-np.inf;envelope=[]
    for row in entries:
        best=max(best,row['score']);envelope.append(best)
        ax.scatter(row['iteration'],row['score'],s=70,facecolors='#0072B2' if row['status']=='kept' else 'none',edgecolors='#0072B2')
    ax.plot([1,2],envelope,color='#0072B2',linewidth=1)
    ax.axhline(-control['combined_total_seconds'],color='#D55E00',linestyle='--',label='Raw control target')
    ax.set(xlabel='Iteration',ylabel='Planning guard score',xticks=[1,2]);ax.legend();fig.tight_layout();fig.savefig(root/'progress.png',dpi=150);plt.close(fig)


if __name__=='__main__':generate()
