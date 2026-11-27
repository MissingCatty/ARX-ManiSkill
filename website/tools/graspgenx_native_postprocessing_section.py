"""Publish V7's full hard filters without changing V8 or old NMS results."""
import json
import shutil
from arx_project.layout import ROOT,now,read_json,write_json,sha256

LABELS={'control':'原始GraspGenX候选','native':'ZeroGrasp原版逻辑移植',
    'native_rechecked':'宽度和深度修正后重新验碰撞','arx_observed':'ARX实际指腹适配'}


def section(website,site):
    pointer=ROOT/'tmp/graspgenx_native_postprocessing/active.json'
    if not pointer.exists():return '',None,None
    active=read_json(pointer);offline=ROOT/active['offline'];planning=ROOT/active['planning']
    accepted=read_json(offline/'acceptance.json');checked=read_json(offline/'independent_review.json')
    reg=read_json(planning/'registration.json');progress=read_json(planning/'progress.json')
    if progress.get('results') and not progress.get('summaries'):
        from modules.pipeline.experiments.graspgenx_native_postprocessing.planning import summarize
        progress['summaries']=summarize(progress['results'],reg['variants'])
    review=read_json(planning/'independent_review.json') if (planning/'independent_review.json').exists() else None
    status='完整后处理规划对照已完成' if review and review['passed'] else \
        '后处理完成，规划因资源或程序故障停止' if (planning/'STOP').exists() else '完整后处理的规划对照进行中'
    def attach(source,name):
        target=website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink!=1:
            raise ValueError('Native postprocessing resources must be independent copies')
        shutil.copy2(source,target)
        if sha256(source)!=sha256(target):raise ValueError('Resource copy mismatch')
        site['localAssets'][name]='../../resources/media/'+name
        return '/media/'+name
    rows=[json.loads((offline/'cases'/c['id']/'result.json').read_text()) for c in read_json(offline/'registration.json')['cases']]
    # Publish per-scene evidence and counts. Long per-candidate ID lists stay
    # in the experiment directory; they are not needed to render this page.
    for row in rows:
        for metrics in row['variants'].values():
            for key in ['original_ranks','before_nms_original_ranks','raw_generator_indices']:
                metrics.pop(key,None)
    snapshot=offline/'publication_snapshot.json'
    write_json(snapshot,{'at':now(),'offline':accepted,'offline_review':checked,'cases':rows,
        'planning':progress,'planning_review':review,'physical_grasps':0,'default_activated':False})
    data=attach(snapshot,'sam3-v7-native-postprocessing-results.json')
    lines=['## 原始GraspGenX候选的完整后处理','',status+'。', '',
        '本次从原始模型数组重新读入全部166,008个候选，覆盖69个场景。空抓检测、宽深修正、碰撞筛选后做NMS，淘汰的候选不再放回规划器。', '',
        '| 后处理方案 | 碰撞筛选后候选中位数 | NMS后候选中位数 | 有候选的场景 | 原成功候选来源通过筛选 | 仍被NMS保留 | CPU后处理中位数 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key,value in accepted['summaries'].items():
        lines.append(f"| {LABELS[key]} | {value['before_nms_median']:.0f} | {value['after_nms_median']:.0f} | "
            f"{value['nonempty_cases']}/69 | {value['historical_selected_seed_passes']}/55 | "
            f"{value['historical_selected_seed_after_nms']}/55 | {value['postprocessing_median_ms']/1000:.3f}秒 |")
    lines+=['','原始候选中位数为2240个。“原成功候选来源”按原始编号追踪；修正后的位姿、开爪宽度和保留状态逐项保存。','',
        '原版逻辑组以40毫米作为适配初始深度，宽深修正后按30毫米/30°去重。复检组用修正后的宽度和深度重算手指与手掌碰撞。',
        'ARX组使用实际15毫米指腹，沿原接近轴最多修正5毫米；目标点用于判断夹持区是否有物体，观测点用于碰撞检查，按15毫米/15°去重。', '',
        '## 五场景规划验证','',f"本轮完成{progress['completed_requests']}/20项，程序中断{len(reg.get('interrupted_request_ids',[]))}项，物理抓取0次。各组共用原V3双臂交替规划器，碰撞模型按候选宽度绑定。",'',
        '| 方案 | 规划通过 | 后处理空池 | 实际检查候选 | 后处理＋完整规划累计 | 中位数 / P90 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key,value in progress.get('summaries',{}).items():
        lines.append(f"| {LABELS[key]} | {value['planning_successes']}/{value['completed']} | {value['postprocessing_empty']} | "
            f"{value['candidate_checks']} | {value['combined_total_seconds']:.3f}秒 | "
            f"{value['combined_median_seconds']:.3f} / {value['combined_p90_seconds']:.3f}秒 |")
    if review:
        lines+=['',f"{review['checked_requests']}条规划记录和{review['checked_endpoints']}个关节终点独立核对通过。",'']
    lines += ['',f'[逐场景筛选、规划和延时数据]({data})','']
    spatial=ROOT/'tmp/graspgenx_native_postprocessing/spatial69_20261127_01/acceptance.json'
    if spatial.exists():
        accelerated=read_json(spatial)
        if not accelerated['passed'] or not accelerated['all_detector_outputs_exactly_equal']:
            raise ValueError('Spatial crop has not passed output equality')
        spatial_url=attach(spatial,'sam3-v7-spatial-postprocessing-results.json')
        lines+=['## 保持检测结果不变的计算加速','',
            '先计算全部候选夹爪可能接触的范围，排除范围外的背景点。保留原点序列中会影响接触点选择的首个背景点，使原算法的判断保持。', '',
            '| CPU碰撞检测 | 原完整点云 | 等值裁剪后 |','| --- | ---: | ---: |',
            f"| 中位数 | {accelerated['full_median_seconds']:.3f}秒 | {accelerated['crop_median_seconds']:.3f}秒 |",
            f"| P90 | {accelerated['full_p90_seconds']:.3f}秒 | {accelerated['crop_p90_seconds']:.3f}秒 |",
            f"| 69例累计 | {accelerated['full_total_seconds']:.3f}秒 | {accelerated['crop_total_seconds']:.3f}秒 |",'',
            f"166,008个候选的空抓判断、碰撞判断、点数、宽度和深度逐项完全一致，累计省时{accelerated['saving_fraction']:.1%}。上面的规划对照沿用本次实测后处理耗时。",'',
            f'[完整等值核验与逐例计时]({spatial_url})','']
    report=ROOT/'modules/pipeline/docs/mainline_versions/V7_原始候选完整后处理与优化结果.md'
    if report.exists():
        url=attach(report,'sam3-v7-native-postprocessing-report.md')
        lines+=[f'[完整方法、结果与分析]({url})','']
    if (planning/'STOP').exists():
        lines+=['规划未执行项按故障停止单列，不计为普通规划失败。','']
    batch={'id':planning.name,'title':'原始候选完整后处理与优化对照','date':'2026-11-27','status':status,
        'summary':f"69份离线输入已完成，规划{progress['completed_requests']}/20，物理抓取0。",
        'outputs':[active['offline'],active['planning']],'artifacts':[{'label':'完整筛选和规划数据','url':data}]}
    return '\n'.join(lines),status,batch
