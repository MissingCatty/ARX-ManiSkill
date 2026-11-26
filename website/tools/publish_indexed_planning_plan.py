"""Publish V8 methods and recorded planning progress with standalone resources."""
import shutil

from arx_project.layout import ROOT,read_json,write_json,sha256


def planning_progress(website, config, relative=None):
    """Copy a compact progress snapshot; never start or reconnect a planner."""
    relative = relative or config['execution'].get('active_run')
    if not relative:
        return None
    run = (ROOT/relative).resolve()
    if not run.is_relative_to(ROOT/'tmp/indexed_grasp_planning'):
        raise ValueError('V8 publication must read its registered planning directory')
    registration = read_json(run/'registration.json')
    progress = read_json(run/'progress.json')
    acceptance = read_json(run/'acceptance.json') if (run/'acceptance.json').is_file() else None
    snapshot = {key: progress.get(key) for key in ['at','phase','completed_requests','total_requests','current']}
    binding = registration.get('experiment') == 'binding_reuse'
    component = 'binding_reuse_implementation' if binding else 'native_batch_implementation'
    snapshot.update(run=relative,gpu=registration['gpu'],cases=len(registration['cases']),
        experiment='binding_reuse' if binding else 'native_batch',variants=registration['variants'],
        physical_grasps=0,offline_checks=config.get(component,{}).get('offline_checks',0),
        model_warmup_complete=(run/'startup.json').is_file(),stopped=(run/'STOP').is_file())
    snapshot['results'] = [{'id': row['id'], 'variants': {mode: {key: value.get(key) for key in
        ['seconds','success','selected_arm','selected_source_index','candidate_checks',
         'binding_seconds','full_world_rebinds','binding_cache_hits']} for mode,value in row['variants'].items()},
        'comparison':row['comparison']} for row in progress.get('results',[])]
    if acceptance:
        snapshot.update(summaries=acceptance['summaries'],
            control_successes_retained=acceptance['control_successes_retained'],
            record_integrity=acceptance['record_integrity'])
    if (run/'independent_review.json').is_file():
        checked = read_json(run/'independent_review.json')
        snapshot.update(independent_review_passed=checked['passed'],
            independent_endpoints=checked['checked_endpoints'])
    name = 'sam3-v8-'+run.name.replace('_','-')+'-status.json'
    target = website/'resources/media'/name
    if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
        raise ValueError('V8 progress must be a standalone website resource')
    write_json(target,snapshot)
    return snapshot,name


def result_table(snapshot, title):
    labels = {'control':'现行逐个检查','batch16':'批量16个','batch32':'批量32个',
        'reuse':'左右臂上下文复用'}
    lines = ['## '+title, '',
        '| 方法 | 完整规划次数 | 规划通过 | 中位耗时 | P90耗时 | 合计耗时 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode,value in snapshot['summaries'].items():
        lines.append(f"| {labels[mode]} | {value['requests']} | {value['successes']} | "
            f"{value['median_seconds']:.2f}秒 | {value['p90_seconds']:.2f}秒 | {value['total_seconds']:.2f}秒 |")
    lines += ['', '表中时间是一整个场景返回完整规划的耗时。批量16表示每批最多16个目标；'
        'P90按完整请求统计，包含双臂搜索、开爪检查、IK、碰撞与路径。']
    return '\n'.join(lines)


def publish():
    website = ROOT/'website'
    config = read_json(ROOT/'configs/experiments/indexed_grasp_planning.json')
    if config['execution']['physical_grasp_budget']:
        raise ValueError('This publisher describes planning-only V8 comparisons')
    review = ROOT/'tmp/indexed_grasp_planning/research_20261126_01'
    site_path,registry_path = website/'内容/网站.json',website/'内容/experiments.json'
    site,registry = read_json(site_path),read_json(registry_path)
    attachments = [
        (ROOT/'configs/experiments/indexed_grasp_planning.json','sam3-v8-indexed-planning-config.json'),
        (ROOT/'modules/pipeline/docs/mainline_versions/V8_可达性索引与批量IK加速实验方案.md','sam3-v8-indexed-planning-plan.md'),
        (ROOT/'modules/pipeline/docs/mainline_versions/V8_规划加速方法调研.md','sam3-v8-planning-research.md'),
        (ROOT/'modules/pipeline/docs/mainline_versions/V8_原生批量IK实施.md','sam3-v8-native-batch-method.md'),
        (ROOT/'modules/pipeline/docs/mainline_versions/V8_原生批量IK完整对照结果.md','sam3-v8-native-batch-final.md'),
        (ROOT/'modules/pipeline/docs/mainline_versions/V8_上下文复用实施.md','sam3-v8-binding-reuse-method.md'),
        (review/'arm_geometry_review.json','sam3-v8-arm-geometry-review.json'),
        (review/'source_review.json','sam3-v8-source-review.json')]
    for source,name in attachments:
        target = website/'resources/media'/name
        if target.is_symlink():raise ValueError('Website resource must be a real copy')
        shutil.copy2(source,target)
        if target.stat().st_nlink != 1 or sha256(source) != sha256(target):
            raise ValueError('Independent resource copy failed')
        site['localAssets'][name] = '../../resources/media/'+name
    before = ROOT/'tmp/indexed_grasp_planning/ik_rows_20261126_03/results.json'
    after = ROOT/'tmp/indexed_grasp_planning/ik_rows_20261126_04/results.json'
    if before.is_file() and after.is_file():
        proof = {'original_interface':read_json(before),'per_target_early_return':read_json(after),
            'validation':read_json(after.with_name('complete.json')),'physical_grasps':0}
        proof_name = 'sam3-v8-batch-rows-validation.json'
        write_json(website/'resources/media'/proof_name,proof)
        site['localAssets'][proof_name] = '../../resources/media/'+proof_name
    runtime = planning_progress(website,config)
    native_run = config.get('native_batch_implementation',{}).get('full_run')
    native_runtime = (planning_progress(website,config,native_run) if native_run else None)
    if native_run and (ROOT/native_run/'independent_review.json').is_file():
        name = 'sam3-v8-native-batch-fk-review.json'
        source = ROOT/native_run/'independent_review.json'
        target = website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
            raise ValueError('Independent FK attachment must be a real standalone copy')
        shutil.copy2(source,target)
        if sha256(source)!=sha256(target):raise ValueError('FK attachment copy mismatch')
        site['localAssets'][name] = '../../resources/media/'+name
    body = '''## 本版进度

方法调研、本地接口与ARX静态几何核查已完成，下一步适配并比较各方法。当前V8规划回放0次、物理抓取0次。

## 优化目标

把候选检查与整次规划请求做快，同时保留原抓姿、评分顺序和规划效果。以已启用的双臂交替搜索为基线，分别测量批量IK、几何IK、地图复用和绑定复用，最后组合有效部分。

双臂交替搜索的香蕉428.23→8.55秒已归入[V3双臂版本](/iterations/sam3-graspgenx/versions/V3/)。V8各组都使用相同的首轮16、后续64调度，比较在此基础上还能节省多少时间。

## 优先比较什么

| 分组 | 具体做法 | 想解决的耗时 | 当前进度 |
| --- | --- | --- | --- |
| A：现行基线 | 当前交替搜索、逐候选IK与原碰撞/路径检查 | 建立优化后的真实对照 | 已实现，待完整回放 |
| B：原生批量IK | 一次处理16或32个目标，各自保留编号与结果 | 逐个调用小规模求解的开销 | 待适配 |
| C：绑定复用 | 同一请求内复用每臂的模型和场景上下文 | 交替窗口反复绑定世界 | 待适配与测显存 |
| D：几何IK | 用EAIK/IK-Geo生成关节分支，再走当前检查 | 六轴数值搜索 | 结构已核查，求解待验证 |
| E：索引复用 | 地图见证直接验证或作为原目标初值 | 重复寻找已知解 | 待扩展接口 |
| F：有效组合 | 组合单项测得有效的部分 | 最终请求总耗时 | 待各组结果 |

批量IK与地图分开测试，便于看清各自的收益。所有组都按原候选排名决定哪个完整方案先取得执行权。

批量求解同时保留完整场景IK和纯运动学回退两层流程，后者继续经过独立FK、限位与V2近桌检查。

## 本地已经确认的条件

| 核查项 | 结果 | 下一步 |
| --- | --- | --- |
| cuRobo接口 | 已有批量目标与种子输入；当前辅助IK配置一次1个目标 | 预热16、32目标的独立求解器 |
| ARX运动链 | right_joint22/23/24三个连续转轴平行，腕部25/26轴相交 | 验证EAIK自动分解、原限位与工具变换 |
| 关节见证 | 已有约1056万组前方记录 | 先测命中率和查询/回退合计耗时 |
| 现有地图查询 | 单目标取8192个位置近邻，再比较旋转 | 测量成批查询成本 |
| 现有朝向索引 | 用重力方向做允许底盘转向时的必要条件筛选 | 固定站位查询加入完整旋转比较 |

16/64是两只手的轮换窗口，16/32是GPU一次处理的目标数量，两者分别记录。几何求解返回的角度先转换到原限位允许的等价分支，再独立复算FK；最小二乘结果也按原2毫米、0.03弧度门槛检查。

## 地图怎样接

查询原抓姿附近的关节见证，达到原目标容差的结果进入当前场景碰撞检查；邻近记录作为原目标的求解初值。缺少合格记录时调用现有IK。

地图负责提供运动学解，场景碰撞和完整路径继续检查。先测现有地图的收益，再按具体缺口决定是否补数据。成功解缓存与当前场景碰撞结果分别管理，首轮比较使用冻结缓存。

## 已有问题定位数据

交替优化前的65个配对规划场景中，GraspGenX两臂合计检查12452次抓取终点。下表是候选检查次数：

| 检查结果 | 次数 | 比例 |
| --- | ---: | ---: |
| 当次未得到运动学解 | 10878 | 87.4% |
| 有解后碰撞检查未通过 | 1447 | 11.6% |
| 抓取终点通过 | 127 | 1.0% |

当前交替搜索的完整69份观测耗时将在V8重新测量。地图主要帮助已有解的复用；批量求解也需要覆盖长串当次未解出的候选。

## 怎样实施和比较

| 顺序 | 工作 | 当前进度 |
| --- | --- | --- |
| 1 | 更新基线，核查文献、接口和ARX几何 | 已完成 |
| 2 | 冻结现行交替输入，建立分项计时 | 待实施 |
| 3 | 原生批量IK与绑定复用分别适配 | 待实施 |
| 4 | 几何IK适配与FK、限位、多分支检查 | 待实施 |
| 5 | 地图批量查询与见证/初值对照 | 待实施 |
| 6 | 全69份观测配对回放，筛选有效组合 | 待登记批次 |
| 7 | 按规划结果确定物理抓取验证 | 待确定 |

使用相同的69份V6保存观测、GraspGenX原候选和机器人状态。5例共同站位失败在74例总体中保留，SAM失败与原中断单列。同卡、统一预热，交替先后顺序；短时计时多次重复。

总请求耗时统计中位数和P90，并拆出解码、查询、世界绑定、开爪、IK、碰撞、路径与交接。规划通过率、原成功输入保留、选中抓法和手臂变化一并记录。每臂871球、TCP43、指尖2毫米规则及场景物理沿用当前配置。

## 接入位置

头部RGB-D → SAM3 → RaySt3R → GraspGenX原候选 → 双臂交替调度 → 本版待比较的IK/复用模块 → 当前碰撞与路径检查 → 选臂抓取。

[[流程卡片]]

## 参考方法与文件

- [cuRoboV2批量IK](/papers/curobo-v2/)
- [MoveIt Cached IK](/papers/cached-ik/)
- [EAIK / IK-Geo几何IK](/papers/eaik/)
- [VAMP-MR向量化检查](/papers/vamp-mr/)
- [完整实施方案](/media/sam3-v8-indexed-planning-plan.md)
- [方法调研](/media/sam3-v8-planning-research.md)
- [实验配置](/media/sam3-v8-indexed-planning-config.json)
- [ARX静态几何核查](/media/sam3-v8-arm-geometry-review.json)
- [本地接口与来源记录](/media/sam3-v8-source-review.json)
'''
    status = '调研与静态核查完成，待适配'
    summary = '以现行交替搜索为基线，独立比较批量IK、绑定复用、几何IK和地图；V8回放0次。'
    if runtime:
        snapshot,name = runtime
        done,total = snapshot['completed_requests'],snapshot['total_requests']
        binding = snapshot['experiment']=='binding_reuse'
        label = 'C：上下文复用' if binding else 'B：原生批量IK'
        status = ('纯规划对照已完成' if snapshot['phase']=='complete' else
            '纯规划对照停止，保留记录' if snapshot['stopped'] else
            '纯规划对照进行中' if snapshot['model_warmup_complete'] else '常驻模型预热中')
        status = label+'，'+status
        summary = f"{label}：{snapshot['cases']}份观测，规划{done}/{total}次，物理抓取0次。"
        body = body.replace('方法调研、本地接口与ARX静态几何核查已完成，下一步适配并比较各方法。当前V8规划回放0次、物理抓取0次。',
            summary)
        body = body.replace('## 优化目标', f'[本轮进度与逐例数据](/media/{name})\n\n## 优化目标',1)
        body = body.replace('## 本地已经确认的条件',
            '## 批量接口怎样保持逐目标检查\n\n'
            '先成批产生并检查原32个初值。每个目标已有合格解时先保留，只有未通过的行继续优化；这样其他失败目标不会冲掉该行已通过的初始解。补齐行不参与选择，返回解继续经过原碰撞、接触和路径检查。\n\n'
            '两种批次容量的混合目标检查与独立FK均通过。\n\n'
            '[实施说明](/media/sam3-v8-native-batch-method.md) · '
            '[接口验证数据](/media/sam3-v8-batch-rows-validation.json)\n\n## 本地已经确认的条件',1)
        if binding:
            body = body.replace('交替窗口反复绑定世界 | 待适配与测显存 |',
                '交替窗口反复绑定世界 | 已实现，'+status+' |')
            body = body.replace('原生批量IK与绑定复用分别适配 | 待实施 |',
                '原生批量IK与绑定复用分别适配 | B已验收，C对照中 |')
            section = ('## C怎样减少重复绑定\n\n'
                '左右臂分别保留开放和闭合上下文，在同一次完整请求内复用相同的模型和场景。'
                '纯运动学回退仍共享原求解器。复用前核对完整测得状态、实际网格和原始回执，'
                '并重新检查FK与起始碰撞。\n\n'
                '20项新增故障检查与原31项回归共51项通过。先用五份输入按ABBA/BAAB顺序做20次规划，'
                '确认抓法保留、重绑定减少、总耗时至少降低2%后，再比较全部69份观测。\n\n'
                '[上下文复用实施说明](/media/sam3-v8-binding-reuse-method.md)\n\n')
            if snapshot.get('summaries'):
                section += result_table(snapshot,'C上下文复用对照结果')+'\n\n'
            body = body.replace('## 优化目标',section+'## 优化目标',1)
        elif not native_runtime or not native_runtime[0].get('summaries'):
            body = body.replace('逐个调用小规模求解的开销 | 待适配 |',
                '逐个调用小规模求解的开销 | 已接通，独立对照中 |')
        site['localAssets'][name] = '../../resources/media/'+name
    if native_runtime and native_runtime[0].get('summaries'):
        native,name = native_runtime
        section = result_table(native,'B原生批量IK完整结果')
        section += ('\n\n三组各69/69通过，原抓姿和选臂全部一致。621个终点独立复算通过。'
            '批量16合计耗时减少32.84%，中位数降低29.12%；后续组合优先使用批量16。\n\n'
            '[完整结果与分析](/media/sam3-v8-native-batch-final.md) · '
            f'[69例逐项数据](/media/{name}) · '
            '[621个终点复算](/media/sam3-v8-native-batch-fk-review.json)\n\n')
        body = body.replace('## 优化目标',section+'## 优化目标',1)
        body = body.replace('建立优化后的真实对照 | 已实现，待完整回放 |',
            '建立优化后的真实对照 | 69份回放完成 |')
        body = body.replace('逐个调用小规模求解的开销 | 待适配 |',
            '逐个调用小规模求解的开销 | 207次对照完成，优先保留批量16 |')
        body = body.replace('冻结现行交替输入，建立分项计时 | 待实施 |',
            '冻结现行交替输入，建立分项计时 | B已完成，C增加逐次绑定记录 |')
        site['localAssets'][name] = '../../resources/media/'+name
    (website/'内容/迭代/sam3-graspgenx-v8.md').write_text(body)
    group = next(v for v in registry['experiments'] if v['id']=='sam3-graspgenx')
    version = next(v for v in group['iterations'] if v['code']=='V8')
    version.update(status=status,resultSummary=summary,
        change='以现行双臂交替搜索为基线，分别比较原生批量IK、几何IK、关节解索引与模型场景绑定复用，保留原抓姿和评分顺序。',
        pipeline='头部RGB-D → SAM3 → RaySt3R → GraspGenX原候选 → 双臂交替调度 → 批量IK/几何IK/索引复用对照 → 当前碰撞与路径检查 → 选臂抓取')
    version['settings'] = [['基线','已启用的双臂交替搜索，首轮16、后续64。'],
        ['对照','批量IK、绑定复用、几何IK、地图复用分别比较。'],
        ['输入','最终V6的69份有效观测，保留原抓姿与排名。'],
        ['当前进度',status]]
    if runtime:
        snapshot,name = runtime
        run = snapshot['run']
        batch = {'id':run.split('/')[-1],'title':('上下文复用纯规划对照' if snapshot['experiment']=='binding_reuse'
            else '原生批量IK纯规划对照'),'date':'2026-11-26',
            'status':status,'summary':summary,'outputs':[run],
            'artifacts':[{'label':'进度与逐例结果','url':'/media/'+name}]}
        version['batches'] = [v for v in version['batches'] if v['id']!=batch['id']]+[batch]
    if group['current']=='V8':
        entry = next(v for v in site['versions'] if v['id']=='sam3-graspgenx')
        entry.update(status='V8'+status,metric=summary)
    write_json(site_path,site);write_json(registry_path,registry)
    print('Published V8 methods and recorded progress; physical grasps remain zero.')


if __name__=='__main__':
    publish()
