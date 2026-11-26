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
    geometric = registration.get('experiment') == 'geometric_ik'
    indexed = registration.get('experiment') in ['index_lookup_diagnosis','index_seed_probe']
    component = ('index_reuse_implementation' if indexed else 'binding_reuse_implementation' if binding
        else 'geometric_ik' if geometric else 'native_batch_implementation')
    snapshot.update(run=relative,gpu=registration.get('gpu','CPU'),cases=len(registration['cases']),
        experiment=registration.get('experiment','native_batch'),variants=registration.get('variants',{}),
        physical_grasps=0,offline_checks=config.get(component,{}).get('offline_checks',0),
        model_warmup_complete=(run/'startup.json').is_file(),stopped=(run/'STOP').is_file())
    snapshot['results'] = [{'id': row['id'], 'variants': {mode: {key: value.get(key) for key in
        ['seconds','success','selected_arm','selected_source_index','candidate_checks',
         'binding_seconds','full_world_rebinds','binding_cache_hits']} for mode,value in row['variants'].items()},
        'comparison':row['comparison']} for row in progress.get('results',[])]
    if indexed:
        for key in ['completed_queries','total_queries','completed_IK_calls','total_IK_calls',
                    'completed_fixtures','total_fixtures']:
            if key in progress:snapshot[key]=progress[key]
        if acceptance:
            snapshot['acceptance']={k:v for k,v in acceptance.items()
                if k not in ['cases','direct_witnesses','seed_results']}
        if (run/'stop_summary.json').is_file():
            snapshot['stop_summary']=read_json(run/'stop_summary.json')
    elif acceptance:
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
        'reuse':'左右臂上下文复用','geometric':'几何IK优先，原IK回退'}
    lines = ['## '+title, '',
        '| 方法 | 完整规划次数 | 规划通过 | 中位耗时 | P90耗时 | 合计耗时 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode,value in snapshot['summaries'].items():
        lines.append(f"| {labels[mode]} | {value['requests']} | {value['successes']} | "
            f"{value['median_seconds']:.2f}秒 | {value['p90_seconds']:.2f}秒 | {value['total_seconds']:.2f}秒 |")
    lines += ['', '表中时间是一整个场景返回完整规划的耗时，包含双臂搜索、开爪检查、IK、碰撞与路径。'
        'P90同样按完整请求统计。']
    return '\n'.join(lines)


def lookup_table(snapshot):
    accepted=snapshot.get('acceptance')
    if not accepted:return ''
    import math
    lines=['## E地图近邻查询结果','','在69份保存观测中，重放原控制组实际检查过的10,322个候选。'
        '查询位置范围为2厘米，比较完整三维朝向。','',
        '| 查询角度范围 | 找到近邻的候选数 | 比例 |','| --- | ---: | ---: |']
    for angle,hits in accepted['neighborhood_hits_by_angle'].items():
        degrees=math.degrees(float(angle))
        lines.append(f"| {degrees:.2f}° | {hits:,} / {accepted['queries']:,} | {hits/accepted['queries']:.2%} |")
    lines+=['','1.72°用于邻近初值查询过于严格。本轮扩大到3°、5°、10°，取出附近的关节角，'
        '再让IK求解原抓取位姿。最终位置误差2毫米、朝向误差1.72°与原抓姿保持。','',
        f"直接满足原抓姿精度的地图记录命中{accepted['direct_hits']}次。"
        f"近邻查询中位耗时{accepted['neighborhood_timing']['median_ms']:.3f}毫秒；"
        f"旧8192位置近邻加多分支选择为{accepted['seed_query_timing']['legacy_8192_seconds']['median_ms']:.2f}毫秒。",'',
        '[地图索引实施说明](/media/sam3-v8-index-reuse-method.md)','']
    return '\n'.join(lines)


def seed_probe_table(snapshot):
    accepted=snapshot.get('acceptance')
    if not accepted:return ''
    labels={'control':'原IK初值','map3':'3°近邻初值','map5':'5°近邻初值','map10':'10°近邻初值'}
    lines=['## E近邻初值的实际IK对照','','同一261个原始终点，各组重复3次；每目标仍使用32个初值，'
        '地图最多提供其中8个。逐次核对初值实际进入原生LM，并独立复算返回关节解。','',
        '| 方法 | 求出合格解 | 使用地图初值的调用 | 中位耗时 | P90耗时 | 合计耗时 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode,value in accepted['summaries'].items():
        lines.append(f"| {labels[mode]} | {value['successes']}/{value['IK_calls']} | {value['map_seed_calls']} | "
            f"{value['median_ms']:.2f}毫秒 | {value['p90_ms']:.2f}毫秒 | {value['total_seconds']:.2f}秒 |")
    lines+=['','表中计时包含查询、初值交接和IK。这一项先比较原目标的运动学求解，'
        '是否值得接入完整规划，按本轮耗时及原成功保留情况确定。','']
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
    index_doc=ROOT/'modules/pipeline/docs/mainline_versions/V8_地图索引复用实施.md'
    if index_doc.is_file():attachments.append((index_doc,'sam3-v8-index-reuse-method.md'))
    funnel=config.get('funnel_proposal_review')
    if funnel:
        attachments.extend([(ROOT/funnel['document'],'sam3-v8-funnel-review.md'),
            (ROOT/funnel['review_run']/'review.json','sam3-v8-funnel-source-review.json')])
    diagnosis=None;diagnosis_run=None;funnel_active={}
    active=ROOT/'tmp/indexed_grasp_planning/funnel_active.json'
    if active.exists():
        funnel_active=read_json(active)
        diagnosis_run=(ROOT/funnel_active['run']).resolve()
        if not diagnosis_run.is_relative_to(ROOT/'tmp/indexed_grasp_planning'):
            raise ValueError('Unexpected V8 funnel diagnosis path')
        diagnosis=read_json(diagnosis_run/'acceptance.json')
        if not diagnosis['passed']:raise ValueError('V8 funnel diagnosis did not pass')
        attachments.extend([(diagnosis_run/'acceptance.json','sam3-v8-full69-funnel-diagnosis.json'),
            (ROOT/'modules/pipeline/docs/mainline_versions/V8_完整候选检查耗时定位.md','sam3-v8-full69-funnel-diagnosis.md')])
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
    binding_run=config.get('binding_reuse_implementation',{}).get('full_run')
    binding_runtime=(planning_progress(website,config,binding_run) if binding_run else None)
    geometric_run=config.get('geometric_ik',{}).get('accepted_run')
    geometric_runtime=(planning_progress(website,config,geometric_run) if geometric_run else None)
    lookup_run=config.get('index_reuse_implementation',{}).get('diagnosis_run')
    lookup_runtime=(planning_progress(website,config,lookup_run) if lookup_run else None)
    geometric_doc=ROOT/'modules/pipeline/docs/mainline_versions/V8_几何IK实施.md'
    if geometric_doc.is_file():
        name='sam3-v8-geometric-ik-method.md'
        target=website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink!=1:
            raise ValueError('Geometric method attachment must be an independent copy')
        shutil.copy2(geometric_doc,target);site['localAssets'][name]='../../resources/media/'+name
    contract_run=config.get('geometric_ik',{}).get('contract_run')
    if contract_run and (ROOT/contract_run/'acceptance.json').is_file():
        name='sam3-v8-geometric-contract.json'
        target=website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink!=1:
            raise ValueError('Geometric contract must be an independent copy')
        shutil.copy2(ROOT/contract_run/'acceptance.json',target)
        site['localAssets'][name]='../../resources/media/'+name
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
- [VAMP-MR向量化检查](/papers/vamp-mr/)：后续FK与碰撞检查加速参考，尚未接入。
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
        geometric = snapshot['experiment']=='geometric_ik'
        indexed=snapshot['experiment'] in ['index_lookup_diagnosis','index_seed_probe']
        label = 'E：地图索引复用' if indexed else 'C：上下文复用' if binding else 'D：几何IK' if geometric else 'B：原生批量IK'
        status = ('纯规划对照已完成' if snapshot['phase']=='complete' else
            '纯规划对照停止，保留记录' if snapshot['stopped'] else
            '纯规划对照进行中' if snapshot['model_warmup_complete'] else '常驻模型预热中')
        status = label+'，'+status
        summary = f"{label}：{snapshot['cases']}份观测，规划{done}/{total}次，物理抓取0次。"
        if indexed:
            probing=snapshot['experiment']=='index_seed_probe'
            done=snapshot.get('completed_IK_calls' if probing else 'completed_queries',0)
            total=snapshot.get('total_IK_calls' if probing else 'total_queries',0)
            stage='原目标IK初值对照' if probing else 'CPU近邻查询'
            state=('已完成' if snapshot['phase']=='complete' else '已停止' if snapshot['stopped'] else
                '等待空闲显卡' if snapshot['phase']=='waiting_for_gpu' else '进行中')
            status=label+'，'+stage+state
            summary=f'E：{stage}{done:,}/{total:,}次，比较3°、5°、10°查询范围。'
            if snapshot.get('stop_summary',{}).get('phase')=='stopped_before_model_loading':
                summary+='模型加载前可用显存低于10,000 MiB，IK实际0次，未执行3,132次。'
        body = body.replace('方法调研、本地接口与ARX静态几何核查已完成，下一步适配并比较各方法。当前V8规划回放0次、物理抓取0次。',
            summary)
        body = body.replace('## 优化目标', f'[本轮进度与逐例数据](/media/{name})\n\n## 优化目标',1)
        body = body.replace('## 本地已经确认的条件',
            '## 批量接口怎样保持逐目标检查\n\n'
            '先成批产生并检查原32个初值。每个目标已有合格解时先保留，只有未通过的行继续优化；这样其他失败目标不会冲掉该行已通过的初始解。补齐行不参与选择，返回解继续经过原碰撞、接触和路径检查。\n\n'
            '两种批次容量的混合目标检查与独立FK均通过。\n\n'
            '[实施说明](/media/sam3-v8-native-batch-method.md) · '
            '[接口验证数据](/media/sam3-v8-batch-rows-validation.json)\n\n## 本地已经确认的条件',1)
        if indexed:
            body=body.replace('重复寻找已知解 | 待扩展接口 |','重复寻找已知解 | '+status+' |')
            body=body.replace('地图批量查询与见证/初值对照 | 待实施 |',
                '地图批量查询与见证/初值对照 | '+status+' |')
            if snapshot['experiment']=='index_seed_probe' and snapshot.get('acceptance'):
                section=seed_probe_table(snapshot)
                if config.get('index_reuse_implementation',{}).get('include_in_default_combination') is False:
                    section+=('\n四组解出的终点相同，原成功全部保留。当前地图初值策略没有带来总耗时收益，'
                        '保留可选接口，本轮未追加完整规划。后续优先保留B批量16。\n\n')
                body=body.replace('## 优化目标',section+'\n## 优化目标',1)
        elif binding:
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
            if snapshot.get('summaries') and not binding_runtime:
                section += result_table(snapshot,'C上下文复用对照结果')+'\n\n'
            body = body.replace('## 优化目标',section+'## 优化目标',1)
        elif geometric:
            section=('## D怎样加快寻找关节角\n\n'
                '先用EAIK为原抓点求出多个关节解，按真实URDF方向和限位还原，再经过独立FK、当前碰撞与路径检查。'
                '没有合格解时使用原IK；几何解后续接近或抬升不通过时，也先完整尝试同一个原抓法，再检查下一候选。\n\n'
                '27项离线检查通过。256组随机物理姿态与原69份规划的207个终点全部通过独立核查。'
                f"该207个终点上，几何求解加限位与精度筛选中位耗时约{config['geometric_ik']['contract_query_and_filter_median_ms']:.2f}毫秒。完整规划耗时由本轮对照测量。\n\n"
                '[几何IK实施说明](/media/sam3-v8-geometric-ik-method.md)\n\n')
            if snapshot.get('summaries'):section+=result_table(snapshot,'D几何IK对照结果')+'\n\n'
            if config.get('geometric_ik',{}).get('include_in_default_combination') is False:
                increase=-config['geometric_ik']['total_wall_reduction_fraction']
                section+=(f'20次规划全部通过，60个终点独立复算通过，抓法与选臂一致。累计耗时增加{increase:.2%}，'
                    '未达到扩展门槛，因此没有启动69场景对照。\n\n'
                    '1434个候选中，1200个没有合格几何分支，198个有分支但终点检查未通过；'
                    '36个几何终点通过，其中22个在后续接近、抬升或路径检查时回到原方法。'
                    '大量原IK调用仍然存在，又增加了几何检查与回退开销。\n\n'
                    '当前保留D作为可选实现，下一项是E地图索引复用，B批量16继续优先保留。\n\n')
            body=body.replace('## 优化目标',section+'## 优化目标',1)
        elif not native_runtime or not native_runtime[0].get('summaries'):
            body = body.replace('逐个调用小规模求解的开销 | 待适配 |',
                '逐个调用小规模求解的开销 | 已接通，独立对照中 |')
        site['localAssets'][name] = '../../resources/media/'+name
    if binding_runtime and binding_runtime[0].get('summaries'):
        value,name=binding_runtime;summary_values=value['summaries']
        control,reuse=summary_values['control'],summary_values['reuse']
        saving=1-reuse['total_seconds']/control['total_seconds']
        section=result_table(value,'C上下文复用完整结果')
        section+=(f"\n\n累计耗时降低{saving:.2%}，完整重绑定{control['full_world_rebinds']}→{reuse['full_world_rebinds']}次。"
            f"\n\n[逐例结果](/media/{name}) · [方法说明](/media/sam3-v8-binding-reuse-method.md)\n\n")
        if config.get('binding_reuse_implementation',{}).get('include_in_default_combination') is False:
            section+=('中位耗时高于同期控制组，当前保留为可选方案。后续优先组合批量16与其他实测有效的模块。\n\n')
        body=body.replace('## 优化目标',section+'## 优化目标',1)
        body=body.replace('交替窗口反复绑定世界 | 待适配与测显存 |','交替窗口反复绑定世界 | 138次完整对照已完成 |')
        body=body.replace('原生批量IK与绑定复用分别适配 | 待实施 |',
            '原生批量IK与绑定复用分别适配 | B/C完整对照与独立复算完成 |')
        site['localAssets'][name]='../../resources/media/'+name
    if geometric_runtime and (not runtime or runtime[0]['experiment']!='geometric_ik'):
        value,name=geometric_runtime
        if value.get('summaries'):
            section=result_table(value,'D几何IK先导结果')
            section+=('\n\n两组各10/10通过，原抓姿与选臂相同，60个终点独立复算通过。'
                '累计慢6.14%，当前不纳入组合，也未启动69场景扩展。\n\n'
                f'[逐次记录](/media/{name}) · [几何IK说明](/media/sam3-v8-geometric-ik-method.md)\n\n')
            body=body.replace('## 优化目标',section+'## 优化目标',1)
        site['localAssets'][name]='../../resources/media/'+name
    if lookup_runtime:
        value,name=lookup_runtime
        if value.get('acceptance'):body=body.replace('## 优化目标',lookup_table(value)+'\n## 优化目标',1)
        site['localAssets'][name]='../../resources/media/'+name
    if config.get('geometric_ik',{}).get('contract_run'):
        body=body.replace('结构已核查，求解待验证 |','27项检查与463个终点核验通过，按队列开展规划对照 |')
        body=body.replace('验证EAIK自动分解、原限位与工具变换','独立CPU适配与463个终点核验通过，测完整规划')
        body=body.replace('几何IK适配与FK、限位、多分支检查 | 待实施 |',
            '几何IK适配与FK、限位、多分支检查 | CPU核验完成，按队列开展完整规划 |')
        if config['geometric_ik'].get('include_in_default_combination') is False:
            body=body.replace('27项检查与463个终点核验通过，按队列开展规划对照',
                '20次与60个终点复核完成，慢6.14%，未扩69场景')
            body=body.replace('CPU核验完成，按队列开展完整规划',
                '先导与60终点复核完成，当前方案未提速')
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
        body=body.replace('当前交替搜索的完整69份观测耗时将在V8重新测量。',
            '当前交替搜索的完整69份观测已完成B、C同期回放，结果见上方表格。')
        site['localAssets'][name] = '../../resources/media/'+name
    if funnel:
        section=('## 后续方向：分级筛选与批量路径\n\n'
            'V7已完成候选去重与原候选后备的69场景对照，NMS15与B16组合累计省时28.65%，两组各69/69通过。'
            '接下来推进完整抓法联合检查，以及直接复用预抓取关节解的小批量路径规划。\n\n'
            '| 工作 | 安排 |\n| --- | --- |\n'
            '| 夹爪粗筛和代表候选去重 | V7离线和138次规划完成，保留NMS15；几何硬筛误删3条原成功抓法，暂缓启用 |\n'
            '| 预抓取、抓取和抬升联合检查 | 基于B16补批量处理，保留同一组关节分支 |\n'
            '| 少量已知关节终点并行规划 | 优先评估本地批量关节空间接口，减少重复IK |\n'
            '| Goalset | 单独比较选点与避障表现，保留原路径回退 |\n\n'
            '原IK能解出的115个终点中，有85个在2厘米/10°地图邻域里没有记录。'
            '地图空值继续表示未知，不能据此删除候选。\n\n'
            '[建议核对与实施顺序](/media/sam3-v8-funnel-review.md) · '
            '[本地接口与反例数据](/media/sam3-v8-funnel-source-review.json)\n\n')
        body=body.replace('## 优化目标',section+'## 优化目标',1)
    if diagnosis:
        values=diagnosis['summaries']['nms15'];counts=diagnosis['endpoint_candidates']['nms15']
        section=['## V8后续阶段：完整69例定位已完成','',
            '已从V7完整138次规划记录中，拆分69个配对场景的耗时和终点拒绝原因。'+
            ('根据定位开展抓取分支批量复核先导，进度见本页最上方。' if funnel_active.get('validation_run') else
             '当前进入完整抓法批量验证的适配准备，下一阶段再比较4、8个已知预抓取关节目标的并行路径。'),'',
            '| NMS15＋B16的环节 | 每场景平均耗时 |','| --- | ---: |']
        for key,label in [('opening_seconds','双臂开爪检查'),('candidate_search_seconds','候选搜索'),
            ('postprocessing_seconds','NMS与候选队列构造'),('decode_seconds','共享候选读取与解码'),
            ('other_request_seconds','其余状态准备与请求记录'),('postprocess_plus_plan_seconds','合计')]:
            section.append(f"| {label} | {values[key]['mean']:.3f}秒 |")
        section += ['',f"候选搜索中的批量抓取IK，GPU事件计时平均{values['batched_grasp_IK_GPU_seconds']['mean']:.2f}秒；"
            '这是当前优先核对的计算开销。','',
            '| 候选终点检查结果 | 次数 |','| --- | ---: |']
        for key,label in [('no_kinematic_solution_returned','本次未返回合格运动学解'),
            ('solutions_rejected_table','有解，被桌面碰撞拒绝'),
            ('solutions_rejected_other_world','有解，被非桌面场景碰撞拒绝'),('endpoint_passed','抓取终点通过')]:
            section.append(f"| {label} | {counts.get(key,0)} |")
        section+=['',('每次绑定、预抓取/抬升IK和PRM的计时已接入本轮先导，抓取分支批量复核正在对照。' if funnel_active.get('validation_run') else
            '下一步先补齐每次绑定、预抓取/抬升IK和PRM的独立计时，同时适配抓取分支的批量FK与碰撞验证。')+
            '继续保留原候选、最终精度、871球和V2规则。','',
            '[完整69例耗时定位报告](/media/sam3-v8-full69-funnel-diagnosis.md) · '
            '[逐例数据与拒绝来源](/media/sam3-v8-full69-funnel-diagnosis.json) · '
            '[V7已完成的NMS规划对照](/iterations/sam3-graspgenx/versions/V7/)','']
        body='\n'.join(section)+'\n'+body
    from website.tools.indexed_validation_section import section as validation_section
    if funnel_active.get('validation_pilot_run'):
        validation_section(website,site,{'validation_run':funnel_active['validation_pilot_run']})
    validation=validation_section(website,site,funnel_active)
    if validation:
        body=validation['body']+'\n'+body
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
            else '几何IK纯规划对照' if snapshot['experiment']=='geometric_ik'
            else '地图近邻CPU查询' if snapshot['experiment']=='index_lookup_diagnosis'
            else '地图初值IK对照' if snapshot['experiment']=='index_seed_probe'
            else '原生批量IK纯规划对照'),'date':'2026-11-26',
            'status':status,'summary':summary,'outputs':[run],
            'artifacts':[{'label':'进度与逐例结果','url':'/media/'+name}]}
        version['batches'] = [v for v in version['batches'] if v['id']!=batch['id']]+[batch]
    if diagnosis:
        status='后续阶段：完整69例定位完成，联合批量检查待适配'
        summary='69例耗时与终点拒绝拆分完成；当前重点为批量抓取IK与开爪/绑定开销。'
        version.update(status=status,resultSummary=summary)
        version['settings'][-1]=['当前进度',status]
        batch={'id':diagnosis_run.name,'title':'完整69例的剩余耗时与拒绝定位','date':'2026-11-26',
            'status':'已完成','summary':summary,'outputs':[str(diagnosis_run.relative_to(ROOT))],
            'artifacts':[{'label':'定位报告','url':'/media/sam3-v8-full69-funnel-diagnosis.md'},
                {'label':'逐例数据','url':'/media/sam3-v8-full69-funnel-diagnosis.json'}]}
        version['batches']=[v for v in version['batches'] if v['id']!=batch['id']]+[batch]
    if validation:
        status,summary=validation['status'],validation['summary']
        version.update(status=status,resultSummary=summary)
        version['settings'][-1]=['当前进度',status]
        version['batches']=[v for v in version['batches'] if v['id']!=validation['batch']['id']]+[validation['batch']]
    if group['current']=='V8':
        entry = next(v for v in site['versions'] if v['id']=='sam3-graspgenx')
        entry.update(status='V8'+status,metric=summary)
    write_json(site_path,site);write_json(registry_path,registry)
    print('Published V8 methods and recorded progress; physical grasps remain zero.')


if __name__=='__main__':
    publish()
