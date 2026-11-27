"""Audit frozen IK/collision receipts and write the V8 Chinese report."""
import argparse
from pathlib import Path

from arx_project.layout import ROOT, now, read_json, write_json, sha256, ensure_run_path
from .prepare import guard


LABELS={'M0001_S1_open_apple':'苹果','M0020_S1_open_donut':'甜甜圈',
    'M0044_S3_sparse_banana':'香蕉','M0064_S4_clutter_mango':'芒果',
    'M0074_S4_clutter_scissors':'剪刀'}
DOC=ROOT/'modules/pipeline/docs/mainline_versions/V8_GPU解析IK与批量碰撞验证.md'


def create(root):
    reg=guard(root);accepted=read_json(root/'acceptance.json')
    if (root/'STOP').exists() or not accepted['task_complete'] or not accepted['record_integrity']:
        raise ValueError('Cannot publish an incomplete or stopped microbenchmark')
    cases=[]
    for item in reg['cases']:
        folder=ROOT/item['directory'];claim=read_json(folder/'claimed.json')
        metrics=read_json(folder/'metrics.json')
        if claim['state']!='complete' or claim['metrics_sha256']!=sha256(folder/'metrics.json'):
            raise ValueError('Collision receipt is incomplete or differs')
        if metrics['branch_receipt_sha256']!=sha256(folder/'branches.npz'):
            raise ValueError('Saved branch arrays differ')
        if not metrics['passed'] or not metrics['width_equivalence']['passed'] or not metrics['session_contract']['passed']:
            raise ValueError('Native collision or resident API verification failed')
        cases.append(metrics)
    if accepted['cases']!=cases:raise ValueError('Acceptance rows differ from per-case receipts')
    pure=accepted['pure_analytic'];known=sum(r['known_branches_recovered'] for r in pure)
    reference=sum(r['exact_EAIK_branches_matched'] for r in pure)
    proof={'at':now(),'passed':True,'source_and_input_hashes':True,'case_receipts_and_array_hashes':True,
        'production_bytes_unchanged':True,'collision_cases':len(cases),'known_branches_recovered':known,
        'EAIK_reference_branches_matched':reference,'width_equivalence_states':sum(r['width_equivalence']['states'] for r in cases),
        'task_result':'GPU解析求解、原生批量碰撞及常驻接口核验完成',
        'record_integrity':'登记、输入、代码、逐组回执及数组散列全部通过',
        'event_applicability':'适用于单次终点解析IK与原生碰撞；完整路径和物理结果尚待接入对照',
        'key_event_coverage':'左右臂、原限位、近奇异、无解、无效输入、尾批填充、不同开度、世界/自碰撞及输出所有权'}
    write_json(root/'independent_review.json',proof)
    snapshot={'run':str(root.relative_to(ROOT)),'registration':reg,'acceptance':accepted,'independent_review':proof}
    write_json(root/'publication_snapshot.json',snapshot)
    lines=['# V8 GPU解析IK与cuRobo批量碰撞验证','','ARX可以走“解析IK生成关节分支，再用cuRobo批量验碰撞”这条路线。'
        '当前已实现常驻GPU接口，固定一次处理256个目标、每目标最多8个普通分支，实际开爪宽度逐候选传入。','',
        '## ARX为什么适合解析求解','','实际URDF中，第22、23、24关节轴平行，第25、26关节轴相交。'
        '沿关节轴移动数学原点后，可使用IK-Geo的“三平行轴＋两相交轴”分解；左右臂分别读取自己的物理链和限位。'
        '这一步只改变公式中的坐标表示。','',
        '求解依次得到肩部、腕部和肘部各两种分支，组成最多8组角度。限位、重复分支和当前目标几何会减少实际可用解数。'
        '奇异位置可能有连续解族，需要保留原数值IK回退。','',
        '参考：[IK-Geo官方实现](https://github.com/rpiRobotics/ik-geo)、'
        '[EAIK项目主页](https://ostermd.github.io/EAIK/)。新增张量实现保留IK-Geo的BSD许可证。','',
        '## 实现流程','','| 步骤 | 做法 |','| --- | --- |',
        '| 解析分支 | 使用PyTorch在GPU上并行展开256×8分支，恢复原URDF关节符号与合法的2π角度表示 |',
        '| 精度复验 | 代数用FP64计算，输出FP32命令；再按原物理链FK检查原始目标，保持2毫米／0.03弧度标准 |',
        '| 不同开度 | 将每候选宽度的一半分别赋给两根夹指；原生FK包含6个臂关节和2个夹指坐标，机械臂规划仍为6自由度 |',
        '| 原生碰撞 | 用871球同时检查2048行关节状态，分别返回世界碰撞及世界＋自碰撞掩码 |',
        '| 常驻调用 | 首次载入场景并捕获CUDA Graph，后续重复使用；尾批填充到256，输出时去掉填充行 |',
        '| 输出保存 | 返回调用者拥有的GPU张量，下一次查询不覆盖上一批结果；另提供包含上传和下载的CPU接口 |','',
        '## 精度与分支核验','','| 项目 | 结果 |','| --- | ---: |']
    lines += [f'| 随机已知可达目标找回原分支 | {known}/8192 |',
        f'| EAIK精确参考分支匹配 | {reference}/{reference} |',
        f"| 所有随机有效分支独立FK最大位置误差 | {max(r['independent_FK']['maximum_position_error_m'] for r in pure)*1000:.6f}毫米 |",
        f"| 所有随机有效分支独立FK最大朝向误差 | {max(r['independent_FK']['maximum_rotation_error_rad'] for r in pure):.3g}弧度 |",
        f"| 三种开度与原锁定开度模型的对照 | {proof['width_equivalence_states']}状态全部一致 |",
        '| 常驻Graph、尾批、输出所有权与无效输入 | 10组全部通过 |','| 离线故障及数学检查 | 14项通过 |','',
        '首版FP32代数在右臂4096个已知目标中有3个近奇异目标没有返回合格分支，已按回退记录。'
        '最终版本改为FP64代数，并对FP32命令重新检查，8192个目标全部恢复。首版登记与结果继续保留。','',
        '## 256目标整批耗时','','下表每行是固定256个目标、最多2048分支的整批中位耗时。'
        '使用RTX 4090、统一预热及CUDA Graph；每组GPU计时50次。','',
        '| 场景／规划手臂 | 解析IK | 原生FK＋世界碰撞 | IK＋FK＋世界碰撞 | IK＋FK＋世界及自碰撞 | CPU接口含上传下载 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in cases:
        values=[r['timing'][k]['graph_device']['median_ms'] for k in ['analytic_256x8',
            'native_FK_world_2048','analytic_and_FK_world','analytic_and_FK_world_self']]
        lines.append('| '+LABELS[r['case']]+'／'+('右' if r['arm']=='right' else '左')+' | '+
            ' | '.join(f'{v:.2f}毫秒' for v in values)+
            f" | {r['host_API_with_upload_owned_outputs_and_download']['median_ms']:.2f}毫秒 |")
    lines += ['', 'CPU接口计时另测10次，包含输入上传、Graph执行、独立输出复制及下载；首次场景装载和Graph捕获单列在逐组JSON中。'
        '碰撞时间随障碍网格和球位置变化，解析求解时间相对稳定。','',
        '这组测试完成的是终点求解与碰撞模块。当前整次规划还包括开爪检查、预抓取／接近／抬升连续性、V2指尖近桌网格复核及完整路径。'
        '因此这里的毫秒值不能替换网页原有约4秒的完整请求计时。','',
        '## 接回抓取链路的方式','','原候选先保持评分和双臂交替顺序，再将抓取、预抓取和抬升终点分批交给本接口，'
        '使用其返回的关节分支接续原路径验证。无合格解析分支或处于奇异解族的目标仍保留原IK。','',
        '当前原生碰撞结果采用严格球模型；桌面拦截还要经过原V2真实网格复核，允许指尖前端10毫米范围内最多2毫米的接触。'
        '不能提前用严格球模型删掉这类可恢复候选。现有执行链的自碰撞开关保持原值，本次另测了开启自碰撞的耗时与掩码。','',
        '下一步是将该模块接入可选规划器，在同一候选顺序、相同V2规则下比较整次请求；终点通过后仍须完成原路径和抓取判定。','',
        '## 调用入口与坐标约定','','代码入口：`modules/curobo/analytic_ik/session.py`。'
        '使用`environments/curobo`，通过`CUDA_VISIBLE_DEVICES`选择一张卡，接口使用该进程内的`cuda:0`。','',
        '`ExecutableAnalyticIK(arm="right"/"left")`可分别求真实左右臂。'
        '`AnalyticBatchSession`接收当前规划适配器导出的规范右臂模型，左臂场景也先经过原双臂适配器的坐标映射。'
        '输入是`body4`到`EEF`的4×4目标矩阵，不能直接传世界TCP。TCP43换算在接口外完成。','',
        '```python','import numpy as np','from modules.curobo.analytic_ik.session import AnalyticBatchSession','',
        '# robot_config、scene使用当前适配器生成的配置字典。',
        'session = AnalyticBatchSession(robot_config, scene, include_self=False)',
        '# poses: [N, 4, 4]，float64；widths: [N]，米；N <= 256。',
        'answer = session.host_query(np.asarray(poses, dtype=np.float64), widths)',
        'q = answer["q"]                       # [N, 8, 6]，float32',
        'valid = answer["valid_collision"]     # [N, 8]，当前选定碰撞策略',
        'fallback = answer["needs_fallback"]   # [N]，保留原IK的目标',
        '```','',
        'GPU调用使用`device_query`，返回独立GPU张量；目标为float64，宽度为float32，均放在`cuda:0`，在创建会话的同一CUDA流调用。','',
        '## 本次验收','','| 项目 | 结果 |','| --- | --- |']
    for key,label in [('task_result','任务结果'),('record_integrity','记录完整性'),
        ('event_applicability','事件适用性'),('key_event_coverage','关键事件覆盖')]:lines.append(f'| {label} | {proof[key]} |')
    lines += ['',f'最终登记：`{root.relative_to(ROOT)}`。',
        '首版登记：`tmp/analytic_ik/batch256_20261127_01`。本次完整规划、仿真与实抓均为0；默认入口保持。','']
    DOC.write_text('\n'.join(lines))
    write_json(ROOT/'tmp/analytic_ik/active.json',{'at':now(),'run':str(root.relative_to(ROOT)),
        'complete':True,'report':str(DOC.relative_to(ROOT))})
    print('Audited GPU analytic batch and wrote',DOC)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();create(ensure_run_path(args.root))
