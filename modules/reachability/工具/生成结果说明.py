"""Render the report directly from the completed map and held-out queries."""
from collections import Counter
from pathlib import Path
import numpy as np
from 公共 import ROOT,read


def main():
    directory=ROOT/'产物/首版_01';chapter=ROOT/'docs/可达性地图'
    meta=read(directory/'地图元数据.json');d=read(directory/'查询验证_01/查询对照结果.json')
    display=read(chapter/'resources/首版_01/展示依据.json')
    browser=read(chapter/'resources/首版_01/页面核验.json');assert browser['passed']
    assert browser['html_bytes']==display['html_bytes'],'Recheck the current page before reporting'
    case_lines=[]
    for case in sorted({r['case'] for r in d['cases']}):
        rows=[r for r in d['cases'] if r['case']==case]
        case_lines.append(f"| {case} | {len(rows)} | {sum(r['baseline_pass'] for r in rows)} | {sum(r['map_seed_pass'] for r in rows)} |")
    case_table='\n'.join(case_lines)
    positions=np.array([r['best_seed_pose']['translation_error_m'] for r in d['cases']]);angles=np.array([r['best_seed_pose']['orientation_error_rad'] for r in d['cases']])
    region_lines=[]
    for r in display['top_regions'][:5]:
        region_lines.append(f"| {r['rank']} | {r['body4_x_m']*100:.1f} / {r['body4_y_m']*100:.1f} / {r['body4_z_m']*100:.1f} | {r['height_above_table_m']*100:.1f} | {r['observed_down_fraction']:.1%} | {r['joint_samples']} | {r['mean_joint_margin_rad']:.3f} |")
    region_table='\n'.join(region_lines)
    report=f'''# ARX 朝向可达性地图首版结果

日期：2026 年 11 月 20 日。独立目录：`/home/zhaoyechong/projects/ARX-ManiSkill/modules/reachability/`。

首版地图、查询接口、独立模型核验、历史抓取点离线对照和网页展示均已完成。现在能够查看哪些位置具有较丰富的已采样朝向，并为给定 G／H 检索关节初值。**尚未观察到地图初值提高历史困难点的 IK 通过数。**本轮新增实际抓取 0 次，默认执行链路未改。

## 直接查看

- [交互三维地图](resources/首版_01/交互可达性地图.html)：拖动旋转、滚轮缩放、点击体素查看；全部数据与中文字体内置，可下载后离线打开。
- 已同步到现有进度网站的“配置与核查 → ARX 朝向可达性地图”，沿用原网站访问方式。交互页约 {display['html_bytes']/1e6:.2f} MB，无外部 CDN；关节解库不随网页传输。
- [桌面高覆盖区域表](resources/首版_01/桌面高覆盖区域.csv)：基座及世界坐标、桌上高度、覆盖、样本数与质量指标。

![三维工作区概览](resources/首版_01/三维工作区概览.png)

![桌面高度切片](resources/首版_01/桌面高度切片.png)

## 建图与模型

| 项目 | 本次值 |
|---|---|
| 独立 Sobol 关节采样 | {meta['sampled_joint_states']:,} 组，固定种子 20261120 |
| 通过原关节／静态碰撞检查 | {meta['witnesses']:,} 组（采样保留率 {meta['witnesses']/meta['sampled_joint_states']:.2%}） |
| 静态碰撞拒绝 | {meta['static_scene_rejected']:,} 组；关节越界 0 组 |
| 精细位置格 | {meta['fine']['voxels']:,} 个，边长 2 cm |
| 概览位置格 | {meta['coarse']['voxels']:,} 个，边长 5 cm |
| 精细已观察位姿格 | {meta['fine']['observed_pose_bins']:,} 个 |
| 每位置格完整朝向分箱 | 方位 12 × cos(极角) 8 × 工具滚转 12 = 1152 |
| 朝下方向分箱 | 接近轴与竖直朝下夹角小于 60°，共 288 格 |
| 桌面展示范围 | 原桌面 XY 内，TCP 位于桌面上方 3–30 cm；1376 个概览格 |
| 建图本体耗时 | {meta['wall_seconds']:.1f} 秒，含模型启动；不含后续查询验证和网页整理 |

右臂 `right_joint21–26` 共六个转动关节，升降约 0.22 m。位置在 `body4` 坐标系，工具为 `action_tcp43`。原关节限位、指关节锁定、URDF、碰撞球、TCP 变换、桌面与地面尺寸保持不变。原自碰撞策略为关闭，本轮沿用，不能将本地图解读为已经额外验证自碰撞。

地图保存每组有效关节角、TCP 位姿、六关节中的最小限位余量，以及长度尺度 0.1 m 下的雅可比条件指标。条件指标为最小／最大奇异值之比，不是抓取成功率。

**体素中心只是格子的显示坐标，不保证该精确中心及任意朝向有解。**真实 FK 见证在格内，查询返回实际关节解及其与目标位姿的误差。关节空间采样在笛卡尔空间中不均匀，因此原始样本密度和朝向格覆盖均受采样影响；不能将其当作均匀试验得到的可达概率。体素尺寸不同，覆盖比例也不能直接比较。

建图仅使用模型与独立关节采样，未把历史案例的成功／失败关节解写入图库。静态地图不包含本轮各场景杂物，最终执行仍须经过当前完整场景、实际夹爪状态和整条路径检查。若升降、基座相对桌面位置、TCP、限位或碰撞模型改变，当前静态有效性不能直接沿用，应重建或重新核验。

## 较高覆盖区域示例

以下按 5 cm 概览格中的朝下朝向覆盖排序，相邻展示中心间隔至少 10 cm。仅用于查看工作区域，不增加候选过滤规则，也不改变物体摆放。表中高度指 TCP 高度。

| 序号 | body4 XYZ（cm） | 桌上高度（cm） | 已观察朝下覆盖 | 关节样本数 | 平均最小限位余量（rad） |
|---|---|---|---|---|---|
{region_table}

这版图中，靠近机器人一侧的桌面区域有较丰富的朝下抓取方向；远离该区域、接近工作空间边缘时，已采样到的方向变少。颜色低或没有点只能说明本轮覆盖不足，不能据此宣称数学不可达。

## 查询和离线对照

以 10 个已保存场景中的 64 个 G、64 个固定 8 cm 的 H 和 5 个已找到解的短距离 H，组成 133 个查询。包含历史失败筛选与成功参考，**不是随机工作空间抽样，不用于估计总体可达率**。

两组均采用原完整场景、原位姿容差 2 mm／0.03 rad、原求解器与优化预算。区别仅为是否显式提供地图初值；当前测量关节仍包含在初值中。H 查询若原初始化找到 G 解，两组使用同一 G 解作为当前状态。未执行闭合、抬升或物理保持。

| 场景 | 点位查询数 | 原初始化通过 | 地图初值通过 |
|---|---|---|---|
{case_table}
| 合计 | 133 | 20 | 20 |

按阶段分：G 为 12/64，H8 为 3/64，短 H 为 5/5；两组一致。新增通过 0、退步 0。图库能返回邻近位姿，但这批困难点的最佳候选种子与精确目标仍相差中位 {np.median(positions)*1000:.1f} mm／{np.degrees(np.median(angles)):.1f}°，因此不能以“已有几百万点”推定目标的完整六维邻域已经足够密。未解出仍记未找到解，之前独立求解已找到的有效证据继续保留。

| 计时 | 中位数 |
|---|---|
| 地图初值检索 | {d['median_timings']['map_query_seconds']*1000:.1f} ms |
| 原初始化 IK | {d['median_timings']['baseline_seconds']*1000:.1f} ms |
| 地图初值 IK，不含检索 | {d['median_timings']['map_solver_seconds']*1000:.1f} ms |
| 地图载入、散列校验与空间索引建立（一次） | {d['map_load_seconds']:.2f} s |

本次没有证实速度收益；地图检索会增加单次开销。使用时应常驻加载，而不是每个候选重新读取文件。默认链路暂不接入它作为硬门控。若继续验证，可单独比较候选软排序和更局部的初值覆盖，但不能据此跳过场景碰撞检查。

## 使用入口

在独立目录运行位置查询（单位米）：

```bash
/home/zhaoyechong/projects/ARX-ManiSkill/modules/curobo/.venv/bin/python 工具/查询地图.py --位置 0.45 -0.20 0.10
```

完整位姿查询使用 `--目标 请求.json`。请求包含 4×4 `matrix`，`frame` 选择 `body4` 或 `world`，`tool` 选择 `action_tcp43` 或 `right_eef_link`；可带六维 `current_q` 和 `num_seeds`。世界坐标查询可显式传当前 `robot_base_world`，但坐标换算不代表新的桌面关系已通过静态验证。

Python 常驻接口为 `工具/能力索引.py` 中的 `CapabilityMap`，载入一次后重复调用 `query()`。返回关节初值、见证索引、邻近位姿误差和覆盖状态。**未覆盖返回未知，`hard_candidate_rejection` 始终为 false。**已有完整请求与返回见 [I0038 G 示例](../../产物/首版_01/查询验证_01/I0038_G_查询示例.json)。

## 验收与依据

- 独立 URDF FK 抽查 256 个关节见证：最大位置误差约 0.00014 mm；16 组构型的六关节有限差分雅可比核验通过；原 checker 复核通过。
- 查询接口 7 项检查通过，覆盖坐标转换、见证检索、未知语义、异常输入与未完成地图拒绝。
- 浏览器 9 项核验通过，覆盖独立 HTML／网站两种入口、俯视切换、指标／高度过滤及点击体素读取；附 [交互图预览](resources/首版_01/交互地图预览.png)。
- [建图登记](../../产物/首版_01/原始依据.json)、[完成与数据散列](../../产物/首版_01/完成.json)、[独立模型核验](../../产物/首版_01/独立模型核验.json)、[查询接口核验](../../产物/首版_01/查询接口核验.json)、[133 点对照明细](../../产物/首版_01/查询验证_01/查询对照结果.json)、[页面核验](resources/首版_01/页面核验.json)。

采用关节采样 → FK → 体素及能力指标的工程思路，参考 [DLR Reachability Maps](https://github.com/DLR-MO/reachability_map) 与 [ROS-Industrial REACH](https://github.com/ros-industrial/reach)。本次统计由本地原模型计算，不引用其他机器人的覆盖数值。
'''
    (chapter/'结果说明.md').write_text(report)
    print('REPORT_COMPLETE',chapter/'结果说明.md')


if __name__=='__main__':main()
