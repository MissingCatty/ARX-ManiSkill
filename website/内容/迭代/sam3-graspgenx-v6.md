## 本版进度

已完成。已完成 74/74 个配对场景。

本次两组都重新运行模型与抓取，使用相同的74个场景、SAM3掩码、RGB-D、最后站位及物体状态。每组资源就绪后独立预热并开始，先就绪先测。

分组进度：RaySt3R＋GraspGenX完成74/74；ZeroGrasp完成74/74。

## 本次比较的链路

| 方案 | 生成与候选处理 | 候选交接与规划 |
| --- | --- | --- |
| RaySt3R＋GraspGenX | RaySt3R重建、GraspGenX原候选与原评分；不加后处理 | 共享内存、后台保存、一次解码 → 双臂cuRobo |
| ZeroGrasp | 原生重建与抓姿、碰撞筛选、宽度深度修正、NMS | 共享内存、后台保存、一次解码 → 同一双臂cuRobo |

ZeroGrasp允许在一张满足门槛的GPU上完成生成、规划与仿真；RaySt3R独立占用一张GPU，GraspGenX与规划执行使用另一张GPU。模型在每次承接运行中加载和预热一次。实际设备分配单独记录，871球、TCP43、V2指尖2毫米余量与物理成功判据沿用上一轮。

[[流程卡片]]

## 两组进度与结果

| 方案 | 完整结束 | 进行中 | 中断 | 未开始 | 规划成功率 | 抓取成功率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RaySt3R＋GraspGenX | 74/74 | 0 | 0 | 0 | 69/74（93.2%） | 55/74（74.3%） |
| ZeroGrasp | 74/74 | 0 | 0 | 0 | 59/74（79.7%） | 45/74（60.8%） |

25例SAM3最终失败剔除，原1例中断单列。74例中包含5例共同站位失败，两组都保留在分母中。

## 本次承接

保留前批已经完成的146条结果与0条中断。本批只执行ZeroGrasp剩余2例与GraspGenX剩余0例。原场景序号、随机种子和观测保持。
规划服务现在会释放已经不属于当前碰撞场景的历史网格缓存，当前障碍、抓取位姿和规划参数保持。每个新场景领取前检查显存余量。

## 延时对照

每项显示中位数 / P90及实际计时例数。未调用的模块不填零；执行耗时只统计实际动作。共同站位失败的5例不进入模型与规划耗时统计。

| 环节 | RaySt3R＋GraspGenX | ZeroGrasp |
| --- | ---: | ---: |
| 生成、转换并交出共享候选 | 4.5660 / 5.5299 秒（69例） | 1.2878 / 1.9865 秒（69例） |
| 候选一次解码 | 0.1867 / 0.4264 秒（69例） | 0.0021 / 0.0039 秒（65例） |
| 首次cuRobo规划请求 | 14.1286 / 63.1949 秒（69例） | 7.8389 / 11.8892 秒（65例） |
| 生成与首次规划合计 | 17.9938 / 66.9644 秒（69例） | 9.2067 / 13.3616 秒（69例） |
| 执行抓取，含接近与抬升重规划 | 35.2012 / 37.8826 秒（69例） | 40.6856 / 60.9553 秒（59例） |
| 生成开始至本次试验返回 | 51.0129 / 104.2036 秒（69例） | 46.8474 / 72.5022 秒（69例） |
| 整次试验，含场景建立与状态恢复 | 52.6764 / 105.4460 秒（69例） | 49.7354 / 75.0235 秒（69例） |

生成阶段包含模型推理、其原生候选处理以及交出候选的时间。首次规划请求包含共享候选读取、双臂搜索与后台记录完成确认。执行阶段包含开爪、接近、闭合、抬升、保持及过程中重规划。

## 失败分布

按场景最终失败阶段计数，括号内为占该组完整结束场景的比例。

| 失败阶段 | RaySt3R＋GraspGenX | ZeroGrasp |
| --- | ---: | ---: |
| 共同高可达站位未通过 | 5/74（6.8%） | 5/74（6.8%） |
| 没有候选 | 0/74（0.0%） | 4/74（5.4%） |
| 初始规划不出来 | 0/74（0.0%） | 6/74（8.1%） |
| 接近运动未到位 | 3/74（4.1%） | 4/74（5.4%） |
| 闭合未到位 | 0/74（0.0%） | 5/74（6.8%） |
| 未形成双指夹持 | 6/74（8.1%） | 0/74（0.0%） |
| 抬升或保持高度不足 | 5/74（6.8%） | 5/74（6.8%） |

## 首次启动与预热

| 分组 | 模块 | 加载与预热耗时 | GPU |
| --- | --- | ---: | --- |
| ZeroGrasp | zerograsp | 10.32 秒 | `GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b` |
| ZeroGrasp | curobo_dual_arm | 97.65 秒 | `GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b` |

### 实际资源分配

ZeroGrasp：zerograsp：`GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b`；curobo_dual_arm：`GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b`；simulation：`GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b`。


## 本轮配对视频

### M0002_S1_open_avocado

| RaySt3R＋GraspGenX | ZeroGrasp |
| --- | --- |
| <video controls preload="none" src="/media/sam3-v6-v6-complete74-20261126-01-m0002-graspgenx.mp4"></video> | <video controls preload="none" src="/media/sam3-v6-v6-complete74-20261126-01-m0002-zerograsp.mp4"></video> |
### M0004_S1_open_banana

| RaySt3R＋GraspGenX | ZeroGrasp |
| --- | --- |
| <video controls preload="none" src="/media/sam3-v6-v6-complete74-20261126-01-m0004-graspgenx.mp4"></video> | <video controls preload="none" src="/media/sam3-v6-v6-complete74-20261126-01-m0004-zerograsp.mp4"></video> |
### M0005_S1_open_bar_soap

| RaySt3R＋GraspGenX | ZeroGrasp |
| --- | --- |
| <video controls preload="none" src="/media/sam3-v6-v6-complete74-20261126-01-m0005-graspgenx.mp4"></video> | <video controls preload="none" src="/media/sam3-v6-v6-complete74-20261126-01-m0005-zerograsp.mp4"></video> |

## 数据与历史记录

- [本次逐例统计、延时与失败分布](/media/sam3-v6-v6-complete74-20261126-01-summary.json)
- [上一轮文件交接版本的完整页面](/media/sam3-v6-v6-complete74-20261126-01-previous-page.md)
- [ZeroGrasp方法](/papers/zerograsp/)
- [GraspGenX方法](/papers/graspgenx/)

## 最终完整结果与授权补跑

完成时间：2026-11-26T01:14:02.673667+08:00。两组均完整结束74例，合计148条结果，独立验收通过。

| 链路 | 完整结束 | 规划通过 | 抓取成功 |
| --- | ---: | ---: | ---: |
| RaySt3R＋GraspGenX | 74/74 | 69/74（93.2%） | 55/74（74.3%） |
| ZeroGrasp | 74/74 | 59/74（79.7%） | 45/74（60.8%） |

25例SAM3最终失败单列剔除，5例共同站位失败保留在74例分母内。原V4中断1例继续单列。

## 两例中断补跑

牛油果与甜甜圈按本轮明确授权各补跑一次，原中断尝试与停止记录保留；最终74例结果采用本次完整记录。

| 场景 | 原中断原因 | 本次结果 |
| --- | --- | --- |
| M0042_S3_sparse_avocado | 规划服务显存不足 | 抓取成功 |
| M0060_S3_sparse_donut | 另一分组资源门控触发停止 | 抓取成功 |


- [最终结果报告与配对延时](/media/sam3-v6-complete74-final-report.md)
- [逐项配对延时数据](/media/sam3-v6-complete74-paired-latency.json)
- [148条结果及录像独立验收](/media/sam3-v6-complete74-acceptance.json)
- [两例补跑承接记录](/media/sam3-v6-complete74-makeup-receipt.json)

### M0042_S3_sparse_avocado

| RaySt3R＋GraspGenX | ZeroGrasp补跑 |
| --- | --- |
| <video controls preload="none" src="/media/sam3-v6-complete74-m0042-graspgenx.mp4"></video> | <video controls preload="none" src="/media/sam3-v6-complete74-m0042-zerograsp.mp4"></video> |

### M0060_S3_sparse_donut

| RaySt3R＋GraspGenX | ZeroGrasp补跑 |
| --- | --- |
| <video controls preload="none" src="/media/sam3-v6-complete74-m0060-graspgenx.mp4"></video> | <video controls preload="none" src="/media/sam3-v6-complete74-m0060-zerograsp.mp4"></video> |
