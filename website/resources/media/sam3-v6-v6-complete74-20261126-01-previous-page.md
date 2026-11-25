## 本版进度

已完成。已完成 72/74 个配对场景。

本次两组都重新运行模型与抓取，使用相同的74个场景、SAM3掩码、RGB-D、最后站位及物体状态。每组资源就绪后独立预热并开始，先就绪先测。

分组进度：RaySt3R＋GraspGenX完成74/74；ZeroGrasp完成72/74。

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
| ZeroGrasp | 72/74 | 0 | 2 | 0 | 57/72（79.2%） | 43/72（59.7%） |

25例SAM3最终失败剔除，原1例中断单列。74例中包含5例共同站位失败，两组都保留在分母中。

## 本次承接

保留前批已经完成的102条结果与2条中断。本批只执行ZeroGrasp剩余0例与GraspGenX剩余44例。原场景序号、随机种子和观测保持。
规划服务现在会释放已经不属于当前碰撞场景的历史网格缓存，当前障碍、抓取位姿和规划参数保持。每个新场景领取前检查显存余量。

## 延时对照

每项显示中位数 / P90及实际计时例数。未调用的模块不填零；执行耗时只统计实际动作。共同站位失败的5例不进入模型与规划耗时统计。

| 环节 | RaySt3R＋GraspGenX | ZeroGrasp |
| --- | ---: | ---: |
| 生成、转换并交出共享候选 | 4.5660 / 5.5299 秒（69例） | 1.3147 / 1.9870 秒（67例） |
| 候选一次解码 | 0.1867 / 0.4264 秒（69例） | 0.0022 / 0.0039 秒（63例） |
| 首次cuRobo规划请求 | 14.1286 / 63.1949 秒（69例） | 7.9315 / 11.9795 秒（63例） |
| 生成与首次规划合计 | 17.9938 / 66.9644 秒（69例） | 9.4039 / 13.3895 秒（67例） |
| 执行抓取，含接近与抬升重规划 | 35.2012 / 37.8826 秒（69例） | 40.7220 / 61.5337 秒（57例） |
| 生成开始至本次试验返回 | 51.0129 / 104.2036 秒（69例） | 47.2549 / 72.5564 秒（67例） |
| 整次试验，含场景建立与状态恢复 | 52.6764 / 105.4460 秒（69例） | 50.2571 / 75.1349 秒（67例） |

生成阶段包含模型推理、其原生候选处理以及交出候选的时间。首次规划请求包含共享候选读取、双臂搜索与后台记录完成确认。执行阶段包含开爪、接近、闭合、抬升、保持及过程中重规划。

## 失败分布

按场景最终失败阶段计数，括号内为占该组完整结束场景的比例。

| 失败阶段 | RaySt3R＋GraspGenX | ZeroGrasp |
| --- | ---: | ---: |
| 共同高可达站位未通过 | 5/74（6.8%） | 5/72（6.9%） |
| 没有候选 | 0/74（0.0%） | 4/72（5.6%） |
| 初始规划不出来 | 0/74（0.0%） | 6/72（8.3%） |
| 接近运动未到位 | 3/74（4.1%） | 4/72（5.6%） |
| 闭合未到位 | 0/74（0.0%） | 5/72（6.9%） |
| 未形成双指夹持 | 6/74（8.1%） | 0/72（0.0%） |
| 抬升或保持高度不足 | 5/74（6.8%） | 5/72（6.9%） |

## 首次启动与预热

| 分组 | 模块 | 加载与预热耗时 | GPU |
| --- | --- | ---: | --- |
| RaySt3R＋GraspGenX | rayst3r | 17.14 秒 | `GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b` |
| RaySt3R＋GraspGenX | graspgenx | 11.22 秒 | `GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169` |
| RaySt3R＋GraspGenX | curobo_dual_arm | 111.80 秒 | `GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169` |

### 实际资源分配

RaySt3R＋GraspGenX：rayst3r：`GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b`；graspgenx：`GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169`；curobo_dual_arm：`GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169`；simulation：`GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169`。


## 本轮配对视频

### M0002_S1_open_avocado

| RaySt3R＋GraspGenX | ZeroGrasp |
| --- | --- |
| <video controls preload="none" src="/media/sam3-v6-v6-io74-continue-20261125-03-m0002-graspgenx.mp4"></video> | <video controls preload="none" src="/media/sam3-v6-v6-io74-continue-20261125-03-m0002-zerograsp.mp4"></video> |
### M0004_S1_open_banana

| RaySt3R＋GraspGenX | ZeroGrasp |
| --- | --- |
| <video controls preload="none" src="/media/sam3-v6-v6-io74-continue-20261125-03-m0004-graspgenx.mp4"></video> | <video controls preload="none" src="/media/sam3-v6-v6-io74-continue-20261125-03-m0004-zerograsp.mp4"></video> |
### M0005_S1_open_bar_soap

| RaySt3R＋GraspGenX | ZeroGrasp |
| --- | --- |
| <video controls preload="none" src="/media/sam3-v6-v6-io74-continue-20261125-03-m0005-graspgenx.mp4"></video> | <video controls preload="none" src="/media/sam3-v6-v6-io74-continue-20261125-03-m0005-zerograsp.mp4"></video> |

## 数据与历史记录

- [本次逐例统计、延时与失败分布](/media/sam3-v6-v6-io74-continue-20261125-03-summary.json)
- [上一轮文件交接版本的完整页面](/media/sam3-v6-v6-io74-continue-20261125-03-previous-page.md)
- [ZeroGrasp方法](/papers/zerograsp/)
- [GraspGenX方法](/papers/graspgenx/)
