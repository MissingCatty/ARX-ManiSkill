## 本版进度

故障停止，保留记录。已完成 5/74 个配对场景。

本次两组都重新运行模型与抓取，使用相同的74个场景、SAM3掩码、RGB-D、最后站位及物体状态。每组资源就绪后独立预热并开始，先就绪先测。

分组进度：RaySt3R＋GraspGenX完成5/74；ZeroGrasp完成44/74。

## 本次比较的链路

| 方案 | 生成与候选处理 | 候选交接与规划 |
| --- | --- | --- |
| RaySt3R＋GraspGenX | RaySt3R重建、GraspGenX原候选与原评分；不加后处理 | 共享内存、后台保存、一次解码 → 双臂cuRobo |
| ZeroGrasp | 原生重建与抓姿、碰撞筛选、宽度深度修正、NMS | 共享内存、后台保存、一次解码 → 同一双臂cuRobo |

ZeroGrasp允许在一张满足门槛的GPU上完成生成、规划与仿真；RaySt3R独立占用一张GPU，GraspGenX与规划执行使用另一张GPU。模型在每次承接运行中加载和预热一次。实际设备分配单独记录，871球、TCP43、V2指尖2毫米余量与物理成功判据沿用上一轮。

[[流程卡片]]

## 两组进度与结果

| 方案 | 完整结束 | 中断 | 未开始 | 规划成功率 | 抓取成功率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RaySt3R＋GraspGenX | 5/74 | 0 | 69 | 4/5（80.0%） | 4/5（80.0%） |
| ZeroGrasp | 44/74 | 2 | 28 | 36/44（81.8%） | 29/44（65.9%） |

25例SAM3最终失败剔除，原1例中断单列。74例中包含5例共同站位失败，两组都保留在分母中。

## 本次承接

上一批ZeroGrasp完成32例，第33例加载碰撞网格时显存不足中断。已有结果和中断保持，本次只执行ZeroGrasp剩余41例与GraspGenX的74例。
规划服务现在会释放已经不属于当前碰撞场景的历史网格缓存，当前障碍、抓取位姿和规划参数保持。每个新场景领取前检查显存余量。

## 延时对照

每项显示中位数 / P90及实际计时例数。未调用的模块不填零；执行耗时只统计实际动作。共同站位失败的5例不进入模型与规划耗时统计。

| 环节 | RaySt3R＋GraspGenX | ZeroGrasp |
| --- | ---: | ---: |
| 生成、转换并交出共享候选 | 3.4754 / 3.6678 秒（4例） | 1.4099 / 2.0096 秒（42例） |
| 候选一次解码 | 0.3471 / 0.3771 秒（4例） | 0.0020 / 0.0043 秒（39例） |
| 首次cuRobo规划请求 | 6.7249 / 40.3955 秒（4例） | 8.6291 / 12.3773 秒（39例） |
| 生成与首次规划合计 | 10.2380 / 43.9516 秒（4例） | 9.9427 / 14.1116 秒（42例） |
| 执行抓取，含接近与抬升重规划 | 29.3610 / 31.3189 秒（4例） | 48.3352 / 63.5411 秒（36例） |
| 生成开始至本次试验返回 | 41.2236 / 73.4437 秒（4例） | 56.0490 / 75.8142 秒（42例） |
| 整次试验，含场景建立与状态恢复 | 42.7565 / 75.1673 秒（4例） | 57.9134 / 79.4667 秒（42例） |

生成阶段包含模型推理、其原生候选处理以及交出候选的时间。首次规划请求包含共享候选读取、双臂搜索与后台记录完成确认。执行阶段包含开爪、接近、闭合、抬升、保持及过程中重规划。

## 失败分布

按场景最终失败阶段计数，括号内为占该组完整结束场景的比例。

| 失败阶段 | RaySt3R＋GraspGenX | ZeroGrasp |
| --- | ---: | ---: |
| 共同高可达站位未通过 | 1/5（20.0%） | 2/44（4.5%） |
| 没有候选 | 0/5（0.0%） | 3/44（6.8%） |
| 初始规划不出来 | 0/5（0.0%） | 3/44（6.8%） |
| 接近运动未到位 | 0/5（0.0%） | 2/44（4.5%） |
| 闭合未到位 | 0/5（0.0%） | 2/44（4.5%） |
| 抬升或保持高度不足 | 0/5（0.0%） | 3/44（6.8%） |

## 首次启动与预热

| 分组 | 模块 | 加载与预热耗时 | GPU |
| --- | --- | ---: | --- |
| RaySt3R＋GraspGenX | rayst3r | 28.65 秒 | `GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b` |
| RaySt3R＋GraspGenX | graspgenx | 13.83 秒 | `GPU-b827b8be-c675-83f7-1b4c-39b5811c6d5e` |
| RaySt3R＋GraspGenX | curobo_dual_arm | 119.91 秒 | `GPU-b827b8be-c675-83f7-1b4c-39b5811c6d5e` |
| ZeroGrasp | zerograsp | 68.45 秒 | `GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169` |
| ZeroGrasp | curobo_dual_arm | 134.95 秒 | `GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169` |

### 实际资源分配

ZeroGrasp：zerograsp：`GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169`；curobo_dual_arm：`GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169`；simulation：`GPU-2786f56b-9d00-ba9f-5a2f-1e38775dc169`。

RaySt3R＋GraspGenX：rayst3r：`GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b`；graspgenx：`GPU-b827b8be-c675-83f7-1b4c-39b5811c6d5e`；curobo_dual_arm：`GPU-b827b8be-c675-83f7-1b4c-39b5811c6d5e`；simulation：`GPU-b827b8be-c675-83f7-1b4c-39b5811c6d5e`。


## 本轮配对视频

### M0002_S1_open_avocado

| RaySt3R＋GraspGenX | ZeroGrasp |
| --- | --- |
| <video controls preload="none" src="/media/sam3-v6-v6-io74-continue-20261125-01-m0002-graspgenx.mp4"></video> | <video controls preload="none" src="/media/sam3-v6-v6-io74-continue-20261125-01-m0002-zerograsp.mp4"></video> |
### M0004_S1_open_banana

| RaySt3R＋GraspGenX | ZeroGrasp |
| --- | --- |
| <video controls preload="none" src="/media/sam3-v6-v6-io74-continue-20261125-01-m0004-graspgenx.mp4"></video> | <video controls preload="none" src="/media/sam3-v6-v6-io74-continue-20261125-01-m0004-zerograsp.mp4"></video> |
### M0005_S1_open_bar_soap

| RaySt3R＋GraspGenX | ZeroGrasp |
| --- | --- |
| <video controls preload="none" src="/media/sam3-v6-v6-io74-continue-20261125-01-m0005-graspgenx.mp4"></video> | <video controls preload="none" src="/media/sam3-v6-v6-io74-continue-20261125-01-m0005-zerograsp.mp4"></video> |

## 数据与历史记录

- [本次逐例统计、延时与失败分布](/media/sam3-v6-v6-io74-continue-20261125-01-summary.json)
- [上一轮文件交接版本的完整页面](/media/sam3-v6-v6-io74-continue-20261125-01-previous-page.md)
- [ZeroGrasp方法](/papers/zerograsp/)
- [GraspGenX方法](/papers/graspgenx/)
