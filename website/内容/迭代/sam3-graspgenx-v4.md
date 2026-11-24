## 本版流程

先看目标，计算底盘和升降站位；移动到位并停稳后重新拍摄，再由左右臂规划器选择一条臂完成抓取。

| 步骤 | 具体做法 |
| --- | --- |
| 1. 初次观测 | 头部RGB-D与SAM3获取可见目标的三维测量，生成原生抓姿。 |
| 2. 选站位 | 反推底盘XY、朝向和升降高度，检查整机移动路径及左右臂完整抓取规划。 |
| 3. 移动与停稳 | 底盘限速移动，joint4升降；保持双臂，确认到位和连续0.5秒静止。 |
| 4. 重新观测 | 在新位置重新运行SAM3、RaySt3R与GraspGenX。 |
| 5. 抓取 | 从左右臂选择可执行抓法，接近、闭合、抬升并保持3秒。 |

停稳后的感知与抓取模块可点击查看：

[[流程卡片]]

底盘使用仿真的根位姿接口沿平滑路径移动；升降由原joint4驱动执行。原模型三只轮子的连接为固定关节。

## 100场景测试进度

已完成1/100例；站位到位1例，最终抓取规划通过1例，抓取成功1例。

| 布局 | 已完成/登记 | 最终规划通过 | 抓取成功 |
| --- | ---: | ---: | ---: |
| 开阔 | 1/20 | 1 | 1 |
| 桌边 | 0/20 | 0 | 0 |
| 稀疏干扰 | 0/20 | 0 | 0 |
| 密集杂物 | 0/20 | 0 | 0 |
| 部分遮挡 | 0/20 | 0 | 0 |

尚未完成99例；实际执行抓取1例。

[逐例结果](/media/sam3-v4-summary.json) · [试验登记](/media/sam3-v4-registration.json)

## 场景示例

目标放在桌面近侧，每类展示两例。点击图片查看原图。

| 布局 | 场景一 | 场景二 |
| --- | --- | --- |
| 开阔 | [![apple](/media/sam3-v4-m0001-initial.jpg)](/media/sam3-v4-m0001-initial.jpg) | [![bread](/media/sam3-v4-m0011-initial.jpg)](/media/sam3-v4-m0011-initial.jpg) |
| 桌边 | [![eggplant](/media/sam3-v4-m0021-initial.jpg)](/media/sam3-v4-m0021-initial.jpg) | [![pear](/media/sam3-v4-m0031-initial.jpg)](/media/sam3-v4-m0031-initial.jpg) |
| 稀疏干扰 | [![apple](/media/sam3-v4-m0041-initial.jpg)](/media/sam3-v4-m0041-initial.jpg) | [![bread](/media/sam3-v4-m0051-initial.jpg)](/media/sam3-v4-m0051-initial.jpg) |
| 密集杂物 | [![eggplant](/media/sam3-v4-m0061-initial.jpg)](/media/sam3-v4-m0061-initial.jpg) | [![pear](/media/sam3-v4-m0071-initial.jpg)](/media/sam3-v4-m0071-initial.jpg) |
| 部分遮挡 | [![apple](/media/sam3-v4-m0081-initial.jpg)](/media/sam3-v4-m0081-initial.jpg) | [![bread](/media/sam3-v4-m0091-initial.jpg)](/media/sam3-v4-m0091-initial.jpg) |

## 失败阶段

| 原因 | 数量 |
| --- | ---: |

## 实际视频

| 目标 | 结果 | 视频 |
| --- | --- | --- |
| apple | 成功 | [移动](/media/sam3-v4-m0001-station.mp4) · [抓取](/media/sam3-v4-m0001-grasp.mp4) · [头部](/media/sam3-v4-m0001-head.mp4) |

## 主要设置

双臂各871碰撞球，TCP43；头部D405向下45°。升降和底盘到位后固定，再执行六轴手臂抓取。物体实际至少抬高1厘米并连续保持3秒，双指接触、到位和禁止接触检查通过。

## 参考方法

[SAM3](/papers/sam3/) · [RaySt3R](/papers/rayst3r/) · [GraspGenX](/papers/graspgenx/) · [cuRobo](/papers/curobo/)
