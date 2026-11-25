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

已完成88/100例；站位到位69例，最终抓取规划通过64例，抓取成功48例。 另有1例异常中断、11例未开始。

| 布局 | 已完成/登记 | 最终规划通过 | 抓取成功 |
| --- | ---: | ---: | ---: |
| 开阔 | 20/20 | 14 | 12 |
| 桌边 | 20/20 | 14 | 10 |
| 稀疏干扰 | 20/20 | 14 | 12 |
| 密集杂物 | 20/20 | 15 | 10 |
| 部分遮挡 | 8/20 | 7 | 4 |

异常中断1例，未开始11例。实际执行抓取64例。

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
| 到位后SAM3未分割出目标 | 5 |
| 最后接近未到位 | 5 |
| 初次观测SAM3未分割出目标 | 19 |
| 闭合后未形成双指夹持 | 5 |
| 物体抬升或保持不足 | 4 |
| 闭合未到位 | 1 |
| 站位检查未通过，见逐例阶段记录 | 1 |

## 实际视频

| 目标 | 结果 | 视频 |
| --- | --- | --- |
| apple | 成功 | [移动](/media/sam3-v4-m0001-station.mp4) · [抓取](/media/sam3-v4-m0001-grasp.mp4) · [头部](/media/sam3-v4-m0001-head.mp4) |
| avocado | 成功 | [移动](/media/sam3-v4-m0002-station.mp4) · [抓取](/media/sam3-v4-m0002-grasp.mp4) · [头部](/media/sam3-v4-m0002-head.mp4) |
| banana | 成功 | [移动](/media/sam3-v4-m0004-station.mp4) · [抓取](/media/sam3-v4-m0004-grasp.mp4) · [头部](/media/sam3-v4-m0004-head.mp4) |
| bar of soap | 成功 | [移动](/media/sam3-v4-m0005-station.mp4) · [抓取](/media/sam3-v4-m0005-grasp.mp4) · [头部](/media/sam3-v4-m0005-head.mp4) |

## 主要设置

双臂各871碰撞球，TCP43；头部D405向下45°。升降和底盘到位后固定，再执行六轴手臂抓取。物体实际至少抬高1厘米并连续保持3秒，双指接触、到位和禁止接触检查通过。

## 已验证的规划加速

以下是在同一显卡、相同缓存观测和候选上的完整选臂规划回放。加速实现已验证，供后续登记使用。

| 物体 | 原实现 | 加速实现 | 加速倍数 | 手臂与抓姿 |
| --- | ---: | ---: | ---: | --- |
| 苹果 | 29.9秒 | 16.3秒 | 1.84倍 | 一致 |
| 桃子 | 30.4秒 | 17.6秒 | 1.72倍 | 一致 |
| 锅铲 | 394.5秒 | 56.5秒 | 6.99倍 | 一致 |

另一臂保留能胜出或同名次的候选；开爪保留原有全部宽度采样和871球，批量计算关节限位、自碰撞与场景碰撞。

[逐点开爪与完整规划回放核验](/media/sam3-v4-planning-acceleration.json)

## 参考方法

[SAM3](/papers/sam3/) · [RaySt3R](/papers/rayst3r/) · [GraspGenX](/papers/graspgenx/) · [cuRobo](/papers/curobo/)
