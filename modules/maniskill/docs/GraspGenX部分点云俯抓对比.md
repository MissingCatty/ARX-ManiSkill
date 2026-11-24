# GraspGenX 部分点云俯抓对比

- 场景：布丁盒、罐头、杯子、木块、螺丝刀各一个开阔桌面场景，两法各执行一次，共 10 个新测试槽。
- 输入：ZeroGrasp 使用 RGB、真值深度和实例掩码；GraspGenX 使用同一视角、同一深度及掩码提取的可见部分点云。两法共享冻结观测，规划使用当次测量状态。
- 俯抓：GraspGenX 原生配置为 `planner=graspmoe, moe_obb_density=dense`，在包围盒顶面沿长轴采样位置，并扫描夹爪转角和插入深度；合并扩散候选后使用原生评分。两组统一筛选接近方向与竖直向下夹角不超过 30° 的候选，保留原始朝向及来源。
- 执行：当前 ARX 高度、TCP43、相机、碰撞与物理设置保持；仅按登记规则退让过深插入。计划抬升 3 cm，物体实际抬高至少 1 cm 并保持 3 秒，同时满足原双指接触和动作／接触检查。

| 指标 | ZeroGrasp | GraspGenX |
|---|---:|---:|
| 找到有效抓取终点 | 3/5（60%） | 4/5（80%） |
| 完整初始规划通过 | 3/5（60%） | 2/5（40%） |
| 进入抓取执行 | 3/5（60%） | 2/5（40%） |
| 抓取成功 | 1/5（20%） | 0/5（0%） |
| 已进入执行中的成功率 | 1/3（33%） | 0/2（0%） |

所有无候选、未找到有效终点、规划及执行失败都保留在各自 5 场景分母。终点未找到解不表示已证明数学不可达。本轮样本较少，只报告这五个场景的结果。

| 场景 | ZeroGrasp：终点／完整规划／成功 | GraspGenX：终点／完整规划／成功 |
|---|---|---|
| B0031_S1_open_008_pudding_box | 否／否／否 | 否／否／否 |
| B0025_S1_open_007_tuna_fish_can | 是／是／否 | 是／是／否 |
| B0109_S1_open_025_mug | 是／是／否 | 是／否／否 |
| I0066_S1_open_036_wood_block | 否／否／否 | 是／否／否 |
| B0199_S1_open_043_phillips_screwdriver | 是／是／是 | 是／是／否 |

| 最终失败原因 | ZeroGrasp | GraspGenX |
|---|---:|---:|
| 接近未收敛 | 0 | 1 |
| 物体抬升保持高度不足 | 2 | 1 |
| 俯抓候选未找到可执行初始方案 | 2 | 3 |

本次运行含预热 555.2 秒，其中启动预热 157.5 秒；准备与历史启动故障另行保留。

[完整设置](resource/arx_graspgenx_partial_topdown_pair_20261120_01/registration.json) · [逐例结果](resource/arx_graspgenx_partial_topdown_pair_20261120_01/逐场景结果.csv) · [统计](resource/arx_graspgenx_partial_topdown_pair_20261120_01/paired_metrics.json) · [独立核验](resource/arx_graspgenx_partial_topdown_pair_20261120_01/independent_pair_audit.json)

原始产物：`outputs/arx_graspgenx_partial_topdown_pair_20261120_01/`，每例含 Top10 位姿图及第三人称／腕部拼接录像。
