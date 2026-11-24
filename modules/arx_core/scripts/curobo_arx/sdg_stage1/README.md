# ARX SDG 第一阶段

入口对应 `research_plan/docs/sdg_stage1_main_steps_20261031.md`。官方 NVIDIA Grasping SDG 负责网格对向采样，cuRobo 负责完整 ARX 从初态到 H、接近 G 和抬升的运动。原腕部相机同步记录 RGB-D、状态和动作；原物理/数据门控通过后才入库。没有使用官方示例的夹爪瞬移、关闭重力或示例机器人。

官方来源：[6.0 官方教程](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/synthetic_data_generation/tutorial_replicator_grasping_sdg.html)。每次登记从本机 `localhost/arx-s02-isaac:6.0.1` 提取未修改的采样源码并记录版本及哈希；当前扩展为1.1.5。完整组合网格含子节点变换，避免只采第一个Mesh。原 `PAD_CENTER_EEF=[-.006,0,.0375]`、闭合轴+Y、接近轴+Z及相机/动作约定不变。平行夹爪等价翻转围绕接触中心，并重新计算TCP平移；去重比较接触中心的5mm/10度对称类。导出时复核实际G对应的接触中心，防止偏置元数据进入训练集。

2026-11-01 用户变更：暂停 EGAD。旧 `runs/sdg_stage1_20261031_07` 已通过 STOP 停止新调度，旧登记、失败和成功记录保留。新准备入口默认使用 `common_household`，从 `configs/lilo_vla/common_household_collection.json` 读取并校验日用品资产清单哈希；默认包含 **100 个独立几何对象实例**（原有37个日用品＋新下载63个），包括同类别不同模型，不是100个语义类别。资产来源为固定版本 NVIDIA RoboLab 的 HOPE、HOT3D、YCB、HANDAL。模型、纹理和许可证保存在 `asset/common_household_robolab*_20261101_01/`；合并清单为 `asset/common_household_collection_20261101_01/catalog.json`。

选择仅作用于候选生成、调度和配额。历史场景资产不删除，原物理属性保留；历史基准摆放不变，新示教的XY/yaw变化见下文。未选对象保留 `available`，另标 `collection_enabled=false`，禁止直接领取试验。变更对象范围不能伪装成原混合批次的源修复续采。新机器缺资产或清单哈希不符时直接报错。日用品批次已在11月1日至2日执行；11月2日因图像质量和摆放覆盖问题停止旧采集。物理任务成功不表示图像质量或抓取质量已认证。

如需另外发现本地新增对象，不改变源USD或原67个对象：

```bash
CUDA_VISIBLE_DEVICES='' ../curobo/.venv/bin/python -m scripts.curobo_arx.sdg_stage1.discover_assets --out runs/new_object_pack
```

旧混合批次登记过20个新增资产、共87项。当前日用品清单继续按源USD的显式质量、碰撞及材料属性检查，自动惯量在启动时测量并冻结；明确没有新增资产的历史运行参考。原机器人/环境/67对象仍对照原历史指纹，不排除它们的变更。近似几何相似性未建立；仅去除精确组合网格重复。不可用资产保留原因。夹爪开口约束作用于局部接触截面；不因整个物体包围盒较大就排除薄边或把手抓取。

从**仓库根目录**准备和运行，保留所有旧登记和数据：

```bash
CUDA_VISIBLE_DEVICES='' ../curobo/.venv/bin/python -m scripts.curobo_arx.sdg_stage1.prepare \
  --out runs/sdg_stage1_new \
  --envs-per-gpu 2 --wall-hours 24 --max-attempts 2048 --object-block-attempts 8
../curobo/.venv/bin/python -m scripts.curobo_arx.sdg_stage1.run \
  runs/sdg_stage1_new --compute-gpu 2 --sim-gpus 2 3
```

新默认配置：每物体4096个几何采样预算、每接触轴24个朝向、128个去重储备、目标32个G，每G四档各8个成功H。旧固定摆放模式100对象的理论目标102400条；新摆放模式按每物体实际登记组合数扩展配额。示例命令实际受2048次物理尝试或24小时先到者限制，不承诺达到该理论目标。它们是预算/目标，不是已采到的数量或可抓保证；本次资产准备未启动此命令。每G/距离先安排未扰动和扰动入口；旧模式8次对象块后轮转，新摆放配置每块最多2次后轮换摆放；局部H随机种子包含对象/G身份。历史模式重置到原初态；新模式保留原机器人初态、显式登记目标物体XY/yaw，再以正常ARX运动抵达H，记录H_actual。不放宽碰撞、收敛、双侧接触和>6cm/3s保持标准。到H的IK/PRM/密集轨迹优化额外保留目标物体原碰撞几何和实测姿态作为障碍；抓取/抬升预筛仍沿用原允许目标接触的处理，不改机器人模型、物理或任何碰撞裕度。


### 2026-11-02：图像准入与物体摆放覆盖

新默认日用品配置启用[摆放采样](../../../configs/lilo_vla/household_placement_diversity.json)：9个桌面位置×8个水平朝向，保留原Z/静置倾角。完整目标轮廓须位于原台面内；当前100对象共6280个支撑合格候选组合，均不是已经执行成功的案例。每块2次后轮转，优先未访问摆放；每个摆放重新采观测、变换局部SDG候选、检查ARX可达性。原始物体身份和几何划分保持不变，G/储备/配额/去重按摆放分隔，禁止跨摆放复用旧规划池。报告分别列登记、已尝试、成功摆放，失败照实保留。

同一新登记启用相机几何检查：读回实际近裁剪距离，预筛G及实际优化关节路径的FK相机位姿，记录帧再检查一次。相机安装/标定/裁剪参数、机器人及场景物理均不改。实际抓取成功但图像穿模/近裁剪不合格的记录保存为 `visual_rejected`，不进入训练。旧结果不改写，旧训练pilot须重新通过图像准入。

具体证据、限制和小批验证状态见[修正说明](../../../research_plan/docs/household_visual_placement_quality_20261102.md)。这些改动须用新目录登记；改变物体摆放不能沿用旧批次的source-fix承接。

每GPU同一World推进多个完整ARX环境，多GPU环境组共享常驻规划进程；规划仍串行，RTX相机仍分别渲染，不称为全批处理。场景传输复用不可变碰撞网格文件，每次更新真实刚体姿态，并与原导出公式逐顶点比较；不删障碍、不改碰撞几何或边界。

运行中 `SDG_REPORT.md`、`sdg_summary.json`、`sdg_coverage.csv` 自动更新。几何候选、规划预筛G、完整ARX成功示教分开统计；旧166条不算新SDG数据。普通抓取失败保留后继续；配置、记录、资源或不明闲置状态停止。没有自动无限重试。

导出可在生产中对已入库的不可变轨迹作快照；生产退出也自动导出：

```bash
CUDA_VISIBLE_DEVICES='' ../curobo/.venv/bin/python -m scripts.curobo_arx.sdg_stage1.export \
  runs/sdg_stage1_new --out runs/sdg_stage1_new/exports/snapshot_01
```

导出含 train/validation/development JSONL、原始RGB-D路径、8维状态/8×7动作及训练对象归一化统计；另有 `entry_index.jsonl` 与哈希清单，保存对象/几何/夹爪/关节/相机条件以及 `D_requested=inv(G)@H_requested`、`D_actual=inv(G)@H_actual`。这是教师示教入口关系，不是VLA启动成功率，不含在线检索适配或训练。

实际初始化/执行/导出状态以当前生产目录为准；离线采样和检查不能冒充物理示教。实现进度见 WORK_LOG.md。

### 成功数据入口与抓取质量

`groups/` 保存全部成功、失败和中断尝试，不能直接当作成功示教集。训练使用验收后导出的索引；如需浏览原始成功轨迹，可从已有导出创建独立视图：

```bash
CUDA_VISIBLE_DEVICES='' ../curobo/.venv/bin/python -m scripts.curobo_arx.sdg_stage1.success_view \
  --export runs/sdg_stage1_previous/exports/verified_snapshot \
  --out runs/sdg_success_view_new
```

视图复核已提交账本成员、导出与回执哈希，并只读重跑原物理评估器；原失败记录不修改。旧 `grasp_report.json.success` 表示动作流程完成，最终任务成功须同时有物理/记录验收和成功入库证据。

任务成功仍只说明满足原抬升与保持标准。官方采样得到的对向候选点不等于实际夹爪接触点；当前接触日志没有实际接触点、法线和冲量，不能据此认证力闭合或优质对向夹持。成功视图也不是抓取质量认证集合。

离线检查使用现有 cuRobo Python、USD 资产及本机历史 ARX 记录；`runs/` 数据、模型权重和官方提取源码不随 Git 提交。新机器须补齐相同历史输入，才能运行依赖这些记录的检查或生产登记：

```bash
CUDA_VISIBLE_DEVICES='' ../curobo/.venv/bin/python -m unittest \
  scripts.curobo_arx.data_factory.test_factory scripts.curobo_arx.data_factory.test_export \
  scripts.curobo_arx.data_factory.test_resume scripts.curobo_arx.data_factory.test_diagnostics \
  scripts.curobo_arx.data_factory.test_collection \
  scripts.curobo_arx.sdg_stage1.test_stage1 scripts.curobo_arx.sdg_stage1.test_success_view -v
```

源码修复后续采使用新目录，先证明旧协调进程/容器退出及记录已解决：

```bash
CUDA_VISIBLE_DEVICES='' ../curobo/.venv/bin/python -m scripts.curobo_arx.sdg_stage1.prepare \
  --out runs/sdg_stage1_corrected --extra-catalog runs/new_object_pack/catalog.json \
  --source-fix-parent runs/sdg_stage1_previous --envs-per-gpu 2 --placement-profile historical
```

该入口保留旧STOP、登记、结果和源版本问题说明，继承原截止时间，扣除已消耗的物理尝试，继承对象访问次数以优先未采对象。新源版本使用独立账本，不将旧版本带问题的接触元数据混入新训练导出。只复用哈希一致的官方几何候选库，不重采随机库，不复制轨迹冒充新数据。

2026-10-31修复：场景导出先按原ValidationBridge的矩阵乘法顺序计算原始基座坐标顶点，再使用原1.4m边界筛选；仅对入选网格做姿态/四元数重建验证。若原分解或四元数往返损失精度，改存原始基座坐标顶点、单位姿态，最终仍使用原精度阈值，不放宽模型/物理/碰撞裕度。非法刚体变换和非有限坐标继续停止；后续不可恢复精度故障保存NPZ输入。

故障收尾为SDG在途动作预留原单次预算+60s（当前960s），停止新调度但保留规划器直到仿真退出。仿真未退出或容器消失不能确认时，最终训练导出延期，防止遗漏晚提交。旧登记的unresolved_groups不改写；source-fix-parent只在当前所有进程/容器退出、旧超时组另有闲置/排空/账本关闭回执时生成新承接证据。此前无候选的对象保留预筛记录，续采时优先未处理对象，未虚增物理/调度次数。
