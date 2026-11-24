# ARX 自动示教工厂

独立实现目录，对应 [实施规格](../../../research_plan/docs/automated_teacher_pipeline_20261029/README.md)。旧 S02 执行器、登记、模型和结果不作为可修改工作区。

主线为真实 RGB-D → 常驻 ZeroGrasp 原始候选 → 按部位/方向和夹爪对称性分组去重 → G 局部 SE(3) 入口 H → cuRobo IK/PRM/轨迹优化 → ARX 正常运动到 H → 从实测入口接近/闭合/提起/保持 → 原物理评估及新协议记录校验 → SQLite 原子入库。

每 GPU 环境组在一个 Isaac Sim `World` 中克隆多个完整 ARX 场景。每个机器人、目标、标定相机、接触路由、状态机和输出目录独立；只平移克隆，规划/数据恢复到环境局部坐标。环境间使用普通碰撞组排除，不修改 PhysX 全局过滤开关或环境内部约束。只在初始化时整体 reset；采集过程每次仅重置相应环境。

物理执行和同一渲染 tick 的多相机采集并发。当前相机使用多个原有 RTX RGB-D 传感器，不声称已实现 tiled-camera 合并缓冲；常驻 cuRobo 请求分时处理，不能称为全链路完全并行。统一预热 IK、PRM、轨迹优化及开/闭夹爪模型后才派发运动。

原生双环境已通过冻结物理配置检查；当前稳定性续采用单环境。真实 ZeroGrasp 推理、抓取运动及腕部录制均已执行，首批完整抓取已通过物理与数据验收，全量配额仍在采集。最新执行状态及未完成项见 [WORK_LOG.md](WORK_LOG.md)，候选分组修正见 [CANDIDATE_SELECTION.md](CANDIDATE_SELECTION.md)。

## 入口

从仓库根目录使用现有 cuRobo Python；以下命令准备登记，不会启动仿真：

```bash
CUDA_VISIBLE_DEVICES='' ../curobo/.venv/bin/python -m scripts.curobo_arx.data_factory.prepare \
  --out runs/data_factory_new_registration --envs-per-gpu 4 --wall-hours 24
```

默认 67 物体、64 个抓取储备、每物体 16 个 G、每 G 四档各 8 个不同成功起点，目标 34,304 条。默认物理尝试上限为目标四倍；`--max-attempts` 可在新登记时明确调小。每 G/距离档最多 32 次物理尝试和 64 次运动前规划拒绝。900 秒单条边界、墙钟上限、候选/配额缺口均保留。初始化失败不会冒充已执行抓取。

正式启动自动检查资源并记录 GPU UUID：

```bash
../curobo/.venv/bin/python -m scripts.curobo_arx.data_factory.launch runs/data_factory_new_registration
# 可指定 --compute-gpu 1 --sim-gpus 3；索引在本机重新解析为 UUID。
```

采集不需要 VLA 服务，不启动训练。默认复用现有 Isaac 镜像 `localhost/arx-s02-isaac:6.0.1`，以及已安装的神经推理/cuRobo 虚拟环境。登记保存独立源码副本；所有既有改动保留，不提交或推送。物理/配置/记录故障停止调度，普通抓取失败保留并继续。

```bash
CUDA_VISIBLE_DEVICES='' ../curobo/.venv/bin/python -m scripts.curobo_arx.data_factory.report runs/data_factory_new_registration
CUDA_VISIBLE_DEVICES='' ../curobo/.venv/bin/python -m scripts.curobo_arx.data_factory.export runs/data_factory_new_registration --out runs/data_factory_new_registration/export_01
CUDA_VISIBLE_DEVICES='' ../tmp/arx-fullchain/env/bin/python -m scripts.curobo_arx.data_factory.video runs/data_factory_new_registration
CUDA_VISIBLE_DEVICES='' ../curobo/.venv/bin/python -m unittest scripts.curobo_arx.data_factory.test_factory scripts.curobo_arx.data_factory.test_export -v
```

原 ZeroGrasp 工厂登记的物理基线覆盖 67 项。后续 [SDG 第一阶段](../sdg_stage1/README.md) 在独立登记中加入20项源USD资产契约与首次运行物理基线，不能把该首次实测说成历史验证。通用新资产用 `import_assets` 导入本地清单，再在新登记中传 `prepare --extra-catalog <catalog.json>`；额外运行时契约目前由 SDG 入口登记，不能假定通用入口会自动建立。USD 须声明尺寸、来源及使用条件；原始 mesh 还须显式提供单位、质量、惯量、摩擦、恢复系数和碰撞近似。导入失败保留原因，不用方块替代。精确碰撞几何哈希将相同形状放到同一划分；近似形状相似性去重未实现，不宣称未见形状泛化。

## 记录与故障

每条尝试在 `groups/<group>/episodes/<attempt>/` 保留原始腕部 RGB-D、相机标定、关节及末端实测状态、G/H/实际入口、真实阶段命令/执行/接触/保持高度。`physical_receipt.json` 复用未修改的 `egad_evaluate.py`；`quality_receipt.json` 检查环境/身份/时间、动作范围、首帧、命令起始和连续 8 步标签。任何已请求阶段/执行回执对应命令丢失都属于记录故障。两份回执通过且所有文件落盘后才提交成功索引；相同实际入口或复制轨迹不重复贡献配额。

`ledger.sqlite3` 使用 WAL/FULL 事务；心跳超时不会自动重领在途任务。恢复必须提供旧进程退出和机器人空闲证据，旧尝试保留为中断，补采使用新编号。当前启动器不自动恢复已启动登记；需明确生成续接登记，不能删除 STOP 或覆盖旧目录。

`REPORT.md` 区分配额完成、预算结束、规划拒绝、实际失败和基础设施故障。抽样选择优先两种 EGAD 形状及失败，再补距离覆盖；不足六段时报告缺口。流式导出按物体/几何角色先划分，保存 8 维 G 误差及 8×7 标签；规范统计仅使用训练对象；均值/方差/极值流式计算，分位数采用第二遍固定区间直方图，并记录每维数值误差界。原始图像保留，沿用现有图像预处理，不缓存固定视觉骨干特征。

实现阶段的离线通过不等于多环境实机理仿真已通过。实际初始化、运行、入库数及已知问题以对应新登记的 `REPORT.md` 和原始回执为准。

最新续采登记：[runs/data_factory_20261030_15/REPORT.md](../../../runs/data_factory_20261030_15/REPORT.md)。实际是否已启动以该目录的 state.json 为准；运行中报告自动更新，旧登记与所有失败记录保留。

首次真实全链路结果：40/60 mm 起点的 mustard 示教入库。首条保持期最低抬升 11.86 cm，连续保持 3 秒；111 个腕部 RGB-D 样本生成 31 个动作块。39 项离线检查通过，真实导出和视频完整解码通过。机器人、场景物理及原成功判据保持冻结。


## 明确续采（保留旧登记）

```bash
CUDA_VISIBLE_DEVICES='' ../curobo/.venv/bin/python -m scripts.curobo_arx.data_factory.resume \
  --from runs/data_factory_20261030_14 --out runs/data_factory_20261030_15 \
  --envs-per-gpu 1 --object-block-attempts 4
../curobo/.venv/bin/python -m scripts.curobo_arx.data_factory.launch \
  runs/data_factory_20261030_15 --compute-gpu 2 --sim-gpus 3
```

入口先锁定旧协调器、确认原进程退出和原虚拟容器消失，核对已有成功的文件哈希。
只在新登记的账本副本上承接中断；旧 STOP、账本、原始轨迹均保留。已下发但无结果的尝试继续占物理预算，未下发的尝试单独标注。累计成功与本次新增分开报告，原截止时间和预算不重置。
单环境与四次轮转保持不变。最新修正移除了异步定时堆栈采样，改为仿真主线程安全点采样；真实 ARX 压力对照在异步采样下复现崩溃，主线程采样完成 250 步。详见 [DIAGNOSIS_WATCHDOG.md](DIAGNOSIS_WATCHDOG.md)。这支持修正诊断器，但不等于长期稳定性已得到保证。机器人及场景物理设置保持原值。
