# SAM3 后接 ZeroGrasp 的同观测对照

对应网站 SAM3 主线的 V6，承接 V4，与 V5 分割改进并列。

| 方案 | 处理链路 |
| --- | --- |
| 当前链路 | 头部 RGB-D → SAM3 → RaySt3R → GraspGenX → 双臂 cuRobo → 执行 |
| 实验链路 | 同一份 RGB-D 与 SAM3 掩码 → ZeroGrasp 重建、抓姿生成及原生后处理 → 相同双臂 cuRobo → 执行 |

等待 V4 的 `v4_high100_resume_20261125_02` 完成原剩余场景并通过独立验收后，冻结其中完整结束且 SAM3 最终有效的所有场景。原中断、未开始和最终 SAM3 失败分别记录。规划失败和物理失败仍纳入比较。

每例固定使用原链路最终抓取准备时的图像、掩码、内外参、底盘位姿、升降位置、关节状态及所有物体状态。原链路结果直接作为参照，新分支从相同状态执行一次。原链路在站位阶段就无法进入高可达空间的场景，为两组共同规划失败。到位后的状态恢复、相机与物体位置通过数值核对，机器人静止确认后才允许执行。

ZeroGrasp 使用已有 `zerograsp_synthetic.ckpt` 和 640×480 原生配置。原生碰撞检测、宽度和深度修正、NMS 后的 `.grasp.npy` 是唯一规划候选；全被过滤时保留空集合。候选仅进行坐标换算，采用当前 TCP43，不附加几何筛选、姿态翻转或抓点修正。

保持原 cuRobo 双臂选臂与排名策略、每臂871球、桌子长方体、V2前端10毫米内最多2毫米桌面余量、1毫米物理接触偏移及原成功门控。计划抬升3厘米；物体至少升高1厘米并连续保持3秒，满足双指接触、动作收敛和禁止接触检查。每例一次，无自动重抓；资源、配置、记录或状态不确定时停止后续调度。

## 结果和延时

主要结果仅比较规划成功率、抓取成功率，两组使用相同配对分母。SAM3失败剔除，原中断及新中断另列。

逐例保存共用 SAM3、重建与抓姿生成、首次 cuRobo 规划、执行及执行中重规划、从生成到结束的耗时，汇总中位数与 P90 并标出计时例数。模型首次加载与预热单独保存，各组件记录 GPU UUID。模型环节使用 CUDA 同步后的耗时；历史执行环节从规划返回到最终状态写入的时间戳恢复。

## 入口与目录

- 配置：`configs/experiments/sam3_zerograsp_comparison.json`。
- 编排：`modules/pipeline/experiments/sam3_zerograsp_comparison/`。
- ZeroGrasp适配器：`modules/zerograsp/arx_grasp_adapter.py`。
- 权重：`weights/zerograsp/mainline/zerograsp_synthetic.ckpt`。
- 运行记录：`tmp/sam3_zerograsp_comparison/<run_id>/`。
- 网站：`/iterations/sam3-graspgenx/versions/V6/`。

排队程序只等待 V4 结束、服务退出和验收通过，然后登记新批次并启动一次；模型启动前显存不足时等待原登记门槛。模型和试验开始后的故障保留 STOP，不自动重启。独立跟进程序生成验收与网站快照，所有网页媒体和数据复制进 `website/resources/` 后统一压缩。

```bash
environments/simulation/bin/python -m modules.pipeline.experiments.sam3_zerograsp_comparison.queue \
  --root tmp/sam3_zerograsp_comparison/v6_paired_20261125_01
```
