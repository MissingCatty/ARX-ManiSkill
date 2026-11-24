# ARX 客厅桌面抓瓶：ZeroGrasp → cuRobo → Isaac Sim

当前使用场景为 `bottle_clear_00`：ARX 在客厅茶几上抓取 BBQ 酱料瓶，无额外杆子、架子或动态障碍物。保留原场景和全部 67 个目录资产。按用户最新分配，后续任务只使用物理 GPU2、GPU3（均为 RTX 4090）：LiLo-VLA 在 GPU2，ZeroGrasp、cuRobo 和 Isaac Sim 在 GPU3。下文 A100 性能数据仅是历史测量。

这是本仓库当前 cuRobo 使用入口。执行链路调用 cuRobo Python API，不需要启动 8007 HTTP 服务。目标 mask 沿用该场景的真值投影，未调用 LangSAM。历史实验代码和结果保留，旧演示、其他机器人及旧 HTTP 服务的操作说明已清理。

## 已验证结果和计时范围

2026-10-26 普通运行 `runs/bottle_normal_run_20261026_144441` 实际完成五阶段动作，右夹爪双指接触，抬升约 12 cm，保持 3 秒仿真时间，无禁止接触；五阶段均通过严格关节收敛。

| 测量 | 结果 | 范围 |
|---|---:|---|
| A100 热启动规划，12 次 | 159.9–192.4 ms，中位 175.7 ms | 新 IK + PRM + 轨迹优化 + 原有两次稠密碰撞复核 |
| 首次规划 | 开夹爪模型 6.02 s；闭夹爪模型 2.79 s | 冷启动单独记录 |
| 普通运行执行阶段 | 42.05 s | 录像开始后的执行计时，包含相关通信和处理 |
| 完整原始视角录像 | 43.07 s | 按记录的墙钟时间播放，未加速 |
| 规划器初始化/预热 | 66.32 s | 在录像前，独立于上面的执行耗时 |

毫秒数字不包含 ZeroGrasp 推理、模型初始化、最终证据落盘、仿真执行或视频编码，也不代表整次抓取延迟。A100 规划微基准和 4090 上的物理执行分开验证，不能把这组数视为同一 GPU 的整链性能。每次脚本启动仍有进程和模型冷启动开销；当前脚本没有实现常驻 ZeroGrasp/cuRobo 工作进程。

此前约 130.7 秒的录像中，十次逐段配置审计占约 70.4 秒。普通运行已关闭这些逐段扫描及收尾全量扫描。新一轮候选也不同、额外收敛等待更少，不能把全部差额归因于关闭审计。

## 运行前准备

在仓库根目录操作，使用 `zhaoyechong` 分支。以下依赖需已安装；环境、权重、USD 资产和视频不放进 Git：

- Podman、NVIDIA CDI，以及 Isaac Sim 6.0.1 镜像。官方基础镜像为 `nvcr.io/nvidia/isaac-sim:6.0.1`；本机已安装 Python 依赖的镜像为 `localhost/arx-isaac-warm:6.0.1`。
- Isaac 内 Python 为 3.12.13，完整版本 `6.0.1-rc.7+release.42383.32955d8d.gl`。基础镜像需先安装 `arx-sim/requirements.txt`；下面提供构建命令。
- 仓库 `asset/` 中的原始机器人、客厅 USD 及 `tabletop_objects/`、其引用资源。初始化时核对 67 个目录 USD 的 SHA256，不重新生成、缩放或简化资产。
- ZeroGrasp 环境：Python 3.11、PyTorch 2.7.0+cu128、NumPy 1.26.4，以及已编译的 OCNN/OFE 等依赖。当前已安装环境在 `../tmp/arx-fullchain/env`，CUDA 工具链在 `../tmp/arx-fullchain/cuda`，都可通过环境变量覆盖。新机器需要另外准备这些二进制依赖及离线模型缓存；本文不声称提供已验证的全新机器一键安装器。
- cuRobo 源码必须位于 `../curobo/source`，提交固定为 `8e734f3ced1df898990bcd92de40abce475907db`（本机版本 `0.8.0.post1.dev42`，`MotionPlanner` API）。专用解释器默认 `../curobo/.venv/bin/python`；本机 Warp 1.18.0、cuda-core 1.2.2。更换版本需重新验证补丁和 API。
- 权重位于 `servers/zerograsp/vendor/zerograsp_mainline_minimal/checkpoints/zerograsp_cvpr2025/zerograsp_synthetic.ckpt`；SHA256 为 `0460111a60b31b7b35f6a256e17a95be96ff31bc4abfec3f9a341df93213bc65`（223756102 字节）。

如使用官方基础镜像，先在仓库根目录构建依赖层：

```bash
podman pull nvcr.io/nvidia/isaac-sim:6.0.1
podman build -f scripts/curobo_arx/Containerfile.isaac \
  -t localhost/arx-isaac-warm:6.0.1 .
```

## 初始化一次新的抓取记录

```bash
source scripts/curobo_arx/environment.sh
export ARX_SIM_IMAGE=localhost/arx-isaac-warm:6.0.1
RUN="$PWD/runs/bottle_clear_$(date +%Y%m%d_%H%M%S)"
python3 scripts/curobo_arx/static_pick/init_clear_bottle.py "$RUN"
```

参考数据已随代码放在 `static_pick/reference/`：原始初始状态、视角、场景指纹、资产目录及冻结的开/闭夹爪模型。新运行不再依赖本机旧 `runs/` 目录。初始化拒绝覆盖已有目录。输出应位于仓库内，以便容器访问，并始终保留至少 1 GiB 本地可用空间。

`environment.sh` 是已落盘的环境入口，不再 source 本机临时脚本。需要改环境路径时，在 source 前设置 `ARX_INFERENCE_ENV`、`ARX_NEURAL_PYTHON`、`CUROBO_PYTHON`、`CUDA_HOME` 和 `XDG_CACHE_HOME`。当前 CUDA 扩展固定使用计算能力 8.9 与独立的 `torch_extensions_sm89` 缓存，不复用旧 sm80 编译配置。

本机分配按用户指定时的 GPU2/3 身份固定为下列 UUID。可见设备过滤后，它们可能在 `nvidia-smi` 中重编号为 0/1；这不改变授权身份。数字启动参数 2/3 是原分配的别名，UUID 是最终依据；原 GPU0/1 的 UUID、未知设备和裸数字参数 0/1 均被拒绝，不自动回退：

| 用途 | 变量 | 本机默认 |
|---|---|---|
| LiLo-VLA 推理 | `LILO_GPU` / `--policy-gpu` | 4090，索引 2，`GPU-7a543e5f-d6db-f25d-3e54-6227fb4688b2` |
| ZeroGrasp 推理 | `ARX_NEURAL_GPU` | 4090，索引 3，`GPU-c28703d5-5487-1100-18bc-3ac15fd42c94` |
| cuRobo | `ARX_PLANNER_GPU` | 4090，索引 3，`GPU-c28703d5-5487-1100-18bc-3ac15fd42c94` |
| Isaac 渲染与物理 | `ARX_SIM_GPU` | 4090，索引 3 |

进程内的 `cuda:0` 可以映射到原 GPU2，这不代表使用原 GPU0；策略加载后按 PID 查询实际 UUID，`/health` 提供 `gpu_uuid`、原分配编号 `physical_gpu_index`、当前可见编号 `visible_gpu_index` 和绑定验证结果。新仿真启动器通过 CDI 只暴露指定 UUID，容器内使用本地 GPU0。仿真保留原有 8000 MiB 启动门槛，ZeroGrasp 和 cuRobo 按阶段串行。模型启动前只读检查 BF16 权重体积与可用显存；当前主干裸权重约 14.05 GiB，还需动作头和推理空间。显存不足时明确失败，不量化、不改物理配置、不切换授权设备。

## 启动、抓取和录像

新启动仿真：

```bash
bash scripts/curobo_arx/static_pick/start_clear_bottle.sh "$RUN" --detach
podman logs -f arx-sim-zhaoyechong-gpu3
```

启动器只启动一次 Isaac，运行抓取时自动等待 mailbox 就绪。API 使用 `8101`，WebRTC 使用 `49210/49211`，默认只启用头部 RGB-D，相机校准不变。容器名称可用 `ARX_SIM_CONTAINER` 覆盖。检查启动命令可加 `--dry-run`。

在另一个终端进入同一仓库，source 上面的环境并将 `RUN` 设为同一个绝对路径，然后执行：

```bash
bash scripts/curobo_arx/static_pick/record_clear_bottle.sh "$RUN" \
  > "$RUN/fullchain.log" 2>&1
```

脚本依次：应用锁定版本的性能补丁 → 恢复并核对历史初始状态 → 新采集 RGB-D → GPU3 上运行新的 ZeroGrasp 推理 → 载入冻结 ARX 模型 → cuRobo IK/PRM/轨迹优化 → Isaac 五阶段执行与接触检查 → 编码实际录像。

若已有同一场景、同一物理配置的仿真在运行，为新一次抓取重新初始化 `RUN`，然后复用它的 mailbox：

```bash
bash scripts/curobo_arx/static_pick/record_clear_bottle.sh "$RUN" \
  --mailbox /absolute/path/to/running/simulator/mailbox \
  > "$RUN/fullchain.log" 2>&1
```

不要同时向同一个 mailbox 发起两次请求。旧实验的 `run_stage.py`、`setup_bottle.py`、`setup_clear_bottle.py` 等保留作历史代码，不作为当前运行入口。

## 普通执行和实验审计

默认 `ARX_RESEARCH_AUDIT=0`，普通运行关闭每段动作前后及任务结束的全量配置审计。初始化仍做一次场景状态核对，位于动作录像及执行计时之外。原有碰撞、关节限位、收敛、双指接触、抬升和保持标准照常检查；关闭审计不允许写入机械配置。

只有需要实验配置对照时，为该次实验新建 `RUN`，显式加 `--audit`（或设置 `ARX_RESEARCH_AUDIT=1`）：

```bash
bash scripts/curobo_arx/static_pick/record_clear_bottle.sh "$RUN" --audit
```

普通运行的 `verification.json` 中，`current_configuration_unchanged`、`changed_protected_sources`、`configuration_audit_pass` 等未检查项为 `null`，不能据此宣称逐段配置不变。历史 610 项源文件 guard 和历史权重 SHA256 缺失的部分仍标为未验证。物理初始状态证据与当前可检查文件另行保存。

## 当前规划设置

入口显式设置 `GRASP_TRAJOPT_POLICY=interactive`：先用 1 seed/25 次迭代，再尝试 1 seed/100 次；任一原始验收门槛未通过，继续两组历史完整优化尝试。不直接退回 PRM 路径冒充轨迹优化。全局研究脚本的历史默认策略保持不变。

`CUROBO_KERNEL_BACKEND=cuda_core`，保留 CUDA Graphs。`CUROBO_SWEPT_MESH_BOUNDS=1` 开启由原 mesh 顶点计算的保守 AABB 筛选：覆盖整段 sweep；相交或未知边界继续原 SDF 查询。另复用重复的当前点 SDF。不修改 mesh、碰撞球、margin、限位、驱动、材料、成功门槛或自碰撞配置。本场景冻结的历史自碰撞配置为关闭，不能把这次通过解释为已验证自碰撞规避。

补丁在 `patches/`，通过 `apply_curobo_performance_patch.sh` 幂等安装；外部 cuRobo 仓库本身不推送。规划证据异步有序落盘，成功返回前必须 flush 并核对完成报告。

这条瓶子案例已实际执行通过；历史失败 ARX 候选的规划回放仍保持拒绝。该结果不代表其他物体、带障碍场景或所有历史基准均已通过。早期跨编译数值比对曾失败，记录保留；后续同一 kernel 的开/关配对在原 `1e-6` 阈值下通过，没有放宽精度。

打包后按上述新入口复跑 `runs/curobo_usage_packaging_check_20261026` 也通过：使用随代码提交的参考模型和新推理候选 22，双指接触、抬升约 12 cm、禁止接触为零。执行 59.60 秒，完整录像 60.51 秒；lift/hold 各增加 4 秒原有收敛等待，并由原有的稳定末端姿态门槛接受，二者未达到严格关节收敛。它与前述候选 11、五段均严格收敛的 43.07 秒记录分开保存。运行时间随候选与收敛情况变化。

## 单独测规划耗时

先完成一次上面的抓取，保留完整 `RUN`（模型、scene mesh、PRM seeds 等），再执行：

```bash
CUDA_VISIBLE_DEVICES="$ARX_NEURAL_GPU" "$CUROBO_PYTHON" \
  scripts/curobo_arx/benchmark_planning.py \
  --source-run "$RUN" --output "${RUN}_planning_benchmark" \
  --policy interactive --repeats 5 --full-request
```

`--full-request` 包含新 IK、PRM 和两组原始稠密碰撞检查；省略时只测历史 PRM seed 之后的优化。查看 `benchmark.json`，分别统计模型首轮冷调用和后续热调用。该脚本只验证规划模型，不发送仿真动作，也不提供新的实际接触证据。

## 结果文件

| 文件 | 内容 |
|---|---|
| `verification.json` | 总体验证结果、执行与配置审计范围 |
| `setup_verification.json` | 执行前历史物理状态和初始条件核对 |
| `grasp_report.json` | 五阶段收敛、真实抬升、保持与耗时 |
| `network_candidates.json`、`network/` | 新推理来源及原始候选 |
| `*_command.json`、`*_execution.json` | 实际下发轨迹、反馈和接触 |
| `audit_mode.json`、`planning_implementation.json` | 审计开关、策略、依赖提交及补丁证据 |
| `bottle_grasp_no_obstacle_full.mp4` | 完整原始视角，墙钟时间，无文字/音轨/加速 |
| `bottle_grasp_no_obstacle_action.mp4` | 明确标注的动作片段，完整录像另保留 |
| `action_contact_sheet.jpg` | 实际录像关键帧 |

已提交的 [验证摘要](static_pick/validation_evidence.json) 记录了上述两次实际执行与独立规划微基准的范围和结果。完整实验输出留在本机 `runs/`，不提交 Git。检查通过后仍保留失败尝试，不覆盖旧记录。


## LiLo-VLA 腕部策略接入

当前可执行入口是**目标条件动作头＋距所选抓取位姿 10 mm 的交接点**，仅限已反复验证的 A1 与客厅 BBQ 瓶子两个开发场景。A1、瓶子各完成 3/3 次真实抓取，均满足双指接触、无禁止接触、物体保持期所有采样抬升超过 6 cm 且至少 3 s。4 cm 入口下 A1 仍失败；5 mm 的 A1 在规划运动中提前触物，被原交接检查拒绝。不能把这些结果推广为 49 物体成功率，完整批测保持停止。

执行命令统一使用 `probe_entry_range.py --handoff-profile`。历史代码与失败证据保留用于复算，研究记录集中在 [LiLo-VLA 记录与尝试计划](../../servers/lilo_vla/README.md)。默认 cuRobo 教师入口和旧策略服务保持原用途。

### 当前执行命令

先加载本机运行环境：

```bash
source scripts/curobo_arx/environment.sh
```

历史验证使用下列两种 ARX fixture，均为 Isaac Sim 6.0.1、4090，机器人和场景物理及历史初始状态保持冻结：

| 场景 | 仿真 mailbox | 既定渲染器 |
|---|---|---|
| EGAD A1 | `runs/lilo_egad49_20261026_eval01/runtime/mailbox` | RaytracedLighting |
| BBQ 瓶子 | `runs/lilo_P0_bottle_runtime_20261027_01/mailbox` | PathTracing |

原 GPU2 上的旧瓶子实例现已停止，为 LiLo 推理释放显存；其启动器带 `--rm`，容器随退出移除，历史运行目录仍保留。原 GPU3 保留 RaytracedLighting 主实例。单场景命令只检查所选 mailbox，不再要求停止的实例响应。瓶子正控仍需原 PathTracing 条件，后续应在 GPU3 串行重新启动相应 fixture，不将主场景渲染器改成另一种后冒充原条件。新机器还需要对应历史初始文件、资格试验输入、资产和权重；仅拉代码不含这些大文件。

以下命令在原 GPU2 启动独立策略服务，在原 GPU3 计算规划并执行 A1，结束后只清理自己启动的服务。当前只使用 `--scene A1`；瓶子实例恢复前不运行双场景命令。新设备已实际通过固定目标的芥末瓶抓取，45 步、最低保持抬升 7.178 cm、保持 3 s；A1/BBQ 的历史资格点仍需区分原硬件结果与新设备重放，不能自动扩大资格。完整迁移、苹果失败及诊断记录见 [迁移状态](../../servers/lilo_vla/GPU23_migration_status.json)。

```bash
"$ARX_NEURAL_PYTHON" scripts/curobo_arx/lilo_vla/probe_entry_range.py \
  --output "$PWD/runs/lilo_verified_$(date +%Y%m%d_%H%M%S)" \
  --adapter "$PWD/runs/lilo_P2_comparison_20261028_02/goal_adapter" \
  --handoff-profile "$PWD/runs/lilo_P3_entry_range_20261028_01/verified_handoff.json" \
  --policy-gpu 2 \
  --planner-gpu 3 \
  --bottle-mailbox "$PWD/runs/lilo_P0_bottle_runtime_20261027_01/mailbox" \
  --scene A1
```

加入 `--handoff-profile` 后只重放已有资格点，不搜索距离、不重选模型。默认临时服务端口为 8009，可用 `--port` 显式覆盖。`state.json` 保存进度，`results.json` 和 `summary.json` 保存全部实际结果，案例目录中的 `.mp4` 是原速真实录像。

配置绑定动作头、统计和 manifest 的 SHA256、输入状态版本、场景初始摆放、语言指令及选中抓取目标。交接时对第一张真正送入策略的腕部观测检查：既有物理配置/初始状态、视锥、无提前触物、原局部可达性证明，以及实测位姿和张开夹爪是否接近已成功重复的入口。已记录的图像/状态时间须一致。检查只做一次，模型文件身份只在服务加载时计算；不增加逐动作全量扫描。不符合条件记为失败，不改状态数值、不覆盖模型动作、不回退教师后宣称 VLA 成功。

带检查的最终实际验证在 `runs/lilo_P3_guarded_20261028_02`：A1 和瓶子均通过，分别 17/26 次推理，最低保持抬升 7.211/7.256 cm，保持各 3 s。[全部 P3 尝试、部署接线失败和最终验证](../../servers/lilo_vla/P3_diagnosis.json)。

后续单物体研究发现，到达入口的路径可能在中途碰倒目标，整段轨迹优化也可能切掉原 PRM 中间点。新增显式研究选项使用 80 mm 中间目标分段到达，再交给原 10 mm 入口的 VLA；芥末瓶固定目标单次成功，45 次推理，最低保持抬升 7.314 cm、保持 3 s，权重未重训。补充目标查询对某些历史成功路径也会保守拒绝，因此默认关闭，与已有资格配置互斥。研究命令、前两轮失败及取消记录见 [P4 路径对照](../../servers/lilo_vla/README.md#p4新目标迁移与到达路径检查2026-10-28) 和 [完整结果](../../servers/lilo_vla/P4_transfer_diagnosis.json)。

### 输入、控制与权重

头部 RGB-D 用于 ZeroGrasp；VLA 只读取右腕 RGB、8 维状态和抓取技能指令。同一张腕部图像送入公开模型的两个视觉编码器，沿用官方 JPEG/Lanczos3/90% 面积中心裁剪，保留 ARX 图像朝向。原传感器为 640×480、5 Hz 仿真采样，不代表墙钟闭环速率。

当前状态版本为 `grasp_goal_world_error_v1`：世界坐标下 `p_eef - p_selected_grasp`、`rotvec(R_eef R_selected_grasp.T)`，加两指实测位置。目标是动作前保存的 ZeroGrasp 选中位姿，不从未来闭合位姿反推。服务拒绝状态版本与动作头不匹配的请求。

模型输出仍为 8×7，默认每次执行第一步再观察。位置增量单位为 0.05 m，左乘世界轴角增量单位为 0.5 rad，夹爪沿用原 ARX API。每条动作经原 cuRobo IK、限位、精度和稠密碰撞检查，再以原 1.5 rad/s、5 rad/s²、100 rad/s³ 边界播放。保留原目标减实测关节偏差的运动目标补偿，不修改机器人驱动、增益、夹爪或标定。

LiLo 推理代码固定为 `64aa29d77f5457afb3d1642cfc84534dc1e81fad`，公开模型固定为 `yygx/lilo-vla-openvla-oft` 的 `f54c56709004f6680d2ce8bc9e06d8cc96567ad8`。来源和逐文件校验见 `servers/lilo_vla/vendor/provenance.json`。安装与完整权重续传使用 `bash scripts/setup_lilo_vla.sh`；已有完整环境不必重复安装。当前 ARX 适配只训练动作回归头，骨干与 8 维本体投影器冻结，不称为完整 LoRA 微调。

当前权重、统计与 manifest 在 `runs/lilo_P2_comparison_20261028_02/goal_adapter/`。训练、采集、视频、权重和完整交接配置保留在本地 `runs/`，不将大文件提交 Git。

### 数据与训练复现

P2 共保留 16 次新采集尝试，其中 7 次成功，再复用 1 条资格试验中的瓶子轨迹。训练为瓶子 2 条＋魔方 3 条，验证为独立柠檬 3 条；A1 教师轨迹仅用于诊断，未进入任一划分。两组状态表示使用同一批 RGB、语言、动作和划分，最终各为 210 个训练动作块、126 个验证动作块。实际训练和六次原 4 cm 入口对照见 [P2 记录](../../servers/lilo_vla/P2_diagnosis.json)。

教师采集使用 `lilo_vla/collect_shared_teacher.py`，重新采集 RGB-D 和腕部观测，真实运动到入口，再执行接近、闭合、抬升和保持；失败与录制错误完整保留。`--schedule-file` 明确物体、划分、扰动，`--previous-collection` 在新目录续采，`--filter-handoff-candidates` 在原交接检查拒绝某个抓取目标时继续其他候选。固定已成功验证的**计划目标**时通过 schedule 的 `fixed_selection` 显式记录来源，仍保存本次新网络预测和实际图像，不能将未来执行位姿当作目标。

构建数据必须保留真实的命令启动前观测，不能丢掉夹爪闭合前的空闲帧。当前标签版本 `command_interval_v2` 使用真实相邻 0.2 s 帧区间的实测位移和在区间内生效的夹爪命令；闭合启动帧必需有可训练的连续动作块。执行器完成的固定闭合等待和最终保持不再作为新决策的采样起点，但其真实未来标签仍保留，不跨时间缺口拼接动作。

```bash
"$ARX_NEURAL_PYTHON" scripts/curobo_arx/lilo_vla/build_dataset.py \
  --source /absolute/path/to/successful_train_episode \
  --validation-source /absolute/path/to/successful_other_object_episode \
  --output "$PWD/runs/NEW_DATASET" \
  --state-convention grasp_goal_world_error_v1 --dataset-key arx_shared_goal_v1 \
  --object-split --require-entry --include-command-starts --exclude-gated-dwell-starts
```

归一化只用训练数据。数据检查验证物理成功、同步与连续性、实际入口附近的连接段、计划目标、命令起始观测、重复录制及跨物体划分。动作头训练使用 `finetune_head.py --initial-adapter ...`；完整成对构建、相同初始化/预算训练及实际对照使用 `run_p2_comparison.py`。验证 L1 只用于选检查点，不当作抓取成功证据。

交接资格来自真实重复，不来自逐维 min/max。`handoff_profile.py` 只对已完成资格试验生成配置，`validate_handoff_profile.py` 离线检查实测正例与错误策略/场景/目标/状态版本/观测等拒绝路径。状态来源复算使用执行时的 `../curobo/.venv/bin/python`；不同 SciPy 环境会产生约 2e-10 rad 的旋转复算差异，保留原始报告并用 `recheck_entry_range.py` 在执行环境下按原严格阈值复核，不改物理判断标准。

## EGAD49 状态

旧 `runs/lilo_egad49_20261026_eval01` 按用户要求保持停止：完成 31 例、成功 0，E3 中断、17 例未开始，其中 8 次为共享 A100 显存不足。当前 A1 和瓶子的开发结果不能覆盖或改写这些历史失败，不报告完整 49 例成功率，不自动恢复批量。原始资产、初始条件、评测协议及诊断见 [EGAD 诊断](../../servers/lilo_vla/egad49_diagnosis.json) 与 [衔接尝试计划](../../servers/lilo_vla/README.md)。
