# ManiSkill 中的 ARX + ZeroGrasp + cuRobo

用户批准的首批多物体基线只包含五类：开阔桌面、靠近桌边、稀疏混放、中度杂乱、部分遮挡。40 个 YCB 模型各进入五类一次，共 200 例；绕障通道场景不在本批中。分配清单见 `plans/arx_grasp_baseline_v2_five_layouts/`。批量入口是 `baseline.py`，下方 `run.py` 的 5/10 例入口保留用于小规模旧实验。

当前后台任务为 `maniskill-arx-baseline-parallel-20261116-01.service`，结果目录为 `outputs/arx_baseline_20261116_parallel_01/`，主日志为 `logs/arx_baseline_20261116_parallel_01.log`。`progress.json` 显示两个场景工作进程的当前任务，`summary.json` 按布局与物体类别分别汇总。主分母固定为 200，场景接入失败、规划失败、实际执行失败均保留；每个任务最多一次抓取，不自动重试未知执行。

用户要求场景并行、共用一条 ZeroGrasp＋cuRobo 链路。`../baseline_parallel.py` 在两个独立进程中各创建一个 ARX 场景，分别使用登记的 GPU 0、1 UUID；所有推理及规划阶段通过同一个跨进程文件锁排队，共用 GPU 2 UUID，同一时刻最多一个后台请求。某场景执行已规划动作时，另一个场景可以使用共享链路。各场景的物理时钟独立，相机和运动执行逻辑沿用冻结入口。`backend_requests/` 保存排队、执行起止时间，可检查共享链路没有重叠请求。几何接入检查串行登记，以保持近重复检查完整。

原串行任务 `maniskill-arx-baseline-20261116-01.service` 已正常跑完当前试例后退出，原目录 `outputs/arx_baseline_20261116_01/` 和 STOP 保留。新接续引用其 17 例完成记录（5 例成功）并只调度剩余 183 例，累计预算仍为 200；`continuation.json` 保存交接、原始结果与证据散列。机器人、物体物理、场景生成、网络/规划算法及独立判定源文件没有修改，本次只变更调度及 GPU 分配。

用户新增要求：每次推理输出本次目标得分最高的 10 个抓取位姿，分数从高到低、颜色从暖红到冷蓝。独立 CPU 后处理 `../grasp_visuals.py` 读取已保存的 RGB-D 和 `raw_grasps.npy`，在每例自己的 `trials/<例ID>/zg_outputs/` 中输出真实点云三维图 `grasp_top10.png`、原始 RGB 投影图 `grasp_top10_rgb.png` 及候选/分数/来源散列 `grasp_top10.json`。严格取原始网络分数前 10，不增加去重或可执行性筛选；不足 10 个时按实际数量。没有保存推理结果的试例在批次出图索引 `grasp_visualizations/index.json` 中说明，不虚构位姿。服务 `maniskill-arx-grasp-visuals-20261116-01.service` 自动补图并监听后续推理；它不修改冻结执行入口，也不新增推理或抓取。单例命令 `grasp_visuals.py --trial <试例目录>` 默认也输出到该试例的 `zg_outputs/`。

批次先冻结 200 个目标/布局/种子分配及源文件，再逐例进行纯场景接入预检。每例在发出动作前保存具体初始位姿和碰撞几何。近重复几何会在接入阶段拒绝，正式抓取结果不参与场景筛选。部分遮挡由真实邻物造成，静置后测量遮挡率 20%–40%；目标投影边界须完整位于腕部图像内。观察准备使用固定规则的相机偏移，朝向登记布局中的目标包围盒中心，不读取抓取候选或成功反馈。

所有干扰物的 SAPIEN 原生凸碰撞块会按原顶点、尺度及局部变换导出到内容寻址缓存，源文件和缓存散列均核验；cuRobo 按实测物体位姿加载。观察准备规划包含目标及全部干扰物，抓取规划按原链路排除目标、保留所有干扰物。模型、密度、摩擦、机器人增益、限位及成功门槛保持登记值。每例独立进程运行，避免连续创建 SAPIEN 世界造成渲染资源积累；资源不足时等待，不结束其他任务。

```bash
systemctl --user status maniskill-arx-baseline-parallel-20261116-01.service
cat outputs/arx_baseline_20261116_parallel_01/progress.json
```

`ARXPickYCB-v1` 使用本地 ManiSkill 的桌子和 YCB 物体，机器人为 COOLAS-ARX 当前 `arx_x5_lift`。完整链路是新 RGB-D → 当前 ZeroGrasp 权重推理 → cuRobo IK、PRM 与原有轨迹优化器 → SAPIEN 关节控制 → 双指接触、抬升及保持检查。

从 `/home/zhaoyechong/projects` 运行，先用 `nvidia-smi` 查询空闲显卡，再填写 UUID：

```bash
MANISKILL_GPU_UUID=GPU-仿真卡UUID bash ManiSkill/run.sh ManiSkill/arx_pipeline/run.py \
  --output ManiSkill/outputs/新的批次目录 \
  --count 5 \
  --neural-gpu GPU-推理卡UUID \
  --planner-gpu GPU-规划卡UUID
```

默认登记 5 例，按固定顺序为两个汤罐、两个金枪鱼罐和一个苹果；`--count 10` 才会覆盖汤罐、金枪鱼罐、苹果、柠檬、杯子各两个固定种子。`--limit 1` 只执行下一例，后续同一命令可继续剩余例；源代码、依赖与权重哈希必须一致。已经开始但没有结果的试验不会自动重跑。存在 `STOP` 的批次拒绝继续；运行中可用 `budget_amendment.json` 的 `effective_attempt_budget` 缩减尚未启动的任务数量。

`--camera right_wrist --count 1 --limit 1` 使用标定右腕相机完成单次试验，默认 `--camera head`。两路相机均按 COOLAS-ARX `controllers.yaml` 的安装变换、内参和 640×480 分辨率创建；选择的相机及采集时实测外参会写入登记和状态文件。腕部试验额外输出 `right_wrist.mp4`，`execution.mp4` 是场景录像；运动时的腕部录像不替代推理时保存的原始 RGB-D。

2026-11-16 右腕单次试验完成，目录 `outputs/arx_wrist_grasp_20261116_01/`：新腕部 RGB-D 生成 3,489 个 ZeroGrasp 候选，实际完成预抓取、接近、双指闭合、抬升和保持；最小保持抬升约 11.91 cm，保持 3.025 s，禁止接触为零，五段均满足严格关节收敛。初始图像中的罐子上缘有裁剪，原始图像与掩码保留。这是单个 ManiSkill 试验结果，详见该目录的 `summary.json`、`verification.json` 和两路录像。

每例保存相机 RGB/深度/实例掩码、ZeroGrasp 原始候选、规划模型与轨迹、实测状态、120 Hz 执行和接触记录、20 fps 视频、失败原因。根目录 `registration.json` 登记配置，`summary.json` 分别统计规划通过、物理执行和最终成功。成功需要两个右指接触目标、所有执行段收敛、无禁止接触，并且连续至少 3 秒保持物体高于初始位置 6 cm。

模型由 `export_robot.py` 从当前 USD 导出，源模型不被修改。SAPIEN 导入时合并仅用于表达关节坐标系的辅助链接，避免额外质量；保留导出的惯量、关节限位、驱动模式、增益、速度限制和手指碰撞分块。cuRobo 使用当前冻结的开指/闭指模型，并在执行前验证 FK、轨迹起终点和密集碰撞。到达预抓取后从实测状态重规划接近，闭指后按实测指宽规划抬升。

这是新建的 ManiSkill 桌面测试，不是历史 Isaac Sim 场景回归，也不声称两引擎物理等价。物体 XY、yaw、机器人基座和初始关节状态见每例 `initial_conditions.json`。物理使用 CPU、渲染使用 GPU。当前入口按用户要求从 COOLAS-ARX 的 `static_pick/reference/historical_reference/initial_setup.json` 读取抓取链路恢复的关节初态，并使用 `controllers.yaml` 定义的 `body4/head_color_camera` 作为默认 ZeroGrasp RGB-D 输入，分辨率 640×480；目标掩码来自仿真实例分割。场景录像按用户后续要求使用机器人左前方近景：相机相对基座 `[1.20, 0.80, 1.20]` m，看向 `[0.60, -0.20, 0.60]` m，分辨率 1920×1080；`+X` 为机器人前方、`+Y` 为左方。自碰撞沿用当前关闭设置；末端收敛使用严格 0.005 rad 检查。

可对已有执行记录重新选择场景录像视角，无需新增抓取。`render_recorded.py` 使用原记录每个 20 fps 时刻的实测关节角和物体位姿渲染，保持物理停止；输出会标明这是状态重渲染，原执行录像及结果保留。右腕成功案例的左前方录像在 `outputs/arx_wrist_left_front_20261116_01/left_front.mp4`。

```bash
MANISKILL_GPU_UUID=GPU-渲染卡UUID bash ManiSkill/run.sh ManiSkill/arx_pipeline/render_recorded.py \
  --trial ManiSkill/outputs/arx_wrist_grasp_20261116_01/00_005_tomato_soup_can \
  --output ManiSkill/outputs/新的重渲染目录
```

关节初态按源 reset 的规则裁剪到原限位。升降约 0.22 m、左腕关节约 −0.249 rad、右腕关节约 −0.245 rad；这与通用服务器的零位 home、`grasp.yaml` 的启动升降值不同，依据是当前 `record_clear_bottle.sh` → `reproduce_setup.py` 实际恢复的记录状态。相机挂载、局部变换、K、0.05–10 m 裁剪范围和深度毫米单位跟随当前仿真相机定义。光栅化后端仍为 SAPIEN default，与 Isaac PathTracing 的图像外观不同；相机图像未额外施加畸变，和当前 IsaacRgbdCamera 渲染路径一致。

无需执行抓取即可核验：

```bash
MANISKILL_GPU_UUID=GPU-仿真卡UUID bash ManiSkill/run.sh ManiSkill/arx_pipeline/check_alignment.py \
  --output ManiSkill/outputs/新的初态相机核验目录
```

核验输出 `alignment.json`、`head.png`、`depth.png`、`mask.png` 和 `overview.png`，分别检查实际关节状态、驱动、内参、相机世界变换以及目标可见性。相机/初态对齐通过与目标可见性分别记录；目标少于 50 像素时批量入口拒绝推理。

按用户后续要求，当前新场景桌高从约 0.92 m 调为 **0.50 m**：桌面仍在世界 Z=0，地面和机器人基座随桌高设置，桌子 XY 尺寸不变。桌子视觉与碰撞盒的高度同步修改；cuRobo 直接读取实测状态中登记的 `scene_cuboids`，拒绝缺少场景几何的输入，防止沿用旧桌高。原关节初态与头部相机标定不变。记录见 [高度核验](../outputs/arx_height_20261116_01/alignment.json)。

这次高度调整的测试是初始化、静置、新图像推理和规划检查，未执行抓取。相机高于桌面约 0.271 m，目标下部仍有画面裁剪，见 `target_fully_in_view`；规划通过不能当作物理抓取成功。

2026-11-16 用户要求调整到适合抓取的高度后，当前桌高设为 **0.50 m**。保留桌面世界 Z=0 和 XY 尺寸，调整桌体碰撞盒及视觉模型的竖直尺寸，并让地面和机器人基座落在相应地面高度；17 个关节初值和相机标定保持一致。原始 0.9196429 m 桌高的旧记录保留。采用原场景约 0.65 m 的基座相对桌高时，这个 ManiSkill 整块桌体碰撞盒仍与初始夹指相交；0.50 m 初态可以通过原碰撞球检查。

桌体和地面几何随实测状态写入 `scene_cuboids`，cuRobo 直接读取同一几何，缺少几何时拒绝猜测桌高。修改桌高后，不可复用旧批次的状态和规划模型。

若只需验证新高度的初态保持、新 RGB-D 推理和规划，可运行 `check_grasp_setup.py --output 新目录 --neural-gpu UUID --planner-gpu UUID`。该检查保持初态 2 秒，记录接触和图像裁剪情况，生成新的 ZeroGrasp 候选并验证规划，不执行抓取动作。

`assets/arx/manifest.json` 保存导出资产哈希和源配置；集成排障的 pilot、规划回放与正式批次分目录保留，不能混合计算成功率。

2026-11-16 批次按用户中途要求缩减到 5 例并停止，见 [汇总](../outputs/arx_batch_20261116_01/summary.json) 和 [独立核验](../outputs/arx_batch_20261116_01/verification.json)。两个汤罐、两个金枪鱼罐、一个苹果均完成新 RGB-D 推理，原始候选数依次为 7,557、7,184、3,138、3,061、6,413；五例都以 `no_feasible_network_candidate` 结束，规划通过 0 次、物理抓取执行 0 次。不能将这些结果报告为完成了五次物理抓取。失败集中在候选 IK/场景碰撞、闭指碰撞及预抓取/抬升可达性筛选。

上述旧批次使用自定升降 0.46 m、左右腕 1.5 rad 和外部 `base_camera` 推理。它的执行快照和统计保留，不能当作当前初态、标定头部相机的测试结果。本次对齐只做初始化与渲染核验，不追加抓取次数。

本批保留原始 10 例登记和 5 例预算修订。停止信号到达前，串行 runner 创建了第六例的空目录，但尚无图像采集或推理结果；该目录留有取消回执，不计入五例。新入口已增加 `--count` 和 `STOP` 检查；批次 `source/` 保留的是实际执行时的源文件，后续数量控制修复另有回执。
