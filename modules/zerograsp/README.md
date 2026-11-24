# 历史 cuRobo 实验结果

当前 ARX 的安装后运行、抓取和录像用法统一见 [当前用法](../scripts/curobo_arx/README.md)。

下文仅保留其他机器人和旧实验的结果记录，不作为当前运行、配置或验证依据。历史代码、资产配置与原始结果目录未删除；旧安装、启动、调用和调参指南已移除。

## G1-D 旧批次结果

### 已记录结果

在 G1-D 严格桌面、无指尖规划球、完整深度、无试抬和最多三次重规划配置下：

- PickSingle Seed 1-10：6/10 完成物理抬升，成功率 60%；
- PlaceSphere Seed 1-5：2/5 成功，成功率 40%；
- StackCube Seed 1-3：2/3 成功，成功率 66.7%。

这些数字对应本仓库 G1-D 文档中记录的特定仿真、模型和随机种子配置。规划成功与物理抓取成功始终分别统计，不能用三阶段规划成功替代最终任务成功。


## maniskill_curobo/README.md

### 2026-05-15 状态

As of 2026-05-15, Route B is viable on this local machine:

```text
torch cuda: ok
mani_skill + sapien import: ok
curobo import: ok
cuRobo pose planning: ok
cuRobo grasp planning: ok
ManiSkill PickSingleYCB-v1 after cuRobo import: ok
ManiSkill qpos -> cuRobo plan bridge: ok
```

Smoke outputs:

```text
maniskill_curobo/smoke_tests/motion_planning/motion_plan.pdf
maniskill_curobo/smoke_tests/motion_planning/grasp_plan.pdf
maniskill_curobo/smoke_tests/bridge/maniskill_qpos_to_curobo_plan.npz
```

## maniskill_curobo_fangzhou/README.md

The July 13, 2026 Seed 1-10 run planned all three stages for 9/10 seeds and
physically lifted 1/10 objects.  The successful Seed 7 lift was 0.1409 m.  See
`runs/openarm_picksingle_curobo_seed001_010/summary.json` and the contact sheet
in the same directory for per-seed details.

### 历史状态

- The exported ROS `package://lift2/...` mesh paths are converted to absolute
  paths in a generated URDF.
- The robot is registered in ManiSkill as `lift2_visual`.
- The mobile base is fixed on the floor on the opposite side of the table from
  Franka and rotated 180 degrees to face the workspace. Its root is placed at
  `x=0.7069260`, computed from the rotated base bounds and the positive-X table
  edge, leaving a 2 cm gap so the mobile base does not intersect the table.
  Its height is derived from the wheel joint and mesh bounds.
- A 20-DoF rest pose is supplied for the lift, both arms, and both grippers.
- Collision meshes are intentionally removed from this first generated URDF.
  The original STL files are detailed render meshes and are not yet suitable
  as efficient articulation collision geometry.
- Both arms and both grippers also have a generated collision-sphere model.
  The same `center` and `radius` values are written into ManiSkill URDF
  collision elements and a cuRobo-style YAML file.
- The current fit contains 50 physical sphere shapes across 16 moving arm and
  gripper links. A 3 mm conservative radius padding is applied.
- `PickClutterYCBLift2-v1` places the robot in a deterministic clutter seed and
  renders the complete scene.


The sphere model currently covers the two arms and grippers, but not the
mobile base or lift body. Self-collision remains disabled during this first
calibration pass. Replacing Franka for actual grasp execution still requires
a right- or left-arm controller definition, TCP links, gripper mimic control,
and a complete matching cuRobo robot configuration.


## maniskill_curobo_g1d/README.md

On the current RTX 3090, only about 17.2 GiB was free before runtime, while
the BF16 AffordanceVLM checkpoint parameters alone occupy about 15.0 GiB. The
safe simultaneous-residency estimate for AffordanceVLM, ZeroGrasp, ManiSkill,
cuRobo, workspaces, and reserve is about 27.0 GiB. This run therefore follows
the requested fallback: simulator task-target ground-truth segmentation plus
ZeroGrasp, without loading AffordanceVLM. ZeroGrasp is not reloaded between
attempts: model load took 2.00 s; inference took 2.84 s on the first request
and 0.98 s on each later request.

The following paragraphs describe archived trial-lift experiments and are kept
only as historical comparisons; the current runner no longer executes a trial
lift. The recorded Seed 1 run exercised all three physical attempts. Attempts 1 and
2 both completed the full recovery sequence, and the arm selector changed from
right to left after the live recapture. All attempts planned successfully and
had bilateral finger contact, but trial lifts reached only 6.0, 8.1, and
5.0 mm, below the 15 mm retry-success threshold. The report therefore records
the final task outcome as failed rather than treating planning success as a
physical pick success.

Before full-depth-only enforcement, replacing the inherited soft `100 N/m`
all-joint controller with the
Dex1-specific drive above, measured contact increased from roughly
`0.6--1.4 N/finger` to `7.0--7.6 N/finger`. The force-20 Seed 1 rerun fails its
first right-arm trial, completes recovery and live replanning, then succeeds
with the left arm on attempt 2. The full-lift plan raises the object by
`0.113 m`. A `0.5 mm` numerical tolerance is used around the `15 mm` trial-lift
threshold; the successful trial measured `14.978 mm` and is then validated by
the full 13 cm commanded lift rather than being rejected by simulator jitter.
That archived successful run used reduced insertion depth on its selected
candidates and is not representative of the current policy.

With full-depth-only enabled, the strict Seed 1 rerun uses scale `1.0` for every
screened pose. Attempts 1--3 select candidate indices 45, 62, and 7 after
rejecting 45, 62, and 7 earlier full-depth poses, respectively. Trial lifts are
`2.48`, `11.57`, and `7.08 mm`, so all three physical attempts fail. The run
takes about `704 s` wall time because 117 poses are evaluated by cuRobo, of
which 114 are rejected before the three selected plans; this motivates adding a
cheap table/IK prefilter while retaining full-depth execution.

After calibrating the TCP from `0.120 m` to `0.140 m`, the same full-depth Seed
1 run selects candidate indices 0, 1, and 4, evaluating only 1, 2, and 5 poses
per attempt. Attempt runtime falls from `653.7 s` to `53.3 s` (`12.3x` faster),
confirming that most previous tabletop failures were caused by the TCP being
23 mm behind the physical fingertip. The three trial lifts still fail at
`8.37`, `10.72`, and `6.03 mm`; after the reachability correction, the remaining
issue is grasp stability on the extra-large clamp rather than table collision.

The latest Seed 1--10 one-attempt batch limits each scene to 10 full-depth
candidates, removes the four cuRobo fingertip spheres, and uses the complete
table cuboid with `0 mm` inset and `0 mm` back-edge trim. Torso spheres and the
fixed non-selected arm remain active collision geometry. Nine seeds find a
three-stage plan and Seeds 6, 7, and 8 complete the physical lift, for `90%`
planning success and `30%` physical pick success. The batch evaluates 37 poses
total and takes 657 s wall time. All ten per-seed videos were verified as H.264
at `1280x960`. ManiSkill continues to simulate the exact gripper collision
meshes; only the four planning spheres at the fingertips are omitted.

The current no-trial-lift Seed 1--10 batch uses this strict configuration and
up to three attempts per seed. Seeds 1, 2, 6, 7, 8, and 9 succeed, giving a
`60%` physical pick rate. Seed 1 succeeds on attempt 2 after a full-lift failure
and recovery; the other five successful seeds finish on attempt 1. Across the
batch there are 19 perception/planning attempts, 10 executed full lifts, four
physical-failure recoveries, and nine attempts with no strict three-stage plan.
No `trial_lift` trajectory or artifact is present. All ten videos were verified
as H.264 at `1280x960`; total process wall time is 1164 s.


The Seed 1 validation succeeds on its first candidate with the left arm. It
lifts the sphere by `129.14 mm`, releases it in the bin, opens all four fingers,
and retreats. The final horizontal placement error is `3.59 mm`, vertical error
is effectively zero, and the sphere is static, satisfying the original 5 mm
ManiSkill placement criterion.

The Seed 1--5 multi-task check succeeds on seeds 1 and 2 (`40%`). Seed 3 has no
strict three-stage plan; seed 4 has one physical grasp failure followed by two
no-plan attempts; seed 5 records a physical grasp failure and then cleanly
terminates when the displaced target is no longer visible to the RGB-D camera.


Seed 1 selects the right arm and candidate 1, lifts the red cube by `128.63 mm`,
and chooses a 7 cm collision-free release height after rejecting lower release
poses. The final horizontal stacking error is `17.04 mm` under ManiSkill's
`33.28 mm` tolerance; vertical error is below 1 micrometre, the cube is static,
and all four fingers are open. Seed 2 also succeeds with the left arm and a
0 mm release height (`17.85 mm` horizontal error). Seed 3 is laterally outside
the strict reachable workspace and all three attempts have no strict pick plan,
giving `2/3` (`66.7%`) for StackCube. The combined PlaceSphere/StackCube summary
is in `runs/pick_and_place_multitask_summary_20260721/summary.json`.


## Main outputs

- `generated/g1d_collision_spheres.yml`
- `generated/g1d_collision_sphere_metrics.json`
- `config/g1d_left_curobo.yml`
- `config/g1d_right_curobo.yml`
- `runs/curobo_smoke/report.json`
- `runs/maniskill_render/g1d_collision_spheres_maniskill.png`
- `runs/maniskill_render/g1d_curobo_dual_arm_planned_pose.png`
- `runs/maniskill_render/g1d_replaces_openarm_picksingle_seed1.png`
- `runs/picksingle_curobo_seed001/g1d_picksingle_seed001_curobo.mp4`
- `runs/picksingle_curobo_seed001/report.json`
- `runs/rgbd_reverse_edge_lower_seed001_final/zg_input/rgbd.npz`
- `runs/rgbd_reverse_edge_lower_seed001_final/zg_output/recommended_grasp_top1.json`
- `runs/rgbd_reverse_edge_lower_seed001_final/grasp_projection.png`
- `runs/rgbd_reverse_edge_lower_seed001_direct_strict_friction/g1d_rgbd_zerograsp_seed001_curobo.mp4`
- `runs/replanning_seed001_direct_oracle_final/g1d_seed001_closed_loop_replanning.mp4`
- `runs/replanning_seed001_direct_oracle_final/report.json`
- `runs/replanning_seed001_direct_oracle_force20_tol05mm/g1d_seed001_closed_loop_replanning.mp4`
- `runs/replanning_seed001_direct_oracle_force20_tol05mm/report.json`
- `runs/replanning_seed001_direct_oracle_force20_fulldepth/g1d_seed001_closed_loop_replanning.mp4`
- `runs/replanning_seed001_direct_oracle_force20_fulldepth/report.json`
- `runs/replanning_seed001_force20_fulldepth_tcp140/g1d_seed001_closed_loop_replanning.mp4`
- `runs/replanning_seed001_force20_fulldepth_tcp140/report.json`
- `runs/picksingle_seed001_010_oneattempt_top10_tableinset5mm_tcp140/summary.json`
- `runs/picksingle_seed001_010_oneattempt_top10_tableinset5mm_tcp140/summary.csv`
- `runs/picksingle_seed001_010_no_fingertip_spheres_strict_top10_tcp140/summary.json`
- `runs/picksingle_seed001_010_no_fingertip_spheres_strict_top10_tcp140/summary.csv`
- `runs/picksingle_seed001_replanning_no_fingertip_spheres_strict_top10_tcp140/report.json`
- `runs/picksingle_seed001_replanning_no_fingertip_spheres_strict_top10_tcp140/g1d_seed001_closed_loop_replanning.mp4`
- `runs/picksingle_seed001_010_direct_full_lift_replanning_no_fingertip_spheres_strict_top10_tcp140/summary.json`
- `runs/picksingle_seed001_010_direct_full_lift_replanning_no_fingertip_spheres_strict_top10_tcp140/summary.csv`
- `runs/place_sphere_seed001_workspace_shift10cm_canonical_replanning_strict_top10/report.json`
- `runs/place_sphere_seed001_workspace_shift10cm_canonical_replanning_strict_top10/g1d_place_sphere_seed001.mp4`
- `runs/stack_cube_seed001_final_defaults_strict_top10/report.json`
- `runs/stack_cube_seed001_final_defaults_strict_top10/g1d_stack_cube_seed001.mp4`
- `runs/pick_and_place_multitask_summary_20260721/summary.json`

## maniskill_curobo_real/README.md

#### M0/M3/M4 seed1-200 实测结果

完成日期：2026-06-12。

为了公平比较运行时间，M3 也使用与 M4 相同的持久化 ManiSkill 环境、持久化
cuRobo planner 和 `update_world` 方式重新运行。优化后的 M3 仍然得到与历史正式
基线完全相同的 `101/197` 次 lift 成功，因此没有改变成功率基线。

M0 是 ManiSkill 仿真真值碰撞世界 baseline。M0、M3 和三个 M4 版本都使用同一批
ZeroGrasp full-depth top-k 候选、相同 depth fallback、相机、settle steps、
workspace 和执行参数。seed `1、40、169` 缺少可复用的 ZeroGrasp candidate，
因此每个版本实际完成 `197/200`。

| 版本 | 碰撞表达 | Lift 成功 | 相对 M3 | 改善 / 退化 seed | 端到端平均耗时 | 相对 M3 耗时 |
|---|---|---:|---:|---:|---:|---:|
| M0 | ManiSkill truth collision | `94/197 = 47.72%` | `-3.55 pp` | - | `6.40 s` | `0.79x` |
| M3 | instance AABB | `101/197 = 51.27%` | baseline | - | `8.13 s` | `1.00x` |
| M4-A | yaw OBB | `103/197 = 52.28%` | `+1.02 pp` | `8 / 6` | `7.81 s` | `0.96x` |
| M4-B | convex hull mesh | `8/197 = 4.06%` | `-47.21 pp` | `0 / 93` | `16.01 s` | `1.97x` |
| M4-C | 10 mm voxel ESDF | `111/197 = 56.35%` | `+5.08 pp` | `20 / 10` | `7.75 s` | `0.95x` |

这里的端到端耗时包含该 seed 的碰撞世界构建和抓取规划/执行，不包含已经
离线复用的 ZeroGrasp 推理。M0 不需要构建点云 world，因此比 M3/M4 更快。
平均耗时也会受失败类型影响：M4-B 经常依次测试 top-20 候选后才宣布失败，
因此既慢又没有执行抓取。

失败分布：

| 版本 | Candidate selection failed | Grasp planning failed | Lift planning failed | Object not lifted | Execution failed |
|---|---:|---:|---:|---:|---:|
| M0 | 39 | 2 | 0 | 44 | 18 |
| M3 | 49 | 2 | 0 | 44 | 1 |
| M4-A | 47 | 0 | 0 | 46 | 1 |
| M4-B | 184 | 1 | 0 | 3 | 1 |
| M4-C | 24 | 3 | 1 | 57 | 1 |

结论：

- **M0 不是当前成功率上限。** ManiSkill 真值碰撞世界在这批候选上只有
  `94/197 = 47.72%`，低于 M3/M4-C。原因是 M0 对仿真真值碰撞更硬，
  candidate selection 和 execution failed 更多；更真实的碰撞世界不一定给
  当前 ZeroGrasp 候选带来更高执行成功率。
- **M4-C 是当前最好的 M4 版本。** 它把 lift 成功率从 `51.27%` 提高到
  `56.35%`，净增加 10 个成功 seed，端到端耗时没有衰减。
- **M4-A 可以作为低风险 fallback。** 它保持 cuboid 查询，成功率小幅提升，
  运行成本与 M3 基本相同。
- **M4-B 不应直接作为默认 world model。** mesh 文件经过 watertight、winding、
  volume 和坐标范围检查，但硬凸包使 184 个 seed 在 top-k candidate selection
  阶段失败。问题不是 mesh 文件损坏，而是闭合凸包作为硬碰撞体对贴近物体的抓取
  过于保守。
- M4-C 仍使用 oracle instance segmentation 和单帧点云，不等于已经完成真机
  感知迁移。下一步应把 instance 输入换成真实实例分割，并进入 M5 多视角
  实例融合 ESDF。


#### seed1-200 正式结果

本轮已经完成：

- 场景：`PickClutterYCB-v1`
- seed：`1-200`
- 重建输入：settle 20 step 后的 RGB-D 与 `all-objects` oracle instance mask
- 重建完成：`200/200`
- 重建实例：`1153`
- 重建点总数：`8,409,921`
- 抓取候选、top-k、depth fallback 和执行协议与 ManiSkill 点云实验一致
- seed `1`、`40`、`169` 缺少可复用抓取候选，因此每个阶段实际完成 `197` 个

ZeroGrasp 重建点云内部比较：

| Stage | Lift success | 相对 M3-ZG | 主要结果 |
|---|---:|---:|---|
| M3-ZG AABB | 91/197（46.19%） | baseline | 完整重建表面的轴对齐范围会放大几何误差 |
| M4A-ZG yaw OBB | 100/197（50.76%） | +4.57 pp | 比 AABB 更贴合物体平面朝向 |
| M4B-ZG convex mesh | 12/197（6.09%） | -40.10 pp | mesh 碰撞世界仍过于保守，不适合作为主线 |
| M4C-ZG voxel ESDF | 108/197（54.82%） | +8.63 pp | 本轮最佳 ZeroGrasp 重建点云版本 |

与原 ManiSkill depth 点云逐 seed 公平比较：

| Stage | ManiSkill depth | ZeroGrasp reconstruction | 差值 |
|---|---:|---:|---:|
| M3 | 101/197（51.27%） | 91/197（46.19%） | -5.08 pp |
| M4A | 103/197（52.28%） | 100/197（50.76%） | -1.52 pp |
| M4B | 8/197（4.06%） | 12/197（6.09%） | +2.03 pp |
| M4C | 111/197（56.35%） | 108/197（54.82%） | -1.52 pp |

这说明 ZeroGrasp 重建点云已经可以替换 ManiSkill depth 点云来构造 M3/M4
碰撞世界。M4C 的 lift success 只损失 3 个 seed，性能已经非常接近原实验。
M3 下降更明显，原因是完整表面的极值点会直接扩大 AABB；M4A 和 M4C
对点云重建误差更稳健。M4B 虽有小幅提升，但绝对成功率仍太低。

当前仍不是完整真机链路：实例划分和目标标识继续使用 ManiSkill oracle
segmentation。几何点的位置与表面形状来自 ZeroGrasp 重建，不使用
ManiSkill depth 点云或 actor collision shape。

正式产物：

- 多实例重建：
  `runs/pickclutter_zerograsp_reconstructions_seed1_200`
- M3/M4-A/B/C 场景、框图和执行结果：
  `runs/pickclutter_zerograsp_reconstruction_m3_m4abc_seed1_200`
- ZeroGrasp 点云内部 M3-vs-M4：
  `runs/pickclutter_zerograsp_reconstruction_m3_m4abc_seed1_200/comparison`
- ManiSkill-vs-ZeroGrasp 逐 seed 比较：
  `runs/pickclutter_zerograsp_reconstruction_m3_m4abc_seed1_200/source_comparison`


## 可执行性历史对照

### 2.1 已有实验依据

在 `PickSingleYCB-v1` seed 1-200 的 zero residual 实验中：

- 33 个场景在 grasp 阶段规划失败；
- 保留桌面碰撞时，这 33 个目标全部找不到无碰撞 IK；
- 移除桌面碰撞、保留机器人自碰撞后，33/33 全部能够规划；
- 目标物体已从 cuRobo 障碍物中排除，因此失败源是夹爪或机械臂与桌面的碰撞，不是目标物体被误当成障碍；
- 这些失败中包含较多平放的刀、叉、勺、剪刀、铲子和螺丝刀。

在 full depth 实验中还发现：

- 将 ZeroGrasp depth 同时加到 pre-grasp 和 grasp，会让部分物体在接近阶段被撞动；
- 固定原始 pre-grasp，只对最终 grasp 应用 depth 后，接近过程更稳定；
- 对 grasp depth 使用 `100% -> 75% -> 50% -> 25% -> 0%` 自动回退，15 个已知退化场景全部恢复为成功 lift；
- 不同物体选择的有效 depth 比例不同，因此不能用一个固定 depth 阈值解决所有情况。

这些结果说明，zero 与 full depth 不应被看作两个互不相关的执行模式。它们都需要经过同一个“可执行性检查与修正模块”。


## License

ZeroGrasp 代码和资源遵循 [LICENSE.md](LICENSE.md)；外部机器人描述、cuRobo 和模型权重遵循各自上游许可证。
