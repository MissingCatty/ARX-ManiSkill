# ARX Graspness 教师采集

使用固定版本 `graspnet/graspness_unofficial` 的 Graspness/GSNet、公开预训练权重和真实ARX头部深度推理，生成GraspNet格式的抓取G；复用原ARX工具坐标映射，沿抓取接近轴后退10厘米得到H。cuRobo从实测机器人初态驱动到H，确认原收敛和无目标接触后，才开始记录实际H的腕部RGB-D以及接近、闭合、抬升、保持轨迹。

每次最多预筛8个高分候选，得到2个通过G/抬升/H端点检查的候选即可进入执行；未检查的候选单列deferred，不计失败。预抓取关节解只作数值IK种子，轨迹始终从真实机器人当前状态规划，完整目标避碰和实际物理验证不省略。记录实时相机裁剪范围，保持近裁剪/穿模拒收。新摆放先验证参考姿态，再按位置/朝向覆盖轮换；所有旧目录、STOP和失败保留。

源码是项目维护的开源复现版本，不称为授权AnyGrasp。权重已从第三方公开镜像取得并通过严格架构加载；SHA256为 `a0b9d85b300e40c76b66d092de62138d74757d6b7c2037f0d364f33960c54aa9`。下载回执记录仓库/固定版本，未验证与触发配额的原作者Google Drive文件逐字节一致。禁止把规划或网络预测计作实际示教。

2026-11-04成功率研究已完成当前有界检查，结论见 `research_plan/GRASPNESS_SUCCESS_ACCEPTANCE_20261104.md`。`multi_angle.py`、`multi_view.py`、`finger_overlap.py`、`goal_refinement.py`、`load_lift.py` 均为显式登记才启用的研究方法，未替换生产默认值。29项相关离线检查通过；M06实际对照原4/8、新4/8，M07未满足实抓激活条件。当前原样续采从 `runs/graspness_household_20261104_resume_01/` 起由新看护登记承接，实际目录以 `runs/graspness_watchdog_20261104_01/state.json` 为准。

先核对依赖及权重、用保存的真实ARX相机帧检查推理，然后从仓库根目录新建登记：

```bash
../curobo/.venv/bin/python -m scripts.curobo_arx.graspness_teacher.prepare \
  --out runs/graspness_teacher_new --max-attempts 12 --wall-hours 2 \
  --only-object mustard_bottle --only-object household_hope_milk_carton --only-object household_hope_corn_can
../curobo/.venv/bin/python -m scripts.curobo_arx.data_factory.launch \
  runs/graspness_teacher_new --compute-gpu 2 --sim-gpus 3
```

原机器人、相机标定/安装/内参、场景物理和>6厘米/连续3秒保持判据不改。新教师采集单独登记右腕渲染近裁剪从5厘米改为1毫米，远裁剪仍为10米；保留相机进入物体或距表面不足1厘米的拒收。只在内存场景中更改该属性，保存前后回执与实际逐帧读数，不修改源USD或历史评测。只有原物理、记录链及图像准入同时通过的数据入库；记录/资源/配置故障停止。实际进度以运行目录状态、账本和回执为准。
