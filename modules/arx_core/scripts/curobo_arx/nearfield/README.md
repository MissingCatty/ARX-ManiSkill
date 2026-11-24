# ARX 近场 C/D 第一阶段

冻结规范：`research_plan/chat/gemini回复4.md`、`gpt回应1.md` 至 `gpt回应4.md`。

`training/nearfield_arx/contract.py` 实现当前末端坐标系下的 C STEP 与 D REFINE，D 始终相对固定 G0 解码。C 平移/旋转总量上限 25 mm / 5°，D 为 40 mm / 30°。GRASP 单独请求闭合；D 最多两次修正、三次调用，无需运动也消耗修正预算。

`training/nearfield_arx/export.py` 从已提交总账冻结源文件和划分。原始＋direct 使用完全相同的 C/D 观测；control、stitch 单列。G*来自本条轨迹接近完成、闭合之前的实测 TCP。每条轨迹有准确先验及三个固定扰动先验。所有观测不带旧 `state` 的教师目标误差向量；仅保留实测状态、真实命令状态和误差标量。目标绝对坐标保存在审查用源文件中，避免教师目标直接泄露答案。完整几何和种子家族沿用来源划分。

`runtime.py` 将教师标签交给新解释器，在真实 ARX Isaac Sim 中执行。所有运动经过原 cuRobo IK、PRM、轨迹优化、碰撞/限位与相机路径检查；保留各阶段实际命令、回执、接触和原成功判定。D 跳过同一已到位目标时生成独立事件，不伪造执行回执。C 将源连续动作拆成逐步停稳执行，回放用于检验这一时序差异。

`campaign.py prepare` 创建独立源码快照及 8 槽位登记：两个训练物体各 C/D 回放一次、5/20 mm 恢复各一次。恢复先从原入口到偏位点，记录停稳及新观察，再专家纠正；前段仅用于生成状态。源物体发生接触或位移时保留失败，不能沿用静止目标监督。不会启动训练或 48 次主对照。

新登记示例（旧目录不能重复使用）：

```bash
/home/zhaoyechong/projects/curobo/.venv/bin/python -m scripts.curobo_arx.nearfield.campaign prepare --parent <当前原始采集目录> --out <新的执行目录>
/home/zhaoyechong/projects/curobo/.venv/bin/python -m scripts.curobo_arx.nearfield.campaign run <执行目录> --compute-uuid <规划GPU的UUID> --sim-uuid <仿真GPU的UUID>
```

第一版完成后，`prepare --review-of <完成的8槽位目录>` 可创建一次独立的4槽位实现复核登记。它保留首版全部结果及原截止时间，验证两项映射修正：C 的离线真值驱动从当前实测状态计算至教师下一实测路点的 STEP，中间张手动作按原局部运动规划，最后接近仍执行原闭合目标检查；偏位状态生成按原局部接近规划，并保留“发生目标接触或物体位移即拒绝”的实际门控。复核与首版成绩分别报告。

`report.py` 保留原评估回执，并用原评估器按真实请求先后顺序重新核对额外动作的阶段；结果不一致会报错。`diagnose.py` 分别记录单步到位误差、累计教师路径偏差和入口差异。`training/nearfield_arx/recovery.py` 仅从新验收成功的恢复轨迹导出配对增量集，重新取本次 G*，不把制造偏位的前段作为正向动作监督。

普通抓取失败保留并继续；资源、记录、配置或空闲状态异常停止后续调度。不得删除 STOP 后重跑。回放、恢复、训练、正式评测分别计数。
