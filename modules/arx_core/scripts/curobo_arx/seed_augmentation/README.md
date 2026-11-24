# ARX 成功种子入口扩充

用户于2026-11-04确认：保留成功抓法，扰动预抓取入口H′，连接到原成功接近起点S，再复用原approach/close/lift/hold。每条新轨迹重新实际执行并录制腕部RGB-D，旧图像和成功结论不拼接进新数据。

当前持续扩充登记为 `runs/seed_augmentation_production_20261104_03/`；读取其state.json定位当前批次。同一个 `arx-graspness-watchdog.service` 同时维护原始采集和独立扩充，固定看护登记为 `runs/graspness_watchdog_20261104_04/`。71项离线检查及固定快照内模块导入检查通过。操作与恢复协议见 `research_plan/docs/seed_augmentation_production_watchdog_20261104.md`。

该包使用独立planner队列和ARX World，运行代码与原始采集隔离，由同一看护的独立线程维护。持续批次入口为 `production.py`，固定源码执行入口为 `entrypoint.py`。保留所有旧登记、STOP、失败和生产累计预算。没有新训练模型，也没有热修改原始生产快照。

已完成小批验证 `runs/seed_augmentation_20261104_03/` 保留22次结果：21条合格轨迹，含control 8、stitch 7、direct 6；另一次direct原到位门槛拒绝。该批不重启、不修改原24次/4小时总预算；持续生产另立上限2000槽，截止沿用原始采集，逐批选择新种子，排除旧批四个种子ID。

小批验证取牛奶盒、葡萄干盒、芥末酱瓶、砖块的成功静止种子；持续生产沿用每种子两次复现、四次stitch/direct扰动的流程，每批至多四种子。第二种扰动反转方法顺序。control失败或结果未知时，后续槽列为未激活，不计抓取失败、不重试。意外进程退出后只续接从未开始的槽位；不宣称已覆盖原来无G的摆放。

stitch连接器以原轨迹起点的关节构型为目标，沿用原target-avoiding IK/PRM/轨迹优化、限位、碰撞和相机检查。连接后检查实际关节、速度、夹爪及物体状态再派发后段。原后段除新执行ID外，关节位置、时间、夹爪值及其他控制字段逐项一致；运动段在当前原ARX模型中重新检查密集碰撞、限位、时序导数及相机路径。失败仍按原门槛记录。

额外命令名为 `approach_connect`，在请求总账和执行回执中显式保留；训练阶段语义为approach。独立检查确保额外连接的规划证书、执行、到位、连接状态以及原后段payload齐全，并将检查绑定到新数据验收。始终使用原五阶段/双指接触/连续至少3秒每次抬升采样超过6厘米/禁止接触/图像与完整性判据。机器人、标定、初始条件、相机安装和物理配置冻结。

```bash
systemctl --user status arx-graspness-watchdog.service
cat runs/seed_augmentation_production_20261104_03/state.json
../curobo/.venv/bin/python -m unittest scripts.curobo_arx.data_factory.test_watchdog \
  scripts.curobo_arx.data_factory.test_augmentation_watchdog \
  scripts.curobo_arx.seed_augmentation.test_contract
```

GPU由预检绑定实际UUID，可同卡容纳独立planner和单环境仿真。需要原注册GPU显存和至少40GiB可用主机内存才能启动World；资源不足时保留登记，不抢占其他任务。先使用本地空间并保留1GiB下限，源Ceph数据只读挂载。无需Graspness新推理，G来自已通过原验收的成功源。

统计原始成功、合格示教、实际运动、未激活、未执行和未知分别报告；增加的是入口覆盖，物体摆放仍为种子原摆放。训练划分继承源物体角色，所有衍生数据不可跨训练/验证分组。

`report.py`只读导出已提交成功的scoped ledger视图，需要时另导出包含实际连接段的视频；不会把旧图像拼入轨迹。原生连续动作验收与Show-Harness转换分别检查。此前只产出GRASP/DONE的问题已有独立v4动作接口修复及77样本实测导出，见 `runs/show_harness_motion_fix_20261104_01/`；不把新采集的原生成功直接等同于全部图像/mask已通过训练转换，也不据此启动训练。
