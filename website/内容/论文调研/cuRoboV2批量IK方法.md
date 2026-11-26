## 方法与接入点

cuRobo提供GPU并行运动学、IK与碰撞检查。V8取用本地已有的批次IK接口，让每个候选占独立的一行，再按GraspGenX原排名检查路径。[官方实现](https://github.com/NVlabs/curobo/blob/main/curobo/_src/solver/solver_ik.py)

本地辅助IK仍配置为一次一个目标。计划分别预热16、32目标容量，保留每目标求解预算和返回分支，记录CPU调用、GPU计算与完整请求耗时。

[官方项目主页](https://nvlabs.github.io/curobo/latest/) · [技术报告](https://nvlabs.github.io/curobo/latest/technical_reports.html) · [V8计划](/iterations/sam3-graspgenx/versions/V8/)
