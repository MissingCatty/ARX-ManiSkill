## 方法

MoveIt Cached IK保存已经求得的关节解。再次求解时，先查询邻近记录作为初值，失败后回到调用方给定的原初值；足够不同的新解可以加入缓存并持久保存。[官方文档](https://moveit.picknik.ai/main/doc/examples/kinematics_configuration/kinematics_configuration_tutorial.html)

V8将已有可达性见证作为候选解来源，先独立复算FK和限位，再检查当前场景；近邻记录辅助求解原目标。比较查询与回退合计成本，首轮缓存冻结。

[官方实现](https://github.com/moveit/moveit2/tree/main/moveit_kinematics/cached_ik_kinematics_plugin) · [V8计划](/iterations/sam3-graspgenx/versions/V8/)
