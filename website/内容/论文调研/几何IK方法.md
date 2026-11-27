## 参考方法

IK-Geo依据相交轴、平行轴将IK分解为几何子问题；EAIK在此基础上自动识别结构，并提供URDF、齐次变换输入及C++/Python接口。[EAIK项目主页](https://ostermd.github.io/EAIK/) · [IK-Geo官方实现](https://github.com/rpiRobotics/ik-geo)

实际ARX URDF的第22、23、24关节轴平行，第25、26关节轴相交，可以按“三平行轴＋两相交轴”分解。左右臂分别读取原关节限位和工具变换；普通位置最多返回8组分支，奇异位置保留原数值IK回退。

已有CPU版EAIK与GPU张量版两种实现。CPU版D组完成20次完整规划，累计慢6.14%，未采用；GPU版将256个目标的分支同时求出，再交cuRobo批量做871球FK及碰撞检查。GPU版左右臂共8192个已知可达目标全部找回原分支，329个EAIK精确参考分支全部匹配。

固定256目标的可靠版解析IK约1.25毫秒。现已完成第七项接入和50次完整规划对照，全部规划通过；共享场景版两组平均4.240秒与4.377秒，没有整段加速收益。98次候选检查中84次抓取终点仍回退原IK，另有约1.855秒模型与场景绑定。解析实现保留为独立入口，常用入口继续原IK。静态核查、CPU对照、GPU微计时与完整规划结果见[V8结果页](/iterations/sam3-graspgenx/versions/V8/)。

[论文项目主页](https://ostermd.github.io/EAIK/) · [EAIK代码](https://github.com/OstermD/EAIK) · [V8计划](/iterations/sam3-graspgenx/versions/V8/)
