## 参考方法

IK-Geo依据相交轴、平行轴将IK分解为几何子问题；EAIK在此基础上自动识别结构，并提供URDF、齐次变换输入及C++/Python接口。[EAIK项目主页](https://ostermd.github.io/EAIK/) · [IK-Geo官方实现](https://github.com/rpiRobotics/ik-geo)

本地URDF静态核查得到ARX的right_joint22/23/24轴平行，属于值得验证的三平行转轴结构。下一步检查自动分解、原关节限位和工具变换，并将返回分支交给现有独立FK、当前场景碰撞与路径检查。

本轮完成几何核查，实际IK与计时尚待适配。静态数据见[ARX几何核查](/media/sam3-v8-arm-geometry-review.json)。

[论文项目主页](https://ostermd.github.io/EAIK/) · [EAIK代码](https://github.com/OstermD/EAIK) · [V8计划](/iterations/sam3-graspgenx/versions/V8/)
