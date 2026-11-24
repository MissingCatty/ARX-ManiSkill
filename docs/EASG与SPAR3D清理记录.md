# EASG与SPAR3D清理记录

2026年11月24日，按用户确认清理EASG云端近场抓取路线和SPAR3D重建方法。机器人、200场景、物理参数和成功判据保持原值，本轮不运行抓取。

| 内容 | 处理 |
| --- | --- |
| 专用源码与入口 | 删除modules/easg、ManiSkill的easg_pipeline及ARX的EASG执行、提示词、观测入口 |
| 云端控制与配置 | 删除独立适配分支中的EASG控制器、抓取调度器、验证脚本及云端配置；保留通用cuRobo规划端点 |
| 专用资产和原始产物 | 删除simulation/assets/models/source_easg和tmp/easg |
| 共享候选工具 | 移到modules/maniskill/planning_common，保留候选来源核验、坐标转换和相应离线检查 |
| 指垫几何 | 原字节移到simulation/robots/arx/finger_pad_geometry.json，更新现用引用 |
| 动作坐标数学 | 移到arx_core的action_frame.py；独立规划诊断读取公共TCP43配置 |
| 模块与备份登记 | 移除EASG注册，备份增加显式排除项；新版资源位置登记在configs/resource_snapshot.json |
| SPAR3D | 删除专用源码、约7.96 GiB权重、独立环境、50例原始输出、预热与专用源码快照，移除模块和环境登记 |
| modules下的旧兼容入口 | 删除modules/tmp、modules/asset和modules/compat_tmp的7个软链接；原数据、模型和共享环境保留，路径交由公共读取层解析 |
| 其他待确认项 | 原清单第2–8项继续保留 |
| 网站 | 已发布资源副本保留，不依赖被删目录 |

逐路径删除记录、共享几何字节核对、离线检查和备份衔接记录放在tmp/cleanup/easg_retirement_20261124_01。旧备份的元数据、EASG和SPAR3D包退役，其余未变化的资源分片复用到resources_20261124_02；云端完成状态以新版目录中的COMPLETE.json为准。

本机专用源码、权重、环境和原始产物共清理约9.11 GiB。公共检查23项、共享规划检查19项、源文件门控2项通过；工程预检及2406个冻结场景文件核验通过，机器人/相机/碰撞球/TCP和场景索引散列保持。网站发布副本未改动。
