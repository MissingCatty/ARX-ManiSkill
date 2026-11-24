# ARX-ManiSkill

ARX 机器人在 ManiSkill 中的感知、抓取规划、实验评估与成果展示工程。

| 入口 | 内容 |
| --- | --- |
| [工程目录与存放规范](docs/工程目录与存放规范.md) | 代码、仿真资源、权重、运行产物和网站分别存在哪里 |
| [场景与共享三维资产整理](docs/场景与共享三维资产整理.md) | 最新200场景、共享模型库与场景加载接口 |
| [混元重建方法清理记录](docs/混元重建方法清理记录.md) | 两套混元方法退役与SPAR3D待确认项 |
| [模块接口与扩展指南](docs/模块接口与扩展指南.md) | 模块边界、输入输出、如何替换和新增方法 |
| [安装与运行](docs/安装与运行.md) | 环境、预检、运行入口和迁移后的路径绑定 |
| [网站独立发布规范](docs/网站独立发布规范.md) | 发布时复制资源，单独保留网站也能运行 |
| [网站媒体压缩与加载优化](docs/网站媒体压缩与加载优化.md) | 图片、视频压缩结果和后续自动发布方式 |
| [Git仓库与大文件存放](docs/Git仓库与大文件存放.md) | 仓库地址、上传范围与源码更新方式 |
| [GoogleDrive大资源备份与恢复](docs/GoogleDrive大资源备份与恢复.md) | 权重、仿真资产、200场景的分片上传与恢复 |
| [迁移与验收记录](docs/迁移与验收记录.md) | 本次整理结果与验证证据 |
| [旧工程清理与新路径验收](docs/旧工程清理与新路径验收.md) | 删除旧目录后，直接运行新工程的验收结果 |

当前配置入口：`configs/experiments/active_experiment.json`。模块注册入口：`configs/modules.json`。

```bash
python3 -m arx_project inspect
environments/simulation/bin/python -m arx_project doctor --verify-dataset
```
