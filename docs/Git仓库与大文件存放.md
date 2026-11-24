# Git 仓库与大文件存放

仓库：[MissingCatty/ARX-ManiSkill](https://github.com/MissingCatty/ARX-ManiSkill)，主分支为 `main`。

## 上传内容

| 内容 | 存放方式 |
| --- | --- |
| 公共接口、各算法模块、实验编排、维护工具 | 随 Git 上传源码 |
| 工程文档、环境依赖清单、机器人与相机配置 | 随 Git 上传 |
| 当前冻结的独立采样 200 场景 | `simulation/scenes/latest200/`中的定义及参考观测随 Git 上传；场景引用共享资产ID |
| 历史场景 | 当前运行和恢复只使用latest200场景集合 |
| 共享资产索引 | `simulation/assets/catalog.json`随 Git 上传，保存相对路径与物理定义 |
| 网站 | 页面源码、发布用媒体副本、附件、压缩缓存随 Git 上传 |
| 三维模型、权重、Python 环境、运行产物 | 分别保存在 `simulation/assets/models/`、`weights/`、`environments/`、`tmp/`，由根目录忽略规则排除 |
| 账号密钥、机器授权、待办数据库 | 保存在各用户自己的配置或网站状态目录，不进入 Git |

新机器运行抓取还需要另外交付资产库和权重，再按环境清单安装依赖。Git 推送本身不会上传这些大文件，也不会删除本机文件。

大资源备份目标为用户指定的Google Drive文件夹，账号`yzhao6141@gmail.com`。按方法和资产组压缩分片，随包附带校验清单与恢复脚本；上传状态及操作说明见[GoogleDrive大资源备份与恢复](GoogleDrive大资源备份与恢复.md)。

## 克隆与启动网站

```bash
git clone git@github.com:MissingCatty/ARX-ManiSkill.git
cd ARX-ManiSkill/website
npm ci
npm run build
npm start
```

已有压缩缓存随仓库保存，构建网站不需要仿真环境或模型权重。待办数据库在新机器首次启动时创建。

## 更新源码

第三方源码以普通文件保存，保留原许可证；上游地址和原始提交记录在 `configs/upstream_sources.json`。本地针对 ARX 的接口和修改也随仓库上传。

ManiSkill和两版cuRobo的内置网格、纹理、环境贴图已经从源码目录移到共享资产库，通过`simulation/assets/catalog.json`加载。可达性历史实验位于`tmp/reachability/experiments/`，源码目录只保留工具、配置和说明。当前200场景及其参考观测继续完整保存在Git中。

模型仓库原有的 LFS 过滤规则不继承到本仓库，权重按工程目录单独存放。

GraspGenX 上游尚未下载的示例网格和示例深度，以 `configs/external_asset_sources.json` 记录来源、对象哈希和大小，不把占位文本当成资产上传。它们属于上游演示数据，当前 ARX 使用独立的机器人资产和抓取描述。

当前机器还保留各上游源码自己的 Git 元数据。提交整个工程前，使用以下工具暂存，避免把模块误登记为空的子仓库入口：

```bash
python3 tools/stage_repository.py
git diff --cached --stat
git commit -m "Describe the project update"
git push origin main
```

该工具仅在暂存期间移开上游 `.git` 元数据，结束后恢复，不删除上游提交历史。从本仓库克隆出的普通源码目录直接使用 `git add` 也可以。
