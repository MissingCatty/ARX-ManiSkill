# 独立 ManiSkill 安装

安装目录：`/home/zhaoyechong/projects/ManiSkill/`。

Python 虚拟环境、官方源码、运行资产、缓存和检查输出均保存在此目录。

## 使用

```bash
cd /home/zhaoyechong/projects/ManiSkill
source env.sh
python -c 'import mani_skill; print(mani_skill.__version__)'
```

也可以通过包装脚本使用独立环境：

```bash
bash run.sh -c 'import mani_skill; print(mani_skill.__version__)'
```

运行 GPU 程序前，查询当前空闲卡并绑定 UUID：

```bash
nvidia-smi --query-gpu=index,uuid,memory.free --format=csv
export MANISKILL_GPU_UUID=GPU-填写所选设备的完整UUID
source env.sh
```

`env.sh` 不会自动占用或预留显卡。

## 目录

| 路径 | 内容 |
|---|---|
| `.venv/` | Python 3.12 独立环境 |
| `source/` | 官方 `v3.0.1` 源码，安装后以 editable 方式使用 |
| `assets/data/` | 额外任务资产下载目录 |
| `assets/demos/` | 按需下载的示教 |
| `cache/` | pip、uv、CUDA、Hugging Face 等缓存 |
| `logs/` | 安装和检查日志 |
| `outputs/` | 安装检查回执、图片和视频 |
| `requirements.in` | 核心版本约束 |
| `requirements.lock.txt` | 安装后的完整依赖版本 |

## 安装检查

```bash
bash run.sh scripts/check_install.py
```

检查 CUDA 张量计算、官方 `PickCube-v1` 环境初始化、5 步零动作物理步进、RGB/深度/实例分割输出及无窗口渲染。
该检查仅验证独立安装，不构成抓取成功率测试，也不是现有 ARX 链路的验证结果。

额外任务资产按需下载。例如 YCB 物体抓取任务：

```bash
python -m mani_skill.utils.download_asset PickSingleYCB-v1
```

ARX 机器人适配和当前 COOLAS-ARX ZeroGrasp + cuRobo 批量执行入口见 [arx_pipeline/README.md](arx_pipeline/README.md)。该入口使用新的 ManiSkill 桌面场景，并单独记录物理执行结果。

SAPIEN 3.0.3 原生将 PhysX 库和 ImGui 配置放在 `~/.sapien`。
本独立环境对其两个 Python 缓存路径增加了 `SAPIEN_CACHE_DIR` 支持，指向本目录的 `cache/sapien/`。
原始文件保存在 `logs/sapien_original/`，修改哈希记录在 `outputs/sapien_cache_patch.json`。
重新安装 SAPIEN 会覆盖此缓存路径调整。

官方安装说明：<https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html>

## 全部运行资产与高保真渲染

2026-11-15 已按用户要求启动全部 194 项运行资产下载，范围不含官方示教。
后台服务：`maniskill-assets-all-20261115-03.service`。
清单见 `outputs/all_assets_manifest.json`，实时状态见 `outputs/asset_download_state.json`，
逐项完成回执见 `outputs/asset_receipts/`，日志见 `logs/asset_download.log`。
下载使用独立暂存目录、断点续传和官方已有的归档校验值；仅完成后才移入正式资产目录。
UCSD 官方服务器不可达的 175 项资产单独标为 blocked，不能视作下载完成。
上游记录：<https://github.com/mani-skill/ManiSkill/issues/1489>。

高保真环境入口：

```python
import high_quality
env = high_quality.make("PickCube-v1")
```

使用光线追踪 rt、256 samples/pixel、16 次路径深度、OptiX 降噪；
传感器图像 1920×1080，录像相机 3840×2160。
物理参数使用官方任务默认值。官方单环境自动选择 CPU 物理，GPU 渲染；
此版本 SAPIEN 的 GPU 并行渲染路径不支持 rt，不能把该入口切换为多环境 GPU 并行。
`high_quality.make` 为新环境显式应用配置，直接 `gym.make` 仍使用官方默认画质。

高保真检查已通过，回执为 `outputs/high_quality_check.json`，
4K 示例为 `outputs/high_quality_4k.png`。这是一帧安装展示，未执行抓取试验。
