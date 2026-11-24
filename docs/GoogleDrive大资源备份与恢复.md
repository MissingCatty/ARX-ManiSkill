# Google Drive大资源备份与恢复

项目代码和网站保存在GitHub，大资源备份到用户指定的Google Drive文件夹。恢复时，克隆对应代码版本，再把资源包解压到工程根目录，目录结构保持一致。

| 项目 | 位置或内容 |
| --- | --- |
| Google账号 | yzhao6141@gmail.com |
| 目标文件夹 | [ARX资源备份](https://drive.google.com/drive/folders/1-dUTcHeIoismcRMk0JAj5mCRuR7R691A) |
| 云端快照目录 | 目标文件夹下`ARX-ManiSkill/<snapshot_id>/` |
| 模型权重 | `weights/<method>/`，包括当前保留的模型 |
| 三维资产 | `simulation/assets/`，包括共享索引、当前模型及其余源资产库 |
| 场景与配置 | 最新200场景、机器人配置、相机参数、工程配置与恢复说明 |
| 模块内置资产与历史记录 | 包含迁入共享库的ManiSkill和两版cuRobo资产、旧ARX模型，以及登记的可达性实验归档 |
| 压缩与分片 | 按方法和资产组生成tar.zst，每片最多2 GiB |
| 恢复校验 | 每片提供SHA256和MD5，上传后执行远端校验 |
| 本机备份工作目录 | `tmp/cloud_backup/<snapshot_id>/` |
| 登录凭据 | 工程外的`~/.config/arx-maniskill/google-drive.conf`，权限600 |

Google账号的实际可用空间在授权后检查。普通个人账号免费空间为15 GB，Drive、Gmail和相册共享；完整资源包需要按最终压缩体积准备容量。[Google存储空间说明](https://support.google.com/drive/answer/6374270?hl=zh-Hans)

## 首次授权

安装上传工具：

```bash
python3 -m tools.install_backup_tools
python3 -m tools.drive_auth
```

授权程序启动本机回调页面，地址记录在`tmp/cloud_backup/auth/status.json`。从远程Windows电脑操作时，先建立到服务器53682端口的SSH转发，再在浏览器打开程序给出的授权地址，选择上述Google账号并授权。登录凭据自动保存到工程外，程序随后核对账号、文件夹写入权限和存储配额。

```cmd
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:53682:127.0.0.1:53682 zhaoyechong@10.201.35.208
```

上传工具采用[rclone的Google Drive授权流程](https://rclone.org/drive/)。如果Google提示共享OAuth客户端不可用，在Google Cloud创建自己的桌面OAuth客户端，将下载的客户端JSON保存在`~/.config/arx-maniskill/google-oauth-client.json`，权限设为600，再运行授权程序。

## 创建与上传快照

```bash
python3 -u -m tools.drive_backup run \
  --snapshot tmp/cloud_backup/resources_20261124_03 \
  --wait-for-auth
```

程序先打包资源，再等待Google授权；授权完成且空间足够后，自动上传到指定文件夹下的快照目录。已完成的分片按校验值跳过，中断后再次运行即可继续。每个快照绑定代码提交号和场景冻结散列。源资源在打包期间发生变化时，停止该次打包并记录原因。

| 本机文件 | 用途 |
| --- | --- |
| `plan.json` | 快照分组与代码版本 |
| `status.json` | 打包、等待授权、空间不足、上传中或完成 |
| `last_error.json` | 最近一次故障原因 |
| `payload/manifest.json` | 完整分组、分片大小和校验值 |
| `payload/SHA256SUMS` | 分片及恢复工具的SHA256清单 |
| `upload.log` | 上传进度与校验日志 |
| `remote_check.txt` | 本地与远端逐文件核对结果 |

快照完成以`status.json`中的`stage=complete`和`remote_checksum_check_passed=true`为准。

## 资源位置登记

`configs/resource_snapshot.json`登记当前云端快照、资源清单、完成标记及本地目录；`configs/modules.json`登记各模块的权重及共享资产入口，`simulation/assets/catalog.json`登记共享模型。当前目标快照为`ARX-ManiSkill/resources_20261124_03/`，已排除EASG和SPAR3D。云端出现`COMPLETE.json`后，表示全部上传及远端校验完成。

03新增模块资产和实验归档，已有权重分片沿用02正在进行的上传，再在Drive内复制到03。两个快照的资源清单分别绑定各自代码；新目录结构按03恢复。当前200场景的所有参考数据仍在Git中，Drive另外备份一份。

## 在新机器恢复

1. 从GitHub克隆工程，按资源包`manifest.json`中的`source_commit`切换代码版本。
2. 下载Drive中的完整快照目录，保留`packages/`及其下分组结构。
3. 安装系统`tar`、`zstd`和Python，执行快照附带的恢复脚本：

```bash
python3 restore_resources.py \
  --backup-dir /path/to/downloaded_snapshot \
  --destination /path/to/ARX-ManiSkill
```

恢复程序先核对全部分片，再按原目录解压权重、模型和场景。Python环境按`configs/environments/`中的依赖清单创建，随后运行：

```bash
cd /path/to/ARX-ManiSkill
python3 tools/bind_environments.py
environments/simulation/bin/python -m arx_project doctor --verify-dataset
environments/simulation/bin/python -m tools.validate_library_assets
```

只检查下载完整性时使用`--verify-only`，不指定恢复目录。
