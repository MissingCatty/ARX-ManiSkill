# CoolAS ARX API 说明文档

本文档说明 `/home/arx/coolas-arx` 当前暴露的全部 HTTP API、请求参数、调用示例、返回结构，以及对应的 Python 函数接口。

当前项目包含三个 HTTP 服务：

| 服务 | 默认端口 | 启动脚本 | API 入口 | 主要职责 |
|---|---:|---|---|---|
| arx-ops | 8001 | `scripts/run_arx_client.sh` | `arx-ops/coolas_arx_client/api.py` | 真实机器人操作：相机、机械臂、夹爪、底盘升降、底盘移动 |
| langsam | 8002 | `scripts/run_langsam_server.sh` | `servers/langsam/coolas_langsam/api.py` | 图像分割 |
| zerograsp | 8003 | `scripts/run_zerograsp_server.sh` | `servers/zerograsp/coolas_zerograsp/api.py` | 抓取推理、抓取规划、完整抓取任务编排 |

## 函数式调用准备

如果不通过 HTTP，而是在 Python 中直接调用内部函数，需要设置 `PYTHONPATH`：

```bash
export PYTHONPATH=/home/arx/coolas-arx/shared:/home/arx/coolas-arx/arx-ops:/home/arx/coolas-arx/servers/langsam:/home/arx/coolas-arx/servers/zerograsp:${PYTHONPATH:-}
```

所有请求/响应模型位于：

```text
/home/arx/coolas-arx/shared/arx_shared/schemas.py
```

通用执行响应 `ExecutionResult`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `ok` | bool | 操作是否成功 |
| `message` | string | 文本信息，可能包含底层命令输出 |
| `command` | list[string] | 实际执行的底层命令 |
| `returncode` | int/null | 子进程退出码，后台进程或未执行时可能为 null |
| `data` | object | 附加结构化数据 |

---

# 1. arx-ops API

Base URL：

```text
http://127.0.0.1:8001
```

代码目录：

```text
/home/arx/coolas-arx/arx-ops/coolas_arx_client/
```

## 1.1 健康检查

### `GET /health`

检查 arx-ops 服务是否可用。

```bash
curl http://127.0.0.1:8001/health
```

返回示例：

```json
{"service":"arx-ops","status":"ok","stage":"phase2"}
```

函数接口：

```python
from coolas_arx_client.api import health

result = health()
```

---

## 1.2 初始化抓取仿真场景

### `POST /task/reset-grasp-scene`

仿真专用接口。当前 Isaac Sim 进程必须以 `--task grasp` 启动；接口会停止当前运动、清空待执行命令、执行 `World.reset()`，并重建 RGB-D 相机，使场景恢复到抓取任务的初始状态。

```bash
curl -X POST http://127.0.0.1:8001/task/reset-grasp-scene
```

返回示例：

```json
{
  "ok": true,
  "message": "simulation reset from asset default state",
  "data": {
    "generation": 1,
    "command_id": "..."
  }
}
```

如果当前仿真不是 `grasp` 任务，会返回 `ok=false`，需要用 `--task grasp` 重启仿真。

---

## 1.3 读取机械臂状态

### `GET /arm/status`

从 ROS topic `/arm_status` 读取机械臂状态。

```bash
curl http://127.0.0.1:8001/arm/status
```

左右臂差异：不传 `arm_id` 或传 `arm_id=right` 时读取右臂；传 `arm_id=left` 时读取左臂。当前默认右臂状态来自 `/arm_status`，左臂状态来自 `/arm_l_status`。

```bash
# 右臂，默认写法
curl http://127.0.0.1:8001/arm/status

# 右臂，显式写法
curl 'http://127.0.0.1:8001/arm/status?arm_id=right'

# 左臂
curl 'http://127.0.0.1:8001/arm/status?arm_id=left'
```

返回字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `end_pos` | list[float] | 末端位姿 `[x, y, z, roll, pitch, yaw]` |
| `joint_pos` | list[float] | 关节位置，通常包含夹爪关节 |
| `gripper` | float/null | 当前夹爪反馈值 |
| `source` | string | 状态来源 topic，通常是 `/arm_status` |

函数接口：

```python
from coolas_arx_client.robot import get_arm_status

status = get_arm_status(timeout_sec=3.0)
```

注意：如果 `/arm_status` 没有发布，接口会返回 503。通常需要先执行：

```bash
scripts/init_client.sh
```

---

## 1.4 采集 RGB-D 图像

### `POST /camera/capture-rgbd`

调用 RealSense 采集彩色图、深度图和相机参数。

```bash
curl -X POST http://127.0.0.1:8001/camera/capture-rgbd \
  -H 'Content-Type: application/json' \
  -d '{
    "serial": "352122272245",
    "output_dir": "/tmp/coolas_capture"
  }'
```

请求参数 `CaptureRequest`：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `serial` | string/null | 配置文件中的相机序列号 | RealSense 序列号 |
| `camera_id` | string/null | 默认相机 | 配置相机 ID：`right_hand`、`left_hand`、`head` |
| `output_dir` | string/null | 自动生成目录 | 输出目录 |

返回字段 `CaptureResult`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `color_path` | string | 彩色图路径 |
| `depth_path` | string | 深度图路径 |
| `params_path` | string | 相机参数 JSON 路径 |
| `intrinsics` | object/null | 相机内参 |
| `depth_scale` | float/null | 深度比例 |
| `serial_number` | string/null | 实际使用的相机序列号 |

函数接口：

```python
from coolas_arx_client.camera import capture_rgbd

result = capture_rgbd(serial="352122272245", output_dir="/tmp/coolas_capture")
```

多相机调用示例：

```bash
# 右手相机，默认相机；也可显式指定 camera_id=right_hand
curl -X POST http://127.0.0.1:8001/camera/capture-rgbd \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id": "right_hand",
    "output_dir": "/tmp/coolas_capture_right_hand"
  }'

# 左手相机
curl -X POST http://127.0.0.1:8001/camera/capture-rgbd \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id": "left_hand",
    "output_dir": "/tmp/coolas_capture_left_hand"
  }'

# 头部相机
curl -X POST http://127.0.0.1:8001/camera/capture-rgbd \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id": "head",
    "output_dir": "/tmp/coolas_capture_head"
  }'
```

当前配置中的相机序列号：`right_hand=352122272245`、`left_hand=352122273322`、`head=352122270890`。

---

## 1.4 机械臂末端位姿控制

### `POST /arm/move-end-pose`

调用 `legacy/scripts/slow_move_end_pose_v2.py`，以慢速控制机械臂末端移动。

```bash
curl -X POST http://127.0.0.1:8001/arm/move-end-pose \
  -H 'Content-Type: application/json' \
  -d '{
    "target_pose": [0.10, 0.00, 0.02, 0.0, 0.0, 0.0],
    "target_gripper": 0.0,
    "speed": 0.5,
    "close_delay": 0.5,
    "exit_on_pose": true,
    "exit_after_sec": 20.0,
    "timeout_sec": 60
  }'
```

请求参数 `EndPoseRequest`：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `target_pose` | list[float] | 必填 | `[x, y, z, roll, pitch, yaw]` |
| `arm_id` | string | `right` | 机械臂 ID：不传或传 `right` 为右臂；传 `left` 为左臂 |
| `target_gripper` | float/null | null | 目标夹爪值；null 时按 0.0 处理 |
| `speed` | float | 1.0 | 速度倍率 |
| `close_delay` | float | 0.5 | 到达位姿后触发 legacy 闭合逻辑的延迟 |
| `hold_gripper` | bool | false | 是否保持当前夹爪反馈值 |
| `exit_on_pose` | bool | false | 到达目标位姿后是否立即退出 |
| `exit_after_sec` | float/null | null | 执行指定秒数后自动退出，不依赖反馈 |
| `timeout_sec` | float/null | null | 外层子进程超时时间 |

安全范围来自 `configs/arx.yaml`：

```yaml
x_range: [0.00, 0.50]
y_range: [-0.40, 0.40]
z_range: [-0.30, 0.60]
```

函数接口：

```python
from arx_shared.schemas import EndPoseRequest
from coolas_arx_client.motion import move_end_pose

result = move_end_pose(EndPoseRequest(
    target_pose=[0.10, 0.00, 0.02, 0.0, 0.0, 0.0],
    target_gripper=0.0,
    speed=0.5,
    exit_on_pose=True,
    timeout_sec=60,
))
```

左臂移动示例：

```bash
curl -X POST http://127.0.0.1:8001/arm/move-end-pose \
  -H 'Content-Type: application/json' \
  -d '{
    "arm_id": "left",
    "target_pose": [0.03, 0.03, 0.03, 0.0, 0.0, 0.0],
    "target_gripper": 0.0,
    "speed": 0.5,
    "exit_on_pose": true,
    "timeout_sec": 8.0
  }'
```

左右臂接口形式相同，差别只在 `arm_id`：不加 `arm_id` 或传 `"arm_id": "right"` 时控制右臂；传 `"arm_id": "left"` 时控制左臂。右臂底层调用 `slow_move_end_pose_v2.py`，左臂底层调用 `slow_move_end_pose_v2_left.py`。

---

## 1.5 机械臂返回 Home

### `POST /arm/home`

让机械臂返回原点/初始姿态。当前流程为：

1. 如果 `close_gripper=true`，先在当前位置定时发布夹爪闭合命令；
2. 然后执行 q-style home 返回；
3. Home 退出阈值默认放宽为 `6mm / 0.03rad`。

推荐调用：

```bash
curl -X POST "http://127.0.0.1:8001/arm/home?speed=2.0&timeout_sec=120&close_gripper=true&gripper_close_duration_sec=3.0"
```

查询参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `speed` | float | 0.5 | 返回速度倍率 |
| `close_delay` | float | 0.5 | 传给底层 slow move 的闭合延迟参数 |
| `timeout_sec` | float | 120.0 | 整个 home 流程超时时间 |
| `auto_quit_after` | float | 5.0 | 兼容保留；当前 home 直接 `--home-on-start`，不再等待该参数后才返回 |
| `close_gripper` | bool | true | 是否先闭合夹爪再返回 |
| `linear_threshold` | float | 0.006 | Home 平移退出阈值，单位 m |
| `angular_threshold` | float | 0.03 | Home 角度退出阈值，单位 rad |
| `gripper_close_duration_sec` | float/null | 配置 `gripper_close_duration_sec`，当前为 3.0 | 闭合夹爪阶段持续时间；不通过夹爪反馈判断 |

不主动闭合夹爪返回：

```bash
curl -X POST "http://127.0.0.1:8001/arm/home?speed=2.0&timeout_sec=120&close_gripper=false"
```

函数接口：

```python
from coolas_arx_client.motion import home_arm

result = home_arm(
    speed=2.0,
    timeout_sec=120,
    close_gripper=True,
    gripper_close_duration_sec=3.0,
)
```

---

## 1.6 夹爪打开/闭合

### `POST /gripper/open`

按固定时长发布夹爪打开命令，结束后不额外 hold。

```bash
curl -X POST "http://127.0.0.1:8001/gripper/open?duration_sec=2.0"
```

查询参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `duration_sec` | float/null | 配置 `gripper_open_duration_sec`，当前 2.0 | 打开命令持续发布时间 |

说明：上面的 query 写法保留原有文档示例；当前需要区分左右臂时，应使用 JSON body。`arm_id` 不传或传 `right` 时控制右夹爪，传 `left` 时控制左夹爪。

当前推荐使用 JSON body，便于指定左右臂：

```bash
# 右夹爪：不传 arm_id 或显式传 right
curl -X POST http://127.0.0.1:8001/gripper/open \
  -H 'Content-Type: application/json' \
  -d '{
    "arm_id": "right",
    "duration_sec": 2.0
  }'

# 左夹爪
curl -X POST http://127.0.0.1:8001/gripper/open \
  -H 'Content-Type: application/json' \
  -d '{
    "arm_id": "left",
    "duration_sec": 2.0
  }'
```

### `POST /gripper/close`

按固定时长发布夹爪闭合命令。

```bash
curl -X POST "http://127.0.0.1:8001/gripper/close?duration_sec=3.0"
```

查询参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `duration_sec` | float/null | 配置 `gripper_close_duration_sec`，当前 3.0 | 闭合命令持续发布时间 |

说明：上面的 query 写法保留原有文档示例；当前需要区分左右臂时，应使用 JSON body。`arm_id` 不传或传 `right` 时控制右夹爪，传 `left` 时控制左夹爪。

当前推荐使用 JSON body，便于指定左右臂：

```bash
# 右夹爪：不传 arm_id 或显式传 right
curl -X POST http://127.0.0.1:8001/gripper/close \
  -H 'Content-Type: application/json' \
  -d '{
    "arm_id": "right",
    "duration_sec": 3.0
  }'

# 左夹爪
curl -X POST http://127.0.0.1:8001/gripper/close \
  -H 'Content-Type: application/json' \
  -d '{
    "arm_id": "left",
    "duration_sec": 3.0
  }'
```

`/gripper/open` 和 `/gripper/close` 的左右臂差别同样只由 `arm_id` 决定：不加 `arm_id` 或传 `"arm_id": "right"` 时控制右夹爪；传 `"arm_id": "left"` 时控制左夹爪。

配置中的夹爪目标值：

```yaml
open_gripper_value: -3.2
close_gripper_value: 0.0
```

函数接口：

```python
from coolas_arx_client.motion import gripper_command

open_result = gripper_command(open_gripper=True, duration_sec=2.0)
close_result = gripper_command(open_gripper=False, duration_sec=3.0)
```

---

## 1.7 底盘移动

### `POST /chassis/move`

通过统一 C++ Body Controller 执行一次定时底盘运动。HTTP 请求会等待运动结束后返回；控制器在移动期间仍以 400 Hz 下发平台高度命令，因此平台可同时升降或保持高度。

```bash
curl -X POST http://127.0.0.1:8001/chassis/move \
  -H 'Content-Type: application/json' \
  -d '{
    "linear_x": 0.1,
    "linear_y": 0.0,
    "angular_z": 0.0,
    "duration": 1.0
  }'
```

请求参数 `ChassisMoveRequest`：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `linear_x` | float | 0.0 | 前后速度命令；配置范围当前 `[-0.5, 0.5]` |
| `linear_y` | float | 0.0 | 左右平移速度命令；配置范围当前 `[-0.5, 0.5]` |
| `angular_z` | float | 0.0 | 旋转速度命令；配置范围当前 `[-0.8, 0.8]` |
| `duration` | float | 1.0 | 运动持续时间；配置范围当前 `[0.0, 10.0]` |
| `height` | float/null | null | 可选；同时更新平台的原生 SDK 高度目标。省略时保留已有高度目标 |
| `read_current_height` | bool | true | 兼容字段，当前统一 C++ 控制路径不使用 |
| `lock_after` | bool | true | 兼容字段；运动到期后控制器固定将速度清零并发送停止模式 |
| `timeout_sec` | float/null | null | HTTP 等待超时；省略时为 `duration + status_timeout_sec + 5` |

参数限制来自 `configs/arx.yaml`：`linear_x` 和 `linear_y` 为 `[-0.5, 0.5]`，`angular_z` 为 `[-0.8, 0.8]`，`duration` 必须大于 0 且不超过 10 秒。

函数接口：

```python
from arx_shared.schemas import ChassisMoveRequest
from coolas_arx_client.chassis import move_chassis

result = move_chassis(ChassisMoveRequest(
    linear_x=0.1,
    linear_y=0.0,
    angular_z=0.0,
    duration=1.0,
))
```

### `POST /chassis/stop`

立即将底盘速度置零；统一 C++ Body Controller 和当前高度目标继续运行，不释放 `can5`。

```bash
curl -X POST http://127.0.0.1:8001/chassis/stop \
  -H 'Content-Type: application/json' \
  -d '{
    "lock": true
  }'
```

请求参数 `ChassisStopRequest`：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `height` | float/null | null | 可选；停止移动时同时更新平台的原生 SDK 高度目标 |
| `read_current_height` | bool | true | 兼容字段，当前统一 C++ 控制路径不使用 |
| `lock` | bool | true | 兼容字段；当前实现始终清零速度并发送停止模式 |
| `timeout_sec` | float/null | null | 兼容字段，当前实现不使用 |

函数接口：

```python
from arx_shared.schemas import ChassisStopRequest
from coolas_arx_client.chassis import stop_chassis

result = stop_chassis(ChassisStopRequest(lock=True))
```

### `GET /chassis/status`

读取统一控制器状态。返回内容与 `/lift/status` 使用同一份状态，包含移动状态和高度状态。

```bash
curl http://127.0.0.1:8001/chassis/status
```

函数接口：

```python
from coolas_arx_client.chassis import get_chassis_status

result = get_chassis_status()
```

---

## 1.8 底盘升降

### `POST /lift/height`

设置平台高度目标。`height` 是厂商 SDK `LiftHeadControlLoop::setHeight()` 使用的原生控制量，不是厘米，也不能按固定比例换算为物理高度。统一 C++ Body Controller 直接调用 SDK，并在同一个 400 Hz 循环中同时执行 `setHeight()`、`setChassisCmd()` 和 `loop()`。

```bash
curl -X POST http://127.0.0.1:8001/lift/height \
  -H 'Content-Type: application/json' \
  -d '{
    "height": 2.0,
    "hold": true,
    "ramp_step": 0.001
  }'
```

请求参数 `LiftHeightRequest`：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `height` | float | 必填 | SDK 原生高度目标；不是 cm。当前 API 未设置上限，实机安全范围必须通过标定确定 |
| `robot_type` | int | 1 | 兼容字段；统一控制器使用启动配置中的机器人类型 |
| `warmup` | float | 4.0 | 兼容字段；统一控制器常驻运行 |
| `feedback_target` | bool | true | 兼容字段，当前控制器始终使用 SDK 反馈闭环 |
| `ramp_step` | float | 0.001 | 每个 400 Hz 控制周期允许 `height_command` 变化的最大步长，必须大于 0 |
| `max_command` | float | 0.0 | 兼容字段；实际上限由控制器启动参数 `--max-height-command` 决定，0 表示不额外限幅 |
| `until_target` | bool | true | 兼容字段，当前控制器持续运行 |
| `hold` | bool | true | 保存到状态中；控制器会持续发布最终 `height_command` 以保持平台 |
| `stop_existing` | bool | true | 兼容字段；调用只更新目标，不重启控制器 |

控制逻辑：

```text
目标 height
  → 反馈外环（outer_gain）
  → ramp_step 限速
  → height_command
  → SDK setHeight() + loop()
```

`feedback_height`、`target_height` 和 `height_command` 都是 SDK 原生量。达到 `feedback_tolerance` 后，控制器冻结最后的承重 `height_command` 并持续下发。重复调用接口只会原地更新目标，不会停止控制器或中断底盘移动。

函数接口：

```python
from arx_shared.schemas import LiftHeightRequest
from coolas_arx_client.lift import set_lift_height

result = set_lift_height(LiftHeightRequest(height=2.0, hold=True))
```

### `POST /lift/home`

当前 home 的底层行为固定等价于调用 `/lift/height`，参数为 `height=0.5`、`hold=false`、`ramp_step=0.001`。

```bash
curl -X POST http://127.0.0.1:8001/lift/home
```

函数接口：

```python
from coolas_arx_client.lift import home_lift_height

result = home_lift_height()
```

### `GET /lift/status`

读取统一 C++ Body Controller 中的高度目标、反馈、输出命令、移动状态和运行参数。

```bash
curl http://127.0.0.1:8001/lift/status
```

函数接口：

```python
from coolas_arx_client.lift import lift_status

result = lift_status()
```

### `POST /lift/stop`

停止继续追踪原目标：控制器将当前 `height_command` 设为新目标，并持续发布该命令。它不是关闭升降电机或释放 CAN5。

```bash
curl -X POST http://127.0.0.1:8001/lift/stop
```

该接口不会关闭统一控制器，底盘控制仍可继续使用。

函数接口：

```python
from coolas_arx_client.lift import stop_lift_height

result = stop_lift_height()
```

---

# 2. langsam API

Base URL：

```text
http://127.0.0.1:8002
```

代码目录：

```text
/home/arx/coolas-arx/servers/langsam/coolas_langsam/
```

## 2.1 健康检查

### `GET /health`

```bash
curl http://127.0.0.1:8002/health
```

函数接口：

```python
from coolas_langsam.api import health

result = health()
```

---

## 2.2 图像分割

### `POST /segment`

对输入图像执行目标分割，支持 `langsam` 和 `hsv` 方法。

```bash
curl -X POST http://127.0.0.1:8002/segment \
  -H 'Content-Type: application/json' \
  -d '{
    "image_path": "/home/arx/coolas-arx/runs/example/camera/color.png",
    "prompt": "green cube.",
    "output_dir": "/home/arx/coolas-arx/runs/example/camera",
    "method": "langsam",
    "mask_name": "green_cube_mask.png",
    "overlay_name": "green_cube_overlay.png",
    "info_name": "green_cube_mask.json",
    "min_area": 50,
    "keep_largest": true
  }'
```

请求参数 `SegmentRequest`：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `image_path` | string | 必填 | 输入 RGB 图像路径 |
| `prompt` | string | `red cube.` | 文本提示，LangSAM 使用 |
| `output_dir` | string/null | 自动生成 | 输出目录 |
| `method` | `langsam`/`hsv` | `langsam` | 分割方法 |
| `mask_name` | string/null | 自动命名 | mask 文件名 |
| `overlay_name` | string/null | 自动命名 | overlay 文件名 |
| `info_name` | string/null | 自动命名 | JSON 信息文件名 |
| `min_area` | int | 50 | 最小 mask 面积 |
| `keep_largest` | bool | true | 是否只保留已选候选 mask 内的最大连通域；不会合并多个候选 |

LangSAM 的多候选筛选由 `configs/langsam.yaml` 的 `selection` 配置控制。服务会独立评估
每个候选，拒绝检测分数过低、检测框接近全图或 mask 面积比例过大的异常候选，默认从
有效候选中选择 GroundingDINO 检测分数最高者。详细候选诊断写入 `info_path` 对应的
JSON，其中 `selected_index` 为选中候选，`candidates[].rejection_reasons` 为拒绝原因。

返回字段 `SegmentationResult`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `mask_path` | string/null | mask 图路径 |
| `overlay_path` | string/null | 叠加可视化图路径 |
| `info_path` | string/null | 分割信息 JSON 路径 |
| `bbox` | list[float]/null | `[x1, y1, x2, y2]` |
| `centroid` | list[float]/null | mask 中心点 `[x, y]` |
| `score` | float/null | 检测分数 |
| `label` | string/null | 标签 |
| `found` | bool | 是否找到目标 |
| `area` | int/null | mask 面积 |
| `raw` | object | 底层输出 |

函数接口：

```python
from arx_shared.schemas import SegmentRequest
from coolas_langsam.segment import segment_image

result = segment_image(SegmentRequest(
    image_path="/path/to/color.png",
    prompt="green cube.",
    method="langsam",
))
```

---

# 3. zerograsp API

Base URL：

```text
http://127.0.0.1:8003
```

代码目录：

```text
/home/arx/coolas-arx/servers/zerograsp/coolas_zerograsp/
```

## 3.1 健康检查

### `GET /health`

```bash
curl http://127.0.0.1:8003/health
```

函数接口：

```python
from coolas_zerograsp.api import health

result = health()
```

---

## 3.2 ZeroGrasp 推理

### `POST /grasp/predict`

对准备好的 ZeroGrasp 输入运行抓取预测。

```bash
curl -X POST http://127.0.0.1:8003/grasp/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "input_dir": "/home/arx/coolas-arx/runs/example/zg_input",
    "output_dir": "/home/arx/coolas-arx/runs/example/zg_output",
    "timeout_sec": 600,
    "grasp_criterion": "score"
  }'
```

也可以显式传入四个输入文件：

```json
{
  "img_path": "/path/rgb.png",
  "depth_path": "/path/depth.png",
  "mask_path": "/path/mask.png",
  "camera_info_path": "/path/camera.json"
}
```

请求参数 `GraspPredictRequest`：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `input_dir` | string/null | null | 包含 `rgb.png/depth.png/mask.png/camera.json` 的目录 |
| `img_path` | string/null | null | RGB 图路径 |
| `depth_path` | string/null | null | 深度图路径 |
| `mask_path` | string/null | null | mask 路径 |
| `camera_info_path` | string/null | null | 相机参数路径 |
| `output_dir` | string/null | 自动生成 | 输出目录 |
| `timeout_sec` | int | 600 | 推理超时时间 |
| `grasp_criterion` | `score`/`min_rpy` | `score` | 抓取选择策略；当前主要沿用 score |

返回字段 `GraspPrediction`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `grasp_path` | string/null | `recommended_grasp_top1.json` 路径 |
| `output_dir` | string/null | 输出目录 |
| `raw` | object | 底层 ZeroGrasp 输出和日志 |

函数接口：

```python
from arx_shared.schemas import GraspPredictRequest
from coolas_zerograsp.grasp import run_predict_only

result = run_predict_only(GraspPredictRequest(
    input_dir="/path/to/zg_input",
    output_dir="/path/to/zg_output",
))
```

---

## 3.3 抓取规划

### `POST /grasp/plan`

把 ZeroGrasp 的相机坐标抓取结果转换为 ARX 末端目标位姿，并做安全检查。

```bash
curl -X POST http://127.0.0.1:8003/grasp/plan \
  -H 'Content-Type: application/json' \
  -d '{
    "arm_id": "right",
    "prediction_path": "/home/arx/coolas-arx/runs/example/zg_output/recommended_grasp_top1.json",
    "current_end_pos": [0.0, 0.0, 0.003, 0.0, 0.0, 0.0],
    "allow_debug_zero_pose": false,
    "execute": false
  }'
```

请求参数 `GraspPlanRequest`：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `arm_id` | `right`/`left` | `right` | 规划使用的机械臂和标定；不传为右臂，`left` 使用 `left_ee_camera` 和 `left_tcp_gripper` |
| `prediction_path` | string/null | null | ZeroGrasp 推荐抓取 JSON 路径 |
| `prediction` | object/null | null | 直接传入推荐抓取对象 |
| `current_end_pos` | list[float]/null | null | 当前 ARX 末端位姿 |
| `allow_debug_zero_pose` | bool | false | 是否允许没有真实状态时使用零位姿调试 |
| `execute` | bool | false | 是否按执行模式规划；通常由 `/task/run-grasp` 使用 |

返回字段 `GraspPlan`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `approach_pose` | list[float]/null | 预抓取位姿 |
| `target_pose` | list[float]/null | 最终抓取位姿 |
| `target_gripper` | float/null | 目标夹爪值 |
| `advance_depth_m` | float | 沿抓取深度推进距离 |
| `safety_passed` | bool | 是否通过安全检查 |
| `safety_failures` | list[string] | 安全检查失败原因 |
| `metadata` | object | 坐标变换、ZeroGrasp 分数、安全距离等调试信息 |

函数接口：

```python
from arx_shared.schemas import GraspPlanRequest
from coolas_zerograsp.planning import plan_grasp

plan = plan_grasp(GraspPlanRequest(
    arm_id="right",
    prediction_path="/path/to/recommended_grasp_top1.json",
    current_end_pos=[0, 0, 0.003, 0, 0, 0],
))
```

---

## 3.4 完整抓取任务

### `POST /task/run-grasp`

完整流程入口：

```text
采集 RGB-D -> 读取机械臂状态 -> 分割 -> 准备 ZeroGrasp 输入 -> 推理 -> 规划 -> 可选执行
```

只规划不执行：

```bash
curl -X POST http://127.0.0.1:8003/task/run-grasp \
  -H 'Content-Type: application/json' \
  -d '{
    "arm_id": "right",
    "prompt": "green cube.",
    "mask_method": "langsam",
    "execute": false,
    "run_tag": "plan_test"
  }'
```

左臂只规划不执行：

```bash
curl -X POST http://127.0.0.1:8003/task/run-grasp \
  -H 'Content-Type: application/json' \
  -d '{
    "arm_id": "left",
    "prompt": "green cube.",
    "mask_method": "langsam",
    "execute": false,
    "run_tag": "left_plan_test"
  }'
```

执行真实抓取：

```bash
curl -X POST http://127.0.0.1:8003/task/run-grasp \
  -H 'Content-Type: application/json' \
  -d '{
    "arm_id": "right",
    "prompt": "green cube.",
    "mask_method": "langsam",
    "execute": true,
    "execute_duration_sec": 20.0,
    "run_tag": "real_execute_test"
  }'
```

请求参数 `RunGraspRequest`：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `arm_id` | `right`/`left` | `right` | 抓取使用的机械臂；`left` 时采集 `left_hand` 相机、读取左臂状态、使用左臂标定并执行左臂 motion |
| `prompt` | string | `red cube.` | 分割提示词 |
| `execute` | bool | false | 是否执行真实机械臂动作 |
| `mask_method` | `langsam`/`hsv` | `langsam` | 分割方法 |
| `mask_color` | string | `red` | HSV 方法使用的颜色名 |
| `mask_target` | string/null | null | 目标名称覆盖值 |
| `run_tag` | string | 空字符串 | 输出目录标签 |
| `output_dir` | string/null | 自动生成 | 本次任务输出根目录 |
| `skip_projection` | bool | false | 是否跳过投影调试图 |
| `allow_debug_zero_pose` | bool | false | 是否允许零位姿调试 |
| `execute_duration_sec` | float/null | null | `execute=true` 时最终 motion 子进程运行多久后自动退出 |
| `timeout_sec` | int | 600 | ZeroGrasp 推理等长任务超时时间 |

返回字段 `RunGraspResult`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `ok` | bool | 整体流程是否成功 |
| `mode` | string | `plan_only` 或 `execute` |
| `run_dir` | string | 本次任务目录 |
| `current_pose_source` | string | 当前位姿来源 |
| `capture` | object/null | 相机采集结果 |
| `segmentation` | object/null | 分割结果 |
| `zg_input` | object/null | ZeroGrasp 输入准备结果 |
| `prediction` | object/null | ZeroGrasp 推理结果 |
| `plan` | object/null | 抓取规划结果 |
| `execution` | object/null | 执行结果；`execute=false` 时为 null |
| `report_path` | string/null | 完整报告 JSON 路径 |
| `message` | string | 附加消息 |

函数接口：

```python
from arx_shared.schemas import RunGraspRequest
from coolas_zerograsp.orchestrator import run_grasp_task

result = run_grasp_task(RunGraspRequest(
    arm_id="right",
    prompt="green cube.",
    mask_method="langsam",
    execute=False,
    run_tag="api_test",
))
```

执行真实抓取的函数式调用：

```python
result = run_grasp_task(RunGraspRequest(
    arm_id="right",
    prompt="green cube.",
    mask_method="langsam",
    execute=True,
    execute_duration_sec=20.0,
    run_tag="real_execute_test",
))
```

注意：`execute_duration_sec` 只控制最终机械臂 motion 子进程运行时间，不包含采集、分割、ZeroGrasp 推理耗时，也不会自动执行 `/arm/home`。`arm_id=left` 当前依赖 `configs/calibration.yaml` 中的 `left_ee_camera` 和 `left_tcp_gripper`，其中左臂标定如仍是镜像初值，只建议先做 plan-only 或低风险验证。


## 3.9 Brain operator: `grasp_object`

`grasp_object` 是大脑侧抓取 operator，不在 `arx-ops` 的 `8001` HTTP 路由中直接暴露。大脑调用时通过 operator registry 使用。

核心参数：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `target` | string | 必填 | 目标物体英文描述，例如 `red cube` |
| `arm_id` | `right`/`left` | `right` | 抓取使用的机械臂；`right` 使用 `right_hand/right_ee_camera/right_tcp_gripper`，`left` 使用 `left_hand/left_ee_camera/left_tcp_gripper` |
| `points` | array/null | null | 可选图像点，用于多个同类物体时指定目标实例 |
| `reuse_observation` | object/null | null | 编排内部使用；复用上一轮 observe 的相机目录、mask 和分割结果 |

当前默认执行行为由 `configs/brain.yaml` 决定：

```yaml
grasp_object:
  execute: true
  return_home: false
  post_grasp_action: "lift_ee"
  post_grasp_lift_z_m: 0.10
```

因此当前默认不是返回 home，而是：

```text
ZeroGrasp 抓取调用返回后，即使返回 ok:false 或 API 超时
-> 先闭合同一只手的夹爪
-> 读取当前 arm end_pos
-> target_pose = 当前 end_pos 且 z += 0.10
-> 调用 /arm/move-end-pose 抬高同一只手 10cm
```

结果中可关注：

| 字段 | 说明 |
|---|---|
| `evidence.arm_id` | 实际抓取手臂 |
| `evidence.camera_id` | 实际使用的手部相机 |
| `evidence.zerograsp_response` | ZeroGrasp 原始响应 |
| `evidence.post_grasp_gripper_close_response` | 抬高前闭合对应夹爪的响应 |
| `evidence.post_grasp_arm_status` | 闭爪后、抬高前读取到的当前末端位姿 |
| `evidence.post_grasp_lift_request` | z+0.10 后发给 `/arm/move-end-pose` 的请求 |
| `evidence.post_grasp_lift_response` | 抬高动作响应 |

临时实机测试脚本：

```bash
./scripts/test_grasp_object_right_red_cube_execute.py
./scripts/test_grasp_object_left_red_cube_execute.py
```

## 3.10 Brain operator: `observe_local`

`observe_local` 用于局部目标观察。它会编排机械臂观察位姿、相机采集、分割/深度测量，并在 operator 内部完成坐标转换，输出目标相对左右 EE home 的位置。

核心参数：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `target` | string | 必填 | 目标物体英文描述，例如 `red cube` |
| `intent` | string | `inspect` | 观察意图，可用 `inspect`/`locate`/`place`/`grasp` |
| `observer_arm_id` | `right`/`left`/null | null | 指定用哪只手部相机观察；`right` 使用右腕相机，`left` 使用左腕相机 |
| `held_arm_id` | `right`/`left`/null | null | 如果某只手已抓住物体，可传该手，operator 会优先使用另一只手观察 |
| `camera_id` | `right_hand`/`left_hand`/`head`/null | null | 显式指定相机；优先级高于 `observer_arm_id` |
| `view_mode` | `static`/`preset_sweep` | `preset_sweep` | 默认 `preset_sweep`，会移动观察手到固定内看绝对位姿后观察一次；显式传 `static` 时不移动观察手 |
| `visual_hint` | string/null | null | 传给视觉模块的实例提示 |
| `restore_observer` | bool | false | 是否观察后恢复观察手到观察前位姿 |

当前默认内看绝对位姿：

```text
right: [0.05,  0.10, 0.05, 0.0, 0.0,  0.50]
left:  [0.05, -0.10, 0.05, 0.0, 0.0, -0.50]
```

结果中可关注：

| 字段 | 说明 |
|---|---|
| `target_camera_xyz_m` | 目标相对当前相机的 OpenCV 光学坐标，x 图像右、y 图像下、z 相机前 |
| `target_left_ee_xyz_m` | 目标相对左 EE home 的坐标，x 前、y 左、z 上 |
| `target_right_ee_xyz_m` | 目标相对右 EE home 的坐标，x 前、y 左、z 上 |
| `transform_source` | 本次坐标转换链路 |

临时实机测试脚本：

```bash
./scripts/test_observe_local_left_red_cube.py
./scripts/test_observe_local_right_red_cube.py
```


## 3.11 Brain operator: `place_object`

`place_object` 用于把已经被夹爪抓住的物体放到最近观测到的容器/目标位置。它只做放置，不做抓取、不做观察、不做分割；目标位置必须来自新鲜的 `observe_local` 结果。

核心参数：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `held_arm_id` | `right`/`left` | 必填 | 当前抓住物体的机械臂 |
| `destination` | string/null | null | 放置目标名称，仅用于记录，例如 `box` |
| `destination_observation` | object | 必填 | 最近一次 `observe_local` 的 evidence 或包含 `target_right_ee_xyz_m` / `target_left_ee_xyz_m` 的坐标对象 |
| `place_height_offset_m` | number | `0.10` | 预放置 TCP 高度，目标点 z + 该偏移 |
| `release_height_offset_m` | number | `0.02` | 释放 TCP 高度，目标点 z + 该偏移 |
| `retreat_z_m` | number | `0.10` | 松爪后 TCP 向上撤离距离 |
| `speed` | number | `1.0` | 三段 arm move 的默认速度 |
| `timeout_sec` | number | `8.0` | 单段 arm move 超时 |
| `open_gripper_duration_sec` | number | `2.0` | 到达释放附近后打开夹爪持续时间 |
| `execute` | bool | `true` | `false` 时只计算和校验位姿，不执行物理动作 |

放置位姿计算规则：

`place_object` 把 `destination_observation.target_{held_arm_id}_ee_xyz_m` 作为目标 TCP 参考点，而不是 EE 原点。operator 会保持当前 `arm_status.end_pos[3:6]` 姿态不变，用当前 RPY 计算 TCP 在 EE home 坐标系下的偏移，然后反推 EE 目标：

```text
R_current = rotation_matrix(current_rpy)
tcp_offset = R_current * [0.14, 0, 0]
target_ee_xyz = target_tcp_xyz - tcp_offset
```

三段 TCP 目标分别是：

| 阶段 | TCP 目标 |
|---|---|
| `pre_place_pose` | `destination_xyz + [0, 0, place_height_offset_m]` |
| `release_pose` | `destination_xyz + [0, 0, release_height_offset_m]` |
| `retreat_pose` | `release_tcp_xyz + [0, 0, retreat_z_m]` |

超时策略：

- `pre_place_pose` 超时后，如果当前 EE 已接近目标允许继续，否则失败；
- `release_pose` 超时后，如果当前 EE 已在释放容差内（默认 0.08m），仍会打开夹爪；如果偏差太大则失败；
- `retreat_pose` 在松爪之后执行，超时或失败只记录 warning，不把整体放置判失败。

结果中可关注：

| 字段 | 说明 |
|---|---|
| `destination_xyz_m` | 选中的目标位置，坐标系为 `held_arm_id` 对应 EE home |
| `pre_place_pose` | 预放置 EE/end_pos 目标 |
| `release_pose` | 释放 EE/end_pos 目标 |
| `retreat_pose` | 撤离 EE/end_pos 目标 |
| `move_to_pre_place_response` | 预放置移动响应 |
| `move_to_release_pose_response` | 释放位姿移动响应 |
| `gripper_open_response` | 打开夹爪响应 |
| `move_to_retreat_pose_response` | 撤离移动响应 |

示例：

```json
{
  "operator": "place_object",
  "arguments": {
    "held_arm_id": "right",
    "destination": "box",
    "destination_observation": {
      "target_right_ee_xyz_m": [0.36, 0.05, -0.08]
    },
    "execute": true
  },
  "reason": "右臂已抓住红色方块，使用最近 observe_local 得到的纸盒相对右 EE home 坐标执行放置"
}
```

---

# 4. 统一 CAN5 控制和实机注意事项

## 4.1 `can5` 单进程所有权

底盘移动和平台升降由项目内的 `tools/unified_body_controller` 统一控制。该 C++ 进程是 `can5` 的唯一持有者，直接使用厂商 SDK，在同一个 400 Hz 循环中合并高度和速度控制。

- `/lift/height` 更新高度目标，可在底盘移动期间调用；
- `/chassis/move` 更新速度和运动截止时间，不清除高度目标；
- `/chassis/stop` 只将速度置零，平台继续 hold；
- `/lift/stop` 冻结当前 `height_command` 并持续发布，不关闭统一控制器；
- `direct_lift_height_dynamic` 仅保留为离线诊断工具，不能与统一控制器同时运行。

## 4.2 推荐启动顺序

```bash
cd /home/arx/workspace/xufangjin/COOLAS-ARX
scripts/init_client.sh
scripts/run_arx_client.sh
scripts/run_langsam_server.sh
scripts/run_zerograsp_server.sh
```

`init_client.sh` 会在二进制缺失时自动运行 `tools/build_unified_body_controller.sh`，随后启动统一 C++ Body Controller，并等待 Unix Socket 和有效高度反馈。脚本必须保持运行；lift/chassis API 只做健康检查，不再懒启动进程。

## 4.3 抓取后返回 Home

`/task/run-grasp` 只负责执行抓取 motion，不会自动 home。抓取后如需夹住物体返回，调用：

```bash
curl -X POST "http://127.0.0.1:8001/arm/home?speed=2.0&timeout_sec=120&close_gripper=true&gripper_close_duration_sec=3.0"
```

`/arm/home` 当前会先按固定时长闭合夹爪，再返回 home，避免边返回边闭合导致物体掉落。
