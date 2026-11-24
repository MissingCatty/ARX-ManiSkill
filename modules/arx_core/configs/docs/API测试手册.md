# CoolAS ARX API 测试手册

本文档给出 `/home/arx/coolas-arx` 当前所有 API 的完整测试流程。测试目标是确认三个服务可启动、接口可用、参数生效、实机控制行为符合预期，并覆盖 lift/chassis 的统一 CAN5 并发控制场景。

相关文档：

- API 说明：`docs/API说明手册.md`
- 系统设计：`docs/系统详细设计.md`

## 1. 测试范围

当前需要覆盖三个服务：

| 服务 | 端口 | 启动脚本 | 主要 API |
|---|---:|---|---|
| arx-ops | 8001 | `scripts/run_arx_client.sh` | `/arm/*`, `/gripper/*`, `/camera/*`, `/lift/*`, `/chassis/*` |
| langsam | 8002 | `scripts/run_langsam_server.sh` | `/segment` |
| zerograsp | 8003 | `scripts/run_zerograsp_server.sh` | `/grasp/predict`, `/grasp/plan`, `/task/run-grasp` |

## 2. 安全前置条件

实机测试前必须确认：

1. 机械臂周围无遮挡，急停可用。
2. 夹爪和底盘周围没有人员身体部位。
3. CAN 口没有被其他进程占用。
4. 如果测试底盘升降或移动，确认平台周围安全。
5. `/task/run-grasp` 的 `execute=true` 只在规划结果和相机画面确认无误后使用。
6. 第一次测试建议使用小速度、短距离、短时长。

建议开 4 个终端：

- 终端 A：arx-ops 服务
- 终端 B：init_client 实机控制器
- 终端 C：langsam 服务
- 终端 D：zerograsp 服务和 curl 测试

## 3. 启动服务

### 3.1 启动 arx-ops

```bash
cd /home/arx/workspace/xufangjin/COOLAS-ARX
scripts/run_arx_client.sh
```

默认监听：

```text
http://127.0.0.1:8001
```

### 3.2 初始化 ARX 实机控制器

```bash
cd /home/arx/workspace/xufangjin/COOLAS-ARX
scripts/init_client.sh
```

正常情况下会：

- 初始化 CAN；
- 等待 `can3` UP；
- source ARX X5 ROS2 workspace；
- 启动 `v2_single_arm.launch.py`；
- 等待 `/arm_status` 发布；
- 当 `tools/unified_body_controller` 不存在时自动编译；
- 清理遗留 CAN5 控制进程并启动唯一的统一 C++ Body Controller；
- 等待 Unix Socket 返回有效 `feedback_height` 后报告 ready。

如果 CAN 已经初始化，不希望重启 CAN：

```bash
scripts/init_client.sh --skip-can
```

### 3.3 启动 langsam

```bash
cd /home/arx/workspace/xufangjin/COOLAS-ARX
scripts/run_langsam_server.sh
```

默认监听：

```text
http://127.0.0.1:8002
```

### 3.4 启动 zerograsp

```bash
cd /home/arx/workspace/xufangjin/COOLAS-ARX
scripts/run_zerograsp_server.sh
```

默认监听：

```text
http://127.0.0.1:8003
```

## 4. 基础健康检查

### 4.1 arx-ops

```bash
curl http://127.0.0.1:8001/health
```

期望：

```json
{"service":"arx-ops","status":"ok","stage":"phase2"}
```

### 4.2 langsam

```bash
curl http://127.0.0.1:8002/health
```

期望：

```json
{"service":"langsam","status":"ok","stage":"phase2"}
```

### 4.3 zerograsp

```bash
curl http://127.0.0.1:8003/health
```

期望：

```json
{"service":"zerograsp","status":"ok","stage":"phase2"}
```

## 5. arx-ops API 测试

### 5.1 读取机械臂状态：`GET /arm/status`

```bash
curl http://127.0.0.1:8001/arm/status
```

左右臂状态测试：

```bash
# 右臂：默认写法
curl http://127.0.0.1:8001/arm/status

# 右臂：显式写法
curl 'http://127.0.0.1:8001/arm/status?arm_id=right'

# 左臂
curl 'http://127.0.0.1:8001/arm/status?arm_id=left'
```

期望：返回 `end_pos`、`joint_pos`、`gripper`。不加 `arm_id` 或 `arm_id=right` 时为右臂，`arm_id=left` 时为左臂。

示例：

```json
{
  "end_pos": [0.0, 0.0, 0.003, 0.0, 0.0, 0.0],
  "joint_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05],
  "gripper": -0.05,
  "source": "/arm_status"
}
```

异常判断：

- 如果返回 503，说明 `/arm_status` 没有发布。
- 检查 `scripts/init_client.sh` 是否仍在运行。
- 检查 ROS domain、RMW、ARX controller 是否正常。

### 5.2 相机采集：`POST /camera/capture-rgbd`

```bash
curl -X POST http://127.0.0.1:8001/camera/capture-rgbd \
  -H 'Content-Type: application/json' \
  -d '{
    "output_dir": "/tmp/coolas_api_test_camera"
  }'
```

期望：

- 返回 `color_path`、`depth_path`、`params_path`；
- 对应文件存在。

检查文件：

```bash
ls -la /tmp/coolas_api_test_camera
```

可选指定相机序列号：

```bash
curl -X POST http://127.0.0.1:8001/camera/capture-rgbd \
  -H 'Content-Type: application/json' \
  -d '{
    "serial": "352122272245",
    "output_dir": "/tmp/coolas_api_test_camera_serial"
  }'
```

多相机采集测试：

```bash
# 右手相机
curl -X POST http://127.0.0.1:8001/camera/capture-rgbd \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id": "right_hand",
    "output_dir": "/tmp/coolas_api_test_camera_right_hand"
  }'

# 左手相机
curl -X POST http://127.0.0.1:8001/camera/capture-rgbd \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id": "left_hand",
    "output_dir": "/tmp/coolas_api_test_camera_left_hand"
  }'

# 头部相机
curl -X POST http://127.0.0.1:8001/camera/capture-rgbd \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id": "head",
    "output_dir": "/tmp/coolas_api_test_camera_head"
  }'
```

期望：三次调用都返回 `color_path`、`depth_path`、`params_path`，并在对应目录生成 `color.png`、`depth.png`、`params.json`。当前配置中 `right_hand=352122272245`、`left_hand=352122273322`、`head=352122270890`。

### 5.3 夹爪打开：`POST /gripper/open`

```bash
curl -X POST "http://127.0.0.1:8001/gripper/open?duration_sec=2.0"
```

说明：上面的 query 写法保留原有测试内容；当前测试左右夹爪时，以后面的 JSON body 写法为准。

期望：

- 夹爪打开；
- API 在约 2 秒加少量开销后返回；
- 返回 `ok=true`。

加长打开时间：

```bash
curl -X POST "http://127.0.0.1:8001/gripper/open?duration_sec=5.0"
```

左右夹爪 JSON body 测试：

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

期望：指定 `left` 时只有左夹爪打开；不加 `arm_id` 或指定 `right` 时为右夹爪。

### 5.4 夹爪闭合：`POST /gripper/close`

```bash
curl -X POST "http://127.0.0.1:8001/gripper/close?duration_sec=3.0"
```

说明：上面的 query 写法保留原有测试内容；当前测试左右夹爪时，以后面的 JSON body 写法为准。

期望：

- 夹爪闭合；
- 如果闭合不到位，可加长 `duration_sec`。

加长闭合测试：

```bash
curl -X POST "http://127.0.0.1:8001/gripper/close?duration_sec=5.0"
```

左右夹爪 JSON body 测试：

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

期望：指定 `left` 时只有左夹爪闭合；不加 `arm_id` 或指定 `right` 时为右夹爪。

### 5.5 机械臂小范围移动：`POST /arm/move-end-pose`

先读取当前位置：

```bash
curl http://127.0.0.1:8001/arm/status
```

小范围移动示例：

```bash
curl -X POST http://127.0.0.1:8001/arm/move-end-pose \
  -H 'Content-Type: application/json' \
  -d '{
    "target_pose": [0.10, 0.00, 0.02, 0.0, 0.0, 0.0],
    "target_gripper": 0.0,
    "speed": 0.3,
    "close_delay": 0.5,
    "exit_on_pose": true,
    "timeout_sec": 90
  }'
```

期望：

- 机械臂移动到接近目标位姿；
- `exit_on_pose=true` 时到达目标后返回；
- 返回 `ok=true`。

定时退出测试：

```bash
curl -X POST http://127.0.0.1:8001/arm/move-end-pose \
  -H 'Content-Type: application/json' \
  -d '{
    "target_pose": [0.10, 0.00, 0.02, 0.0, 0.0, 0.0],
    "target_gripper": 0.0,
    "speed": 0.3,
    "close_delay": 0.5,
    "exit_after_sec": 5.0,
    "timeout_sec": 20
  }'
```

期望：执行约 5 秒后退出。

左臂小范围移动测试：

```bash
# 先读左臂状态
curl 'http://127.0.0.1:8001/arm/status?arm_id=left'

# 左臂小幅移动
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

期望：只有左臂移动，右臂不动；达到目标阈值后返回 `ok=true`。左右臂移动接口形式相同，区别只在 `arm_id`：不加 `arm_id` 或 `"arm_id": "right"` 时控制右臂，`"arm_id": "left"` 时控制左臂。

### 5.6 机械臂返回 Home：`POST /arm/home`

当前 `/arm/home` 流程：

1. 默认先按固定时间闭合夹爪；
2. 然后机械臂返回 home；
3. 使用放宽后的 home 退出阈值。

推荐测试：

```bash
curl -X POST "http://127.0.0.1:8001/arm/home?speed=2.0&timeout_sec=120&close_gripper=true&gripper_close_duration_sec=3.0"
```

期望：

- 先执行夹爪闭合；
- 闭合阶段结束后机械臂返回原位；
- 达到阈值后 API 返回；
- 不应出现先张开夹爪再返回。

如果只是回 home，不主动闭合夹爪：

```bash
curl -X POST "http://127.0.0.1:8001/arm/home?speed=2.0&timeout_sec=120&close_gripper=false"
```

如果视觉上已到位但 API 仍等待，可测试更宽松阈值：

```bash
curl -X POST "http://127.0.0.1:8001/arm/home?speed=2.0&timeout_sec=120&linear_threshold=0.008&angular_threshold=0.04"
```

### 5.7 lift 状态：`GET /lift/status`

```bash
curl http://127.0.0.1:8001/lift/status
```

期望：

- `ok=true`、`mode=unified_cpp`、`ready=true`、`running=true`；
- `feedback_height` 和 `height_command` 为数值；
- 初次启动时 `target_height` 等于初始反馈高度；
- `body_mode.body_daemon_processes` 中只有一个 `tools/unified_body_controller`。

### 5.8 设置平台高度：`POST /lift/height`

`height` 是 SDK 原生控制量，不是 cm。首次实机测试应从当前 `feedback_height` 附近小步调整，并确保急停可用。

```bash
curl -X POST http://127.0.0.1:8001/lift/height \
  -H 'Content-Type: application/json' \
  -d '{"height":2.0,"hold":true,"ramp_step":0.001}'
```

期望：`target_height=2.0`；调整期间 `height_command` 和 `feedback_height` 变化；到达后 `height_reached=true`，最终承重命令持续发布。

重复设置新目标不会重启控制器或先归零：

```bash
curl -X POST http://127.0.0.1:8001/lift/height \
  -H 'Content-Type: application/json' \
  -d '{"height":1.5,"hold":true,"ramp_step":0.001}'
```

查看控制日志：

```bash
tail -n 30 /tmp/coolas_body_controller.log
```

### 5.9 lift home：`POST /lift/home`

```bash
curl -X POST http://127.0.0.1:8001/lift/home
```

期望：目标更新为 SDK 原生量 `0.5`。接口固定使用 `hold=false`、`ramp_step=0.001`，不接收查询参数。

### 5.10 停止高度调整：`POST /lift/stop`

```bash
curl -X POST http://127.0.0.1:8001/lift/stop
```

期望：当前 `height_command` 被冻结为新目标并继续发布；统一控制器仍持有 CAN5，底盘 API 可继续使用。

### 5.11 chassis 状态：`GET /chassis/status`

```bash
curl http://127.0.0.1:8001/chassis/status
```

期望：返回与 `/lift/status` 相同的统一状态，包含 `moving`、`linear_x`、`linear_y`、`angular_z`、`feedback_height`、`target_height` 和 `height_command`。

### 5.12 底盘移动：`POST /chassis/move`

```bash
curl -X POST http://127.0.0.1:8001/chassis/move \
  -H 'Content-Type: application/json' \
  -d '{"linear_x":0.05,"linear_y":0.0,"angular_z":0.0,"duration":1.0}'
```

期望：请求约 1 秒后返回；底盘短暂前进后 `moving=false`，平台高度目标不变。旋转测试可使用 `linear_x=0.0`、`angular_z=0.2`；左右平移测试可使用 `linear_y=0.05`。

### 5.13 停止底盘：`POST /chassis/stop`

```bash
curl -X POST http://127.0.0.1:8001/chassis/stop \
  -H 'Content-Type: application/json' \
  -d '{"lock":true}'
```

期望：速度立即归零，平台目标和高度 hold 不受影响。

## 6. CAN5 并发控制测试

终端 A 启动较长移动：

```bash
curl -X POST http://127.0.0.1:8001/chassis/move \
  -H 'Content-Type: application/json' \
  -d '{"linear_x":0.05,"angular_z":0.0,"duration":5.0}'
```

终端 B 在移动过程中设置一个已通过实机验证的原生高度目标：

```bash
curl -X POST http://127.0.0.1:8001/lift/height \
  -H 'Content-Type: application/json' \
  -d '{"height":2.0,"hold":true,"ramp_step":0.001}'
```

终端 C 连续观察状态：

```bash
watch -n 0.2 'curl -s http://127.0.0.1:8001/chassis/status'
```

期望：底盘移动不中断，平台同时向目标调整；移动结束后 `moving=false`，平台继续 hold；两个 status 接口均返回 `mode=unified_cpp`。

停止移动不应停止高度控制：

```bash
curl -X POST http://127.0.0.1:8001/chassis/stop
curl http://127.0.0.1:8001/lift/status
```

## 7. langsam API 测试

### 7.1 准备测试图像

可以使用 arx-ops 采集一张图：

```bash
curl -X POST http://127.0.0.1:8001/camera/capture-rgbd \
  -H 'Content-Type: application/json' \
  -d '{"output_dir": "/tmp/coolas_seg_test"}'
```

确认文件：

```bash
ls -la /tmp/coolas_seg_test
```

### 7.2 LangSAM 分割：`POST /segment`

```bash
curl -X POST http://127.0.0.1:8002/segment \
  -H 'Content-Type: application/json' \
  -d '{
    "image_path": "/tmp/coolas_seg_test/color.png",
    "prompt": "green cube.",
    "output_dir": "/tmp/coolas_seg_test",
    "method": "langsam",
    "mask_name": "green_cube_mask.png",
    "overlay_name": "green_cube_overlay.png",
    "info_name": "green_cube_mask.json",
    "min_area": 50,
    "keep_largest": true
  }'
```

期望：

- `found=true`；
- 返回 `mask_path`、`overlay_path`、`info_path`；
- overlay 图中目标被正确标出。

检查文件：

```bash
ls -la /tmp/coolas_seg_test/*green_cube*
```

### 7.3 HSV 分割：`POST /segment`

```bash
curl -X POST http://127.0.0.1:8002/segment \
  -H 'Content-Type: application/json' \
  -d '{
    "image_path": "/tmp/coolas_seg_test/color.png",
    "prompt": "red cube.",
    "output_dir": "/tmp/coolas_seg_test",
    "method": "hsv",
    "mask_name": "hsv_mask.png",
    "overlay_name": "hsv_overlay.png",
    "info_name": "hsv_mask.json",
    "min_area": 50,
    "keep_largest": true
  }'
```

期望：

- 如果图中有对应颜色目标，`found=true`；
- 如果没有，`found=false` 是合理结果。

## 8. zerograsp API 测试

### 8.1 完整 plan-only 测试：`POST /task/run-grasp`

推荐先测试这个接口，因为它会自动完成采集、分割、输入准备、推理和规划。

```bash
curl -X POST http://127.0.0.1:8003/task/run-grasp \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "green cube.",
    "mask_method": "langsam",
    "execute": false,
    "run_tag": "api_plan_test"
  }'
```

期望：

- `ok=true`；
- `mode=plan_only`；
- `segmentation.found=true`；
- `prediction.grasp_path` 存在；
- `plan.safety_passed=true`；
- `execution=null`；
- `run_dir` 下生成完整文件。

检查输出目录：

```bash
# 将下面路径替换为响应中的 run_dir
find /home/arx/coolas-arx/runs/zerograsp_task_xxx -maxdepth 3 -type f | sort
```

重点检查：

```text
camera/color.png
camera/*overlay.png
zg_input/rgb.png
zg_input/depth.png
zg_input/mask.png
zg_input/camera.json
zg_output/recommended_grasp_top1.json
run_report.json
```

### 8.2 ZeroGrasp 推理：`POST /grasp/predict`

先使用上一步生成的 `zg_input` 目录。

```bash
curl -X POST http://127.0.0.1:8003/grasp/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "input_dir": "/home/arx/coolas-arx/runs/zerograsp_task_xxx/zg_input",
    "output_dir": "/tmp/coolas_zg_predict_test",
    "timeout_sec": 600,
    "grasp_criterion": "score"
  }'
```

期望：

- `grasp_path` 指向 `recommended_grasp_top1.json`；
- `raw.returncode=0`；
- `raw.run_report.status=ok`。

### 8.3 抓取规划：`POST /grasp/plan`

先读取当前机械臂状态：

```bash
curl http://127.0.0.1:8001/arm/status
```

使用当前 `end_pos` 和推理结果规划：

```bash
curl -X POST http://127.0.0.1:8003/grasp/plan \
  -H 'Content-Type: application/json' \
  -d '{
    "prediction_path": "/tmp/coolas_zg_predict_test/recommended_grasp_top1.json",
    "current_end_pos": [0.0, 0.0, 0.003, 0.0, 0.0, 0.0],
    "allow_debug_zero_pose": false,
    "execute": false
  }'
```

期望：

- 返回 `approach_pose`、`target_pose`、`target_gripper`；
- `safety_passed=true`；
- `safety_failures=[]`。

如果只是离线验证，没有实机状态，可临时使用：

```json
"allow_debug_zero_pose": true
```

但实机执行前不建议使用 debug zero pose。

### 8.4 完整真实执行：`POST /task/run-grasp execute=true`

执行前必须先完成 plan-only，并检查：

- overlay 是否正确覆盖目标；
- `plan.safety_passed=true`；
- `target_pose` 在合理范围；
- 机械臂周围安全。

执行抓取 motion：

```bash
curl -X POST http://127.0.0.1:8003/task/run-grasp \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "green cube.",
    "mask_method": "langsam",
    "execute": true,
    "execute_duration_sec": 20.0,
    "run_tag": "real_execute_test"
  }'
```

期望：

- 前半段与 plan-only 一致；
- `mode=execute`；
- `execution` 不为 null；
- 机械臂执行抓取 motion；
- `execute_duration_sec` 到时后 motion 子进程自动退出。

注意：`execute_duration_sec` 只控制最终 motion 子进程运行时长，不包含采集、分割、ZeroGrasp 推理耗时，也不会自动 home。

抓取后如需夹住物体返回：

```bash
curl -X POST "http://127.0.0.1:8001/arm/home?speed=2.0&timeout_sec=120&close_gripper=true&gripper_close_duration_sec=3.0"
```

## 9. 全流程推荐测试顺序

### 9.1 非实机安全链路

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8003/health
```

```bash
curl -X POST http://127.0.0.1:8003/task/run-grasp \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "green cube.",
    "mask_method": "langsam",
    "execute": false,
    "run_tag": "plan_only_check"
  }'
```

检查输出图像和报告。

### 9.2 机械臂基础链路

```bash
curl http://127.0.0.1:8001/arm/status
curl -X POST "http://127.0.0.1:8001/gripper/open?duration_sec=2.0"
curl -X POST "http://127.0.0.1:8001/gripper/close?duration_sec=3.0"
curl -X POST "http://127.0.0.1:8001/arm/home?speed=2.0&timeout_sec=120&close_gripper=true&gripper_close_duration_sec=3.0"
```

### 9.3 底盘升降链路

```bash
curl -X POST http://127.0.0.1:8001/lift/height \
  -H 'Content-Type: application/json' \
  -d '{"height": 2.0, "hold": true, "ramp_step": 0.001}'

curl http://127.0.0.1:8001/lift/status

curl -X POST http://127.0.0.1:8001/lift/home

curl -X POST http://127.0.0.1:8001/lift/stop
```

### 9.4 底盘移动链路

```bash
curl -X POST http://127.0.0.1:8001/chassis/move \
  -H 'Content-Type: application/json' \
  -d '{"linear_x": 0.05, "angular_z": 0.0, "duration": 1.0, "lock_after": true}'

curl http://127.0.0.1:8001/chassis/status

curl -X POST http://127.0.0.1:8001/chassis/stop \
  -H 'Content-Type: application/json' \
  -d '{"lock": true}'
```

### 9.5 真实抓取链路

先 plan-only：

```bash
curl -X POST http://127.0.0.1:8003/task/run-grasp \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "green cube.", "mask_method": "langsam", "execute": false, "run_tag": "before_real"}'
```

确认 overlay 和 safety 后再 execute：

```bash
curl -X POST http://127.0.0.1:8003/task/run-grasp \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "green cube.", "mask_method": "langsam", "execute": true, "execute_duration_sec": 20.0, "run_tag": "real_execute"}'
```

抓取后 home：

```bash
curl -X POST "http://127.0.0.1:8001/arm/home?speed=2.0&timeout_sec=120&close_gripper=true&gripper_close_duration_sec=3.0"
```

## 10. 常见问题检查

### 10.1 `/arm/status` 超时

检查：

```bash
ros2 topic list | grep arm
```

如果没有 `/arm_status`：

- 确认 `scripts/init_client.sh` 正在运行；
- 确认 CAN 已初始化；
- 确认 ARX arm controller 没有掉电或异常退出。

### 10.2 `/arm/home` 视觉到位但 API 不返回

检查当前位姿：

```bash
curl http://127.0.0.1:8001/arm/status
```

如果残差略大于默认阈值，可放宽：

```bash
curl -X POST "http://127.0.0.1:8001/arm/home?speed=2.0&timeout_sec=120&linear_threshold=0.008&angular_threshold=0.04"
```

### 10.3 `/task/run-grasp execute=true` 很久不返回

确认底层命令是否带 `--exit-after`：

```bash
ps -eo pid,etimes,cmd | grep slow_move_end_pose_v2
```

如果没有 `--exit-after`：

- 确认请求里有 `execute_duration_sec`；
- 确认 8001 和 8003 服务都已重启；
- 确认当前运行的是 `/home/arx/coolas-arx` 的新代码。

### 10.4 lift 和 chassis 冲突

这是设计行为。两者都占用 `can5`。

- 使用 lift 前：`POST /chassis/stop`
- 使用 chassis 前：`POST /lift/stop`

### 10.5 夹爪闭合不够紧

加长闭合时间：

```bash
curl -X POST "http://127.0.0.1:8001/gripper/close?duration_sec=5.0"
```

home 前加长闭合阶段：

```bash
curl -X POST "http://127.0.0.1:8001/arm/home?speed=2.0&timeout_sec=120&close_gripper=true&gripper_close_duration_sec=5.0"
```

## 11. 测试完成后的清理

停止 lift：

```bash
curl -X POST http://127.0.0.1:8001/lift/stop
```

停止 chassis：

```bash
curl -X POST http://127.0.0.1:8001/chassis/stop \
  -H 'Content-Type: application/json' \
  -d '{"lock": true}'
```

机械臂回 home：

```bash
curl -X POST "http://127.0.0.1:8001/arm/home?speed=2.0&timeout_sec=120&close_gripper=false"
```

停止各服务终端：

```text
Ctrl+C
```

如需清理历史任务输出，可检查：

```bash
ls -lh /home/arx/coolas-arx/runs
```

谨慎删除历史 run 目录，确认其中没有需要保留的相机图、mask、报告或推理结果。
