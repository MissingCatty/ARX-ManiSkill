# arx-sim

`arx-sim` is a standalone Isaac Sim adapter for the HTTP contract exposed by
`main`'s `arx-ops`. It starts Isaac Sim and an API-compatible `sim_client` on
port 8001. Existing upper-layer services continue to use their unchanged URLs,
request models, response models, camera serials, units, frames, and command
semantics.

The module reads the existing `configs/arx.yaml` and
`configs/calibration.yaml` at startup. Isaac-only scene, joint, sensor, ROS 2,
asset-lock, and runtime settings live under `arx-sim/config/`.

The current ARX deployment uses `nvcr.io/nvidia/isaac-sim:6.0.1`. The image's
`/isaac-sim/VERSION` is `6.0.1-rc.7+release.42383.32955d8d.gl`; its bundled
Python is 3.12.13 on Ubuntu 24.04.3. Startup checks the `6.0.1` version prefix
in `config/simulator.lock.yaml`. Use Isaac Sim's `python.sh` to launch the
adapter and install `arx-sim/requirements.txt` into that interpreter.

To reproduce this setup on another host, copy the asset directory alongside
the checkout or set `ARX_SIM_ASSET_ROOT`. Container deployments mount the
checkout at `/workspace/COOLAS-ARX`, assets at `/workspace/asset`, and set
`ISAAC_SIM_ROOT=/isaac-sim`. ZeroGrasp and cuRobo run in separate environments.
The current BBQ bottle workflow includes its frozen reference bundle in Git;
see [the ARX cuRobo guide](../scripts/curobo_arx/README.md) for startup and recording.
Other historical experiment inputs and results under `runs/` remain external archives.

Run it from the repository root:

```bash
./scripts/run_sim_client.sh
./scripts/run_sim_client.sh --webrtc-port 49200
ARX_SIM_WEBRTC_PORT=49200 ./scripts/run_sim_client.sh
./scripts/run_sim_client.sh --task grasp
```

Select an optional task-specific initial pose and object layout with `--task`:

```bash
./scripts/run_sim_client.sh --task grasp
```

The `grasp` layout places the robot in front of and facing the coffee table and
adds red, blue, and green 5 cm dynamic cubes along the robot-facing table edge.
Task definitions live under `arx-sim/config/tasks/`; they compose over the base
configuration and do not modify the locked robot or environment assets. Omitting
`--task` preserves the base scene exactly.

The default WebRTC signaling port is `49100`; the adapter internally reserves
the following UDP port for media. `ISAAC_SIM_ROOT` and `ARX_SIM_ASSET_ROOT` may
be used for deployment paths. These settings remain local to `arx-sim` and
never appear in the robot HTTP API.

The real client remains `./scripts/run_arx_client.sh`; activating its original
Conda environment preserves the real-machine startup behavior.

Arm status and home accept the `arm_id` query parameter (`right` by default,
or `left`). End-pose moves accept `arm_id` in their JSON body. Gripper open
and close accept an optional JSON body such as
`{"arm_id": "left", "duration_sec": 2.0}`; duration is not a query parameter.
Both arms use the existing simulator IK and execution logic, with separate
joint, end-effector, gripper, and status feedback selection. Commands retain
the simulator's single-active-command scheduling; this does not add concurrent
dual-arm motion. Durations use simulation time and queue deadlines use wall time.

RGB-D capture accepts `camera_id` values from `configs/arx.yaml`, including
`right_hand`, `left_hand`, and `head`. An explicit `serial` takes precedence
for sensor selection, matching the real client. The response distinguishes
the logical `camera_id` from the sensor's `serial_number`.

The default camera rendering profile uses PathTracing with 4 samples per frame,
OptiX denoising, and native-resolution TAA. This avoids zero RGBA output from
the DLSS path on the local Isaac Sim 6.0.1 Linux stack; TAA alone in Real-Time
2.0 still produces noisy images. These rendering settings live in
`config/runtime.yaml` and preserve camera calibration and scene physics.

As in the real home implementation, `auto_quit_after` remains a compatibility
parameter; completion uses the home feedback thresholds. Hardware process
commands and diagnostics are not reproduced by the simulator.

On hosts without a system ROS installation, `scripts/run_dualmap_server.sh`
automatically sources `arx-sim/scripts/activate_ros2_runtime.sh`. This exposes
Isaac Sim's bundled ROS 2 Jazzy client runtime to the existing upper-layer
Python environment without replacing its NumPy, OpenCV, PyTorch, or Open3D
packages. `ROS_DOMAIN_ID` defaults to `0` and `RMW_IMPLEMENTATION` defaults to
`rmw_fastrtps_cpp`; explicit deployment values remain respected.
