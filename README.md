# Checkpoint 17 — PID Maze Solver

ROS 2 C++ **PID maze solver** for the **Husarion ROSBot XL** (4-wheel mecanum / holonomic). The node fuses the distance and turn PID controllers into a single two-phase state machine (`TURN → MOVE`) that drives the robot through a 14+1-waypoint YAML-loaded trajectory while a **laser-scan reactive safety layer** keeps it off the walls and an **IMU yaw-rate feedback** sharpens the turn-phase derivative. Works against both the Gazebo maze and the real CyberWorld ROSBot XL — four scenes cover sim / real × forward / reverse, selected via a **scene number** passed as a CLI argument.

<p align="center">
  <img src="media/waypoints-sim.png" alt="PID maze solver waypoint trace through the simulation maze" width="650"/>
</p>

## How It Works

<p align="center">
  <img src="media/maze-world.png" alt="Gazebo maze world top view with ROSBot XL" width="600"/>
</p>

### Control Loop

1. A single-node executable `pid_maze_solver` subscribes to:
   - `/odometry/filtered` (`nav_msgs/Odometry`) — pose `(x, y, φ)` and body-frame twist
   - `/scan_filtered` (`sensor_msgs/LaserScan`) — filtered 2D laser (topic name overridable via the `scan_topic` parameter)
   - `imu_broadcaster/imu` (`sensor_msgs/Imu`) — yaw-rate feedback for the turn phase D-term

   It publishes `geometry_msgs/Twist` on `/cmd_vel`. All subscriptions share a **Reentrant callback group** on a 4-thread `MultiThreadedExecutor`.
2. Yaw is extracted directly from the odometry quaternion via `tf2::getYaw` — no `tf2_ros::TransformListener` / TF tree lookup is required
3. `readWaypointsYAML()` loads a 15-waypoint file from `share/pid_maze_solver/waypoints/` for the requested scene. Both the flat `[dx, dy, dyaw, ...]` legacy layout and the newer `waypoints: [[dx, dy, dyaw], ...]` list are accepted
4. The first odom message starts a **20 Hz timer** (`dt = 50 ms`) that runs the per-segment state machine:
   - **`start_segment`** — latches the target yaw `target_yaw = current_yaw + wp.dyaw` and the target position, rotating the body-frame `(wp.dx, wp.dy)` by `target_yaw` (the heading the robot will face *after* the turn, not the current heading) so the robot slides along its *new* forward axis
   - **`TURN` phase** — PID on `e_yaw = wrap(target_yaw − current_yaw)`, with `de_yaw = 0 − imu_yaw_rate` from the IMU. Exits when `|e_yaw| < 0.01 rad` **and** `|yaw_rate| < angular_vel_tolerance_`, then zeroes the twist and switches to MOVE
   - **`MOVE` phase** — rotates the world-frame position error into the body frame, runs two independent PIDs on `(ex_b, ey_b)`, uses measured body-frame velocity for the D-term, caps `‖v‖ ≤ max_linear_vel_`, then runs `apply_wall_avoidance()` before publishing. Exits when both position (`< 0.01 m`) and velocity (`< 0.01 m/s`) drop below tolerance
5. Between segments the node **pauses for 1.5 s** with a zero-twist so the physical robot settles before the next `TURN → MOVE` cycle
6. On real-robot scenes an **anti-drift pass** runs at the end of each MOVE phase: if the residual yaw error exceeds `5°`, the next waypoint's `dyaw` is offset by that drift so accumulated heading error doesn't compound across the run
7. After the last waypoint, `rclcpp::shutdown()` is called

### PID Configuration

Gains and limits differ per scene. Shared parameters:

| Parameter | Value |
|---|---|
| Control rate | `20 Hz` (`dt = 50 ms`) |
| Inter-waypoint pause | `1.5 s` |
| Position tolerance | `0.01 m` |
| Angular tolerance | `0.01 rad` |
| Linear velocity tolerance | `0.01 m/s` |
| Slow-down distance | `0.5 m` |
| Correction speed | `0.05 m/s` |
| Target nudge step | `0.003 m / tick` |

Scene-dependent parameters:

| Parameter | Scenes 1 / 3 (Sim) | Scenes 2 / 4 (CyberWorld) |
|---|---|---|
| `Kp_x`, `Kp_y` | `2.5` | `2.1` |
| `Ki_x`, `Ki_y` | `0.005` | `0.001` |
| `Kd_x`, `Kd_y` | `0.3` | `0.3` |
| `Kp_yaw` | `1.3` | `1.25` |
| `Ki_yaw` | `0.001` | `0.001` |
| `Kd_yaw` | `0.3` | `0.3` |
| `max_linear_vel_` | `0.8 m/s` | `0.45 m/s` |
| `max_angular_vel_` | `3.14 rad/s` | `1.4 rad/s` |
| Angular velocity tolerance | `0.02 rad/s` | `0.05 rad/s` |
| `stop_distance_` / `correction_distance_` | `0.21 m` | `0.175 m` |
| Antenna echo centre | `−161°` | `−170°` |
| Anti-drift yaw compensation | disabled | enabled (`> 5°`) |

### Laser-Scan Safety Layer

The MOVE phase runs the laser through a three-stage pipeline before `/cmd_vel` is published:

1. **Cardinal sector sampling** — `sector_min()` picks the closest valid beam inside a `±30°` wedge centred on **front / left / back / right** in the base frame. The laser is mounted rotated by `π` in the base frame (`laser_yaw_in_base_ = π`), so beam angles are reprojected before comparison
2. **Antenna echo rejection** — beams inside a narrow cone around the configured `antenna_center_` with range below `antenna_max_range_ = 0.21 m` are dropped; this filters out the robot's own antenna reflecting back into the front-left sector
3. **Correction pass** — if any cardinal sector is inside `correction_distance_`, the command velocity on that axis is clipped *and* the target pose is nudged by `target_nudge_step_` in the opposite direction (in base frame). The target nudge is crucial: without it, the PID would fight the velocity clip and stall the robot against the wall
4. **Slow-near-obstacle** — if no correction fired but an obstacle inside the motion cone is within `slow_distance_`, the speed cap is scaled linearly from `0` at `correction_distance_` up to `max_linear_vel_` at `slow_distance_`

## Waypoint Scenes

One executable, four scenes via CLI argument. Each scene loads a 15-waypoint YAML file of `[dx, dy, dyaw]` triplets (the final triplet `[0, 0, 3.1416]` is a half-turn "park" move):

| `scene_number` | Waypoint file | Description |
|---|---|---|
| `1` (default) | `waypoints_sim.yaml` | Simulation — forward traversal |
| `2` | `waypoints_real.yaml` | Real CyberWorld — forward |
| `3` | `reverse_waypoints_sim.yaml` | Simulation — reverse |
| `4` | `reverse_waypoints_real.yaml` | Real CyberWorld — reverse |

### Forward — Simulation / CyberWorld

<p align="center">
  <img src="media/maze-solver-sim.gif" alt="PID maze solver in the Gazebo maze world" width="650"/>
</p>

<p align="center">
  <img src="media/maze-solver-real.gif" alt="PID maze solver on the real ROSBot XL in CyberWorld" width="650"/>
</p>

### Reverse — Simulation / CyberWorld

<p align="center">
  <img src="media/waypoints-reverse-sim.png" alt="Reverse maze solver waypoints in simulation" width="650"/>
</p>

<p align="center">
  <img src="media/maze-solver-reverse-sim.gif" alt="Reverse PID maze solver in the Gazebo maze world" width="650"/>
</p>

<p align="center">
  <img src="media/maze-solver-reverse-real.gif" alt="Reverse PID maze solver on the real ROSBot XL in CyberWorld" width="650"/>
</p>

## Real Robot Deployment (CyberWorld)

<p align="center">
  <img src="media/waypoints-real.png" alt="Real ROSBot XL PID maze solver waypoint trace in the CyberWorld physical maze" width="650"/>
</p>

<p align="center">
  <img src="media/waypoints-reverse-real.png" alt="Real ROSBot XL reverse PID maze solver waypoint trace in the CyberWorld physical maze" width="650"/>
</p>

The same executable runs **unmodified** on the real Husarion ROSBot XL in The Construct's **CyberWorld** lab — only the scene number changes. Scenes `2` and `4` load hand-tuned waypoint files **and** swap to the real-robot PID / safety parameter set:

1. The ROSBot XL real-robot stack (`rosbot_xl_ros` + EKF + `scan_filter_chain`) streams `/odometry/filtered`, `/scan_filtered` and `imu_broadcaster/imu` from CyberWorld — same topics the sim publishes
2. The `pid_maze_solver` node is launched locally:

   ```bash
   ros2 run pid_maze_solver pid_maze_solver 2   # forward run
   ros2 run pid_maze_solver pid_maze_solver 4   # reverse run
   ```
3. Real-robot-specific behaviour:
   - **Reduced gains** on all three axes to absorb wheel slip and localisation noise
   - **Speed caps** dropped to `0.45 m/s` linear and `1.4 rad/s` angular
   - **Closer safety distances** (`0.175 m` correction threshold) and a **repositioned antenna filter** (`−170°`) to match the real robot's physical geometry
   - **Anti-drift compensation**: the real mecanum platform collects several degrees of yaw drift over the run; pushing residual yaw into the next waypoint's `dyaw` keeps the path from walking off the maze corridor
   - **Relaxed angular-velocity tolerance** (`0.05 rad/s` vs `0.02` in sim) so the TURN→MOVE transition isn't blocked by residual IMU twitch

### Sim ↔ real parity

| Concern | Simulation (scenes 1 / 3) | Real CyberWorld (scenes 2 / 4) |
|---|---|---|
| Feedback | Odom pose + IMU yaw rate + laser | Odom pose + IMU yaw rate + laser |
| Safety scan | `/scan_filtered` (Gazebo plugin) | `/scan_filtered` (physical Hokuyo + filter chain) |
| Waypoint file | `waypoints_sim.yaml`, `reverse_waypoints_sim.yaml` | `waypoints_real.yaml`, `reverse_waypoints_real.yaml` |
| PID `(x, y)` gains | `Kp=2.5, Ki=0.005, Kd=0.3` | `Kp=2.1, Ki=0.001, Kd=0.3` |
| PID `yaw` gains | `Kp=1.3, Ki=0.001, Kd=0.3` | `Kp=1.25, Ki=0.001, Kd=0.3` |
| Max linear / angular | `0.8 m/s` / `3.14 rad/s` | `0.45 m/s` / `1.4 rad/s` |
| Tolerance (pos / yaw) | `0.01 m` / `0.01 rad` | `0.01 m` / `0.01 rad` |
| Safety distance | `0.21 m` | `0.175 m` |
| Anti-drift | off | on (threshold `5°`) |
| Clock | sim time | wall clock |
| Default scene | `1` | — |

## ROS 2 Interface

| Name | Type | Description |
|---|---|---|
| `/odometry/filtered` | `nav_msgs/Odometry` (sub) | EKF-fused pose `(x, y, φ)` + body-frame twist |
| `/scan_filtered` | `sensor_msgs/LaserScan` (sub) | Filtered 2D laser scan for the safety layer. Topic name overridable via parameter `scan_topic` |
| `imu_broadcaster/imu` | `sensor_msgs/Imu` (sub) | Angular velocity `ω_z` used as yaw-rate feedback for the TURN-phase derivative |
| `/cmd_vel` | `geometry_msgs/Twist` (pub) | Body-frame command (`v_x`, `v_y`, `ω_z`) |

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `scan_topic` | `/scan_filtered` | Laser scan topic for the safety layer |

## Project Structure

```
pid_maze_solver/
├── src/
│   └── pid_maze_solver.cpp
├── include/
├── waypoints/
│   ├── waypoints_sim.yaml
│   ├── waypoints_real.yaml
│   ├── reverse_waypoints_sim.yaml
│   └── reverse_waypoints_real.yaml
├── media/
├── CMakeLists.txt
└── package.xml
```

## How to Use

### Prerequisites

- ROS 2 Humble
- Gazebo (bundled with the `rosbot_xl_gazebo` simulation and maze world)
- `yaml-cpp`, `tf2`, `nav_msgs`, `sensor_msgs`, `geometry_msgs`
- `rosbot_xl_ros` stack in the same workspace (description + controllers + EKF + laser filter + IMU broadcaster)

### Build

```bash
cd ~/ros2_ws
colcon build --packages-select pid_maze_solver --symlink-install
source install/setup.bash
```

### Simulation — forward

```bash
# Terminal 1 — ROSBot XL + maze world in Gazebo
ros2 launch rosbot_xl_gazebo simulation.launch.py

# Terminal 2 — PID maze solver (scene 1 = sim forward, default)
ros2 run pid_maze_solver pid_maze_solver 1
```

The scene argument is optional — omitting it defaults to scene `1` (sim forward).

### Simulation — reverse

```bash
ros2 run pid_maze_solver pid_maze_solver 3
```

### Real robot (CyberWorld)

```bash
ros2 run pid_maze_solver pid_maze_solver 2   # forward
ros2 run pid_maze_solver pid_maze_solver 4   # reverse
```

### Sanity checks

```bash
ros2 topic echo /cmd_vel
ros2 topic echo /scan_filtered --once
ros2 topic echo /odometry/filtered --once
ros2 topic echo imu_broadcaster/imu --once
```

## Key Concepts Covered

- **Two-phase state machine** — per segment the robot first snaps to the target heading (`TURN`), then slides to the target position (`MOVE`); integrals and arrival gates are reset at each transition
- **3-DOF PID with heterogeneous feedback** — position D-terms use the **odometry body-frame velocity**, yaw D-term uses the **IMU yaw rate** directly (lower latency than differentiating yaw)
- **Body-frame waypoints, rotated by the post-turn heading** — so `(dx, dy)` is always evaluated along the axis the robot will be facing when the MOVE phase begins
- **Reactive four-sector laser safety layer** — `±30°` wedges around front / left / back / right, with antenna-echo rejection, in-place velocity clipping **and** target pose nudging so the PID doesn't fight the correction
- **Linear speed scaling near obstacles** — `slow_near_obstacle()` ramps the velocity cap between `correction_distance_` and `slow_distance_`
- **Anti-drift feedforward** — residual yaw at the end of each MOVE phase is folded into the next waypoint's `dyaw` on real-robot scenes, compensating accumulated heading error without retuning
- **YAML-driven scene selection** — one executable, four hand-tuned trajectories for sim / real × forward / reverse, plus per-scene parameter overrides (gains, speed caps, safety distances, antenna filter)
- **Flexible YAML schema** — accepts both the flat `[dx, dy, dyaw, ...]` legacy layout and the structured `waypoints: [[dx, dy, dyaw], ...]` list
- **Multi-threaded executor** — four threads so odom, scan, IMU and timer callbacks progress concurrently without blocking each other
- **Runtime-overridable scan topic** — `scan_topic` parameter decouples the solver from the laser pipeline's naming

## Technologies

- ROS 2 Humble
- C++ 17 (`rclcpp`, `tf2`, `nav_msgs`, `sensor_msgs`, `geometry_msgs`)
- `yaml-cpp` (waypoint loading)
- `ament_index_cpp` (runtime resolution of the installed `share/` directory)
- Husarion ROSBot XL (4-wheel mecanum) in Gazebo Sim + CyberWorld
