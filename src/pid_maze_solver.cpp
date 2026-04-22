#include "ament_index_cpp/get_package_share_directory.hpp"
#include "yaml-cpp/yaml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <geometry_msgs/msg/twist.hpp>
#include <limits>
#include <memory>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <stdexcept>
#include <string>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <vector>

struct Waypoint {
  double dx;
  double dy;
  double dyaw;
};

struct PIDGains {
  double kp{0.0};
  double ki{0.0};
  double kd{0.0};
};

struct SectorReading {
  double range{std::numeric_limits<double>::infinity()};
  double angle{0.0};
  bool valid{false};
};

class PIDMazeSolver : public rclcpp::Node {
public:
  explicit PIDMazeSolver(int scene_number)
      : Node("pid_maze_solver"), scene_number_(scene_number) {
    scan_topic_ =
        this->declare_parameter<std::string>("scan_topic", "/scan_filtered");

    waypoints_ = readWaypointsYAML();
    if (waypoints_.empty()) {
      RCLCPP_ERROR(get_logger(), "No waypoints loaded, shutting down.");
      rclcpp::shutdown();
      return;
    }

    setup_scene();

    callback_group_ =
        this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    rclcpp::SubscriptionOptions options;
    options.callback_group = callback_group_;

    odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/filtered", 10,
        std::bind(&PIDMazeSolver::odom_callback, this, std::placeholders::_1),
        options);

    scan_subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        scan_topic_, 10,
        std::bind(&PIDMazeSolver::scan_callback, this, std::placeholders::_1),
        options);

    imu_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "imu_broadcaster/imu", 10,
        std::bind(&PIDMazeSolver::imu_callback, this, std::placeholders::_1),
        options);

    cmd_vel_publisher_ =
        this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

    RCLCPP_INFO(get_logger(), "PID maze solver ready. Using scan topic: %s",
                scan_topic_.c_str());
  }

private:
  enum class Phase { TURN, MOVE };

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr
      scan_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
  rclcpp::TimerBase::SharedPtr control_timer_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;

  int scene_number_{1};
  std::string scan_topic_;
  Phase phase_{Phase::TURN};

  double current_x_{0.0};
  double target_x_{0.0};
  double current_y_{0.0};
  double target_y_{0.0};
  double current_yaw_{0.0};
  double target_yaw_{0.0};
  geometry_msgs::msg::Twist current_twist_;
  double imu_yaw_rate_{0.0};
  bool segment_active_{false};
  bool odom_received_{false};
  std::vector<Waypoint> waypoints_;
  size_t current_idx_{0};

  geometry_msgs::msg::Twist cmd_;

  PIDGains pid_x_{0.35, 0.005, 0.32};
  PIDGains pid_y_{0.35, 0.005, 0.32};
  PIDGains pid_yaw_{0.7, 0.001, 0.25};
  double sum_I_x_{0.0};
  double sum_I_y_{0.0};
  double sum_I_yaw_{0.0};

  const double rate_hz_{20.0};
  const double dt_{1.0 / rate_hz_};
  double max_angular_vel_{3.14};
  double max_linear_vel_{0.80};
  const double angular_tolerance_{0.01};
  const double linear_tolerance_{0.01};
  double angular_vel_tolerance_{0.02};
  const double linear_vel_tolerance_{0.01};

  bool pausing_{false};
  int pause_ticks_{0};
  const double pause_duration_sec_{1.5};
  const int pause_ticks_goal_{static_cast<int>(pause_duration_sec_ * rate_hz_)};

  std::mutex scan_mutex_;
  sensor_msgs::msg::LaserScan::SharedPtr latest_scan_;
  const double laser_yaw_in_base_{M_PI};
  const double sector_half_width_{30.0 * M_PI / 180.0};
  double stop_distance_{0.21};
  double correction_distance_{0.21};
  double slow_distance_{0.5};
  double correction_speed_{0.05};
  double target_nudge_step_{0.003};
  const double min_valid_range_{0.05};

  double antenna_center_{-170.0 * M_PI / 180.0};
  const double antenna_half_width_{0.15};
  double antenna_max_range_{0.21};

  const double yaw_drift_threshold_{5.0 * M_PI / 180.0};
  bool anti_drift_{false};

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    current_x_ = msg->pose.pose.position.x;
    current_y_ = msg->pose.pose.position.y;
    tf2::Quaternion q(
        msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    current_yaw_ = tf2::getYaw(q);
    current_twist_ = msg->twist.twist;

    if (!odom_received_) {
      odom_received_ = true;
      RCLCPP_INFO(get_logger(),
                  "First odom: x=%.3f y=%.3f yaw=%.3f. Starting control loop.",
                  current_x_, current_y_, current_yaw_);

      control_timer_ = this->create_wall_timer(
          std::chrono::duration<double>(dt_),
          std::bind(&PIDMazeSolver::control_loop, this), callback_group_);
    }
  }

  void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(scan_mutex_);
    latest_scan_ = msg;
  }

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    imu_yaw_rate_ = msg->angular_velocity.z;
  }

  std::vector<Waypoint> readWaypointsYAML() {
    std::vector<Waypoint> waypoints;

    const std::string package_share_directory =
        ament_index_cpp::get_package_share_directory("pid_maze_solver");

    std::string waypoint_file_name;
    switch (scene_number_) {
    case 1:
      waypoint_file_name = "waypoints_sim.yaml";
      break;
    case 2:
      waypoint_file_name = "waypoints_real.yaml";
      break;
    case 3:
      waypoint_file_name = "reverse_waypoints_sim.yaml";
      break;
    case 4:
      waypoint_file_name = "reverse_waypoints_real.yaml";
      break;
    default:
      RCLCPP_ERROR(get_logger(), "Invalid Scene Number: %d", scene_number_);
      return waypoints;
    }

    const std::string yaml_file_path =
        package_share_directory + "/waypoints/" + waypoint_file_name;

    try {
      YAML::Node config = YAML::LoadFile(yaml_file_path);
      YAML::Node list = config["waypoints"];

      if (list) {
        for (const auto &wp_node : list) {
          if (!wp_node.IsSequence() || wp_node.size() != 3) {
            RCLCPP_ERROR(get_logger(),
                         "Invalid waypoint in %s, expected [dx, dy, dyaw]",
                         yaml_file_path.c_str());
            continue;
          }
          waypoints.push_back({wp_node[0].as<double>(), wp_node[1].as<double>(),
                               wp_node[2].as<double>()});
        }
      } else {
        const std::string key = (scene_number_ == 1 || scene_number_ == 3)
                                    ? "waypoints_sim"
                                    : "waypoints_real";
        YAML::Node flat =
            config["pid_maze_solver"]["ros__parameters"][key.c_str()];

        if (!flat || !flat.IsSequence() || flat.size() % 3 != 0) {
          RCLCPP_ERROR(get_logger(),
                       "YAML file %s must contain either top-level "
                       "'waypoints' or flat '%s' triples",
                       yaml_file_path.c_str(), key.c_str());
          return waypoints;
        }

        for (std::size_t i = 0; i < flat.size(); i += 3) {
          waypoints.push_back({flat[i].as<double>(), flat[i + 1].as<double>(),
                               flat[i + 2].as<double>()});
        }
      }

      RCLCPP_INFO(get_logger(), "Loaded %zu waypoints from %s",
                  waypoints.size(), yaml_file_path.c_str());
    } catch (const YAML::Exception &e) {
      RCLCPP_ERROR(get_logger(), "Failed to load YAML file %s: %s",
                   yaml_file_path.c_str(), e.what());
    }

    return waypoints;
  }

  void setup_scene() {
    correction_distance_ = 0.21;
    stop_distance_ = 0.21;
    slow_distance_ = 0.5;
    correction_speed_ = 0.05;
    target_nudge_step_ = 0.003;
    anti_drift_ = false;

    switch (scene_number_) {
    case 1:
    case 3: {
      pid_x_ = {2.5, 0.005, 0.3};
      pid_y_ = {2.5, 0.005, 0.3};
      pid_yaw_ = {1.3, 0.001, 0.3};
      antenna_center_ = -161.0 * M_PI / 180.0;

      max_angular_vel_ = 3.14;
      max_linear_vel_ = 0.8;
      angular_vel_tolerance_ = 0.02;
      break;
    }

    case 2:
    case 4: {
      pid_x_ = {2.1, 0.001, 0.3};
      pid_y_ = {2.1, 0.001, 0.3};
      pid_yaw_ = {1.25, 0.001, 0.3};
      antenna_center_ = -170.0 * M_PI / 180.0;
      stop_distance_ = 0.175;
      correction_distance_ = stop_distance_;

      max_angular_vel_ = 1.4;
      max_linear_vel_ = 0.45;
      angular_vel_tolerance_ = 0.05;
      anti_drift_ = true;
      break;
    }

    default:
      RCLCPP_FATAL(get_logger(), "Invalid scene_number: %d", scene_number_);
      throw std::runtime_error("Invalid scene_number");
    }

    RCLCPP_INFO(get_logger(),
                "Scene %d: Kp=(%.2f, %.2f, %.2f) Ki=(%.4f, %.4f, %.4f) "
                "Kd=(%.2f, %.2f, %.2f) v_max=%.2f, w_max=%.2f",
                scene_number_, pid_x_.kp, pid_y_.kp, pid_yaw_.kp, pid_x_.ki,
                pid_y_.ki, pid_yaw_.ki, pid_x_.kd, pid_y_.kd, pid_yaw_.kd,
                max_linear_vel_, max_angular_vel_);
  }

  void control_loop() {
    if (pausing_) {
      stop();
      pause_ticks_++;
      if (pause_ticks_ >= pause_ticks_goal_) {
        pausing_ = false;
      }
      return;
    }

    if (current_idx_ >= waypoints_.size()) {
      stop();
      RCLCPP_INFO(get_logger(), "Trajectory completed.");
      rclcpp::shutdown();
      return;
    }

    if (!segment_active_) {
      start_segment();
    }

    if (phase_ == Phase::TURN) {
      const double e_yaw = normalize_angle(target_yaw_ - current_yaw_);
      const double de_yaw = 0.0 - imu_yaw_rate_;

      if (std::abs(e_yaw) < angular_tolerance_ &&
          std::abs(de_yaw) < angular_vel_tolerance_) {
        RCLCPP_INFO(get_logger(),
                    "Segment %zu: turn done (e_yaw=%.3f). Starting move.",
                    current_idx_, e_yaw);
        sum_I_yaw_ = 0.0;
        phase_ = Phase::MOVE;
        stop();
        return;
      }

      compute_turn_pid(e_yaw, de_yaw);
    } else {
      const double ex = target_x_ - current_x_;
      const double ey = target_y_ - current_y_;
      const double ex_b =
          std::cos(current_yaw_) * ex + std::sin(current_yaw_) * ey;
      const double ey_b =
          -std::sin(current_yaw_) * ex + std::cos(current_yaw_) * ey;
      const double dex = 0.0 - current_twist_.linear.x;
      const double dey = 0.0 - current_twist_.linear.y;
      const double dist = std::hypot(ex_b, ey_b);
      const double yaw_drift = normalize_angle(target_yaw_ - current_yaw_);

      if (dist < linear_tolerance_ &&
          std::hypot(dex, dey) < linear_vel_tolerance_) {
        RCLCPP_INFO(get_logger(), "Segment %zu: move done (dist=%.3f).",
                    current_idx_, dist);

        const size_t next_idx = current_idx_ + 1;
        if (anti_drift_ && std::abs(yaw_drift) > yaw_drift_threshold_ &&
            next_idx < waypoints_.size()) {
          waypoints_[next_idx].dyaw += yaw_drift;
          RCLCPP_INFO(get_logger(),
                      "Applying yaw drift compensation of %.3f rad to "
                      "waypoint %zu (new dyaw=%.3f).",
                      yaw_drift, next_idx, waypoints_[next_idx].dyaw);
        }

        current_idx_++;
        segment_active_ = false;
        phase_ = Phase::TURN;
        stop();
        pausing_ = true;
        pause_ticks_ = 0;
        return;
      }

      compute_move_pid(ex_b, ey_b, dex, dey);
      apply_wall_avoidance();
    }

    cmd_vel_publisher_->publish(cmd_);
  }

  void start_segment() {
    const auto &wp = waypoints_[current_idx_];

    target_yaw_ = current_yaw_ + wp.dyaw;

    const double dx_world =
        wp.dx * std::cos(target_yaw_) - wp.dy * std::sin(target_yaw_);
    const double dy_world =
        wp.dx * std::sin(target_yaw_) + wp.dy * std::cos(target_yaw_);

    target_x_ = current_x_ + dx_world;
    target_y_ = current_y_ + dy_world;

    phase_ = Phase::TURN;
    segment_active_ = true;
    sum_I_x_ = sum_I_y_ = sum_I_yaw_ = 0.0;

    RCLCPP_INFO(get_logger(),
                "Segment %zu: wp(dx=%.3f, dy=%.3f, dyaw=%.3f) -> "
                "target yaw=%.3f, target pos=(%.3f, %.3f)",
                current_idx_, wp.dx, wp.dy, wp.dyaw, target_yaw_, target_x_,
                target_y_);
  }

  void compute_turn_pid(double e_yaw, double de_yaw) {
    sum_I_yaw_ += e_yaw * dt_;

    double u =
        pid_yaw_.kp * e_yaw + pid_yaw_.kd * de_yaw + pid_yaw_.ki * sum_I_yaw_;
    u = std::clamp(u, -max_angular_vel_, max_angular_vel_);

    cmd_.linear.x = 0.0;
    cmd_.linear.y = 0.0;
    cmd_.angular.z = u;
  }

  void compute_move_pid(double ex, double ey, double dex, double dey) {
    sum_I_x_ += ex * dt_;
    sum_I_y_ += ey * dt_;

    cmd_.linear.x = pid_x_.kp * ex + pid_x_.kd * dex + pid_x_.ki * sum_I_x_;
    cmd_.linear.y = pid_y_.kp * ey + pid_y_.kd * dey + pid_y_.ki * sum_I_y_;
    cmd_.angular.z = 0.0;

    cap_linear_speed();
  }

  void apply_wall_avoidance() {
    sensor_msgs::msg::LaserScan::SharedPtr scan_msg;
    {
      std::lock_guard<std::mutex> lock(scan_mutex_);
      scan_msg = latest_scan_;
    }

    if (!scan_msg) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "No laser scan on %s", scan_topic_.c_str());
      return;
    }

    const auto &scan = *scan_msg;
    if (scan.ranges.empty()) {
      return;
    }

    const SectorReading front = sector_min(scan, 0.0, sector_half_width_);
    const SectorReading left = sector_min(scan, M_PI / 2.0, sector_half_width_);
    const SectorReading back = sector_min(scan, M_PI, sector_half_width_);
    const SectorReading right =
        sector_min(scan, -M_PI / 2.0, sector_half_width_);

    double target_dx_base = 0.0;
    double target_dy_base = 0.0;
    bool corrected = false;

    if (left.valid && left.range < correction_distance_) {
      cmd_.linear.y = std::min(cmd_.linear.y, -correction_speed_);
      target_dy_base -= target_nudge_step_;
      corrected = true;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 500,
                           "Course correcting right: left wall %.2f m",
                           left.range);
    } else if (right.valid && right.range < correction_distance_) {
      cmd_.linear.y = std::max(cmd_.linear.y, correction_speed_);
      target_dy_base += target_nudge_step_;
      corrected = true;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 500,
                           "Course correcting left: right wall %.2f m",
                           right.range);
    }

    if (front.valid && front.range < correction_distance_) {
      cmd_.linear.x = std::min(cmd_.linear.x, -correction_speed_);
      target_dx_base -= target_nudge_step_;
      corrected = true;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 500,
                           "Avoiding front wall: %.2f m", front.range);
    } else if (back.valid && back.range < correction_distance_) {
      cmd_.linear.x = std::max(cmd_.linear.x, correction_speed_);
      target_dx_base += target_nudge_step_;
      corrected = true;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 500,
                           "Avoiding back wall: %.2f m", back.range);
    }

    if (corrected) {
      nudge_target_in_base(target_dx_base, target_dy_base);
      cap_linear_speed();
      return;
    }

    slow_near_obstacle(front, left, back, right);
  }

  SectorReading sector_min(const sensor_msgs::msg::LaserScan &scan,
                           double center_base, double half_width) const {
    SectorReading best;
    double beam_angle_scan = scan.angle_min;

    const double range_max =
        std::isfinite(scan.range_max) && scan.range_max > 0.0 ? scan.range_max
                                                              : 10.0;

    for (std::size_t i = 0; i < scan.ranges.size();
         ++i, beam_angle_scan += scan.angle_increment) {
      const double range = scan.ranges[i];
      if (!std::isfinite(range) || range < min_valid_range_ ||
          range > range_max) {
        continue;
      }

      const double angle_base =
          normalize_angle(beam_angle_scan + laser_yaw_in_base_);
      if (std::abs(normalize_angle(angle_base - center_base)) > half_width) {
        continue;
      }

      if (is_antenna_echo(angle_base, range)) {
        continue;
      }

      if (range < best.range) {
        best.range = range;
        best.angle = angle_base;
        best.valid = true;
      }
    }

    return best;
  }

  void slow_near_obstacle(const SectorReading &front, const SectorReading &left,
                          const SectorReading &back,
                          const SectorReading &right) {
    const double vx = cmd_.linear.x;
    const double vy = cmd_.linear.y;
    const double v_norm = std::hypot(vx, vy);
    if (v_norm <= 1e-3) {
      return;
    }

    const double motion_dir = std::atan2(vy, vx);
    double nearest = std::numeric_limits<double>::infinity();

    const auto consider = [&](const SectorReading &reading) {
      if (!reading.valid) {
        return;
      }
      if (std::abs(normalize_angle(reading.angle - motion_dir)) <=
          sector_half_width_) {
        nearest = std::min(nearest, reading.range);
      }
    };

    consider(front);
    consider(left);
    consider(back);
    consider(right);

    if (!std::isfinite(nearest) || nearest >= slow_distance_) {
      return;
    }

    const double scale = std::clamp(
        (nearest - correction_distance_) / (slow_distance_ - correction_distance_),
        0.0, 1.0);
    const double v_cap = scale * max_linear_vel_;

    if (v_norm > v_cap) {
      const double k = v_cap / v_norm;
      cmd_.linear.x *= k;
      cmd_.linear.y *= k;
    }
  }

  void nudge_target_in_base(double dx_base, double dy_base) {
    target_x_ +=
        dx_base * std::cos(current_yaw_) - dy_base * std::sin(current_yaw_);
    target_y_ +=
        dx_base * std::sin(current_yaw_) + dy_base * std::cos(current_yaw_);
  }

  void cap_linear_speed() {
    const double vnorm = std::hypot(cmd_.linear.x, cmd_.linear.y);
    if (vnorm > max_linear_vel_) {
      cmd_.linear.x *= max_linear_vel_ / vnorm;
      cmd_.linear.y *= max_linear_vel_ / vnorm;
    }
  }

  bool is_antenna_echo(double angle_base, double range) const {
    const double d = normalize_angle(angle_base - antenna_center_);
    return (std::abs(d) < antenna_half_width_) && (range < antenna_max_range_);
  }

  void stop() {
    cmd_ = geometry_msgs::msg::Twist{};
    cmd_vel_publisher_->publish(cmd_);
  }

  static double normalize_angle(double theta) {
    while (theta > M_PI) {
      theta -= 2.0 * M_PI;
    }
    while (theta < -M_PI) {
      theta += 2.0 * M_PI;
    }
    return theta;
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  int scene_number = 1;
  if (argc > 1) {
    scene_number = std::atoi(argv[1]);
  }

  auto node = std::make_shared<PIDMazeSolver>(scene_number);
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(),
                                                    4);
  executor.add_node(node);

  try {
    executor.spin();
  } catch (const std::exception &e) {
    RCLCPP_ERROR(node->get_logger(), "Exception: %s", e.what());
  }

  rclcpp::shutdown();
  return 0;
}
