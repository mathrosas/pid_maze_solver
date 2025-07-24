#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <thread>
#include <vector>

#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#include "eigen3/Eigen/Dense"
#include "rosidl_runtime_c/message_initialization.h"

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "yaml-cpp/yaml.h"
#include <filesystem>

using namespace std::chrono_literals;

class PIDMazeSolver : public rclcpp::Node {
public:
  PIDMazeSolver(int scene_number)
      : Node("pid_maze_solver"), scene_number_(scene_number) {
    RCLCPP_INFO(get_logger(),
                "PID maze solver node with 360° obstacle avoidance.");

    pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/filtered", 10,
        std::bind(&PIDMazeSolver::odom_callback, this, std::placeholders::_1));
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", 10,
        std::bind(&PIDMazeSolver::scan_callback, this, std::placeholders::_1));

    motions_ = readWaypointsYAML();
    if (motions_.empty()) {
      RCLCPP_ERROR(get_logger(), "No waypoints loaded, shutting down.");
      rclcpp::shutdown();
    }
  }

  void run() {
    while ((!odom_received_ || !scan_received_) && rclcpp::ok()) {
      rclcpp::spin_some(shared_from_this());
      rclcpp::sleep_for(50ms);
    }

    double error_phi;
    double goal_x = x_, goal_y = y_, goal_phi = phi_;
    double Kp = 0.5, Ki = 0.05, Kd = 0.1;
    double error_phi_prev = 0, error_dist_prev = 0;
    double integral_phi = 0, integral_dist = 0;
    const double I_MAX = 1.0;
    const double V_MAX = 0.4;
    const double W_MAX = 0.4;

    geometry_msgs::msg::Twist twist;

    while (pub_->get_subscription_count() == 0) {
      rclcpp::sleep_for(50ms);
    }

    auto t0 = std::chrono::steady_clock::now();

    for (auto [rel_x, rel_y, rel_phi] : motions_) {
      goal_phi += rel_phi;
      goal_x += rel_x;
      goal_y += rel_y;

      error_dist_prev = error_phi_prev = 0.0;
      integral_dist = integral_phi = 0.0;

      // Rotation control
      while (std::abs(goal_phi - phi_) > ang_tol && (rel_phi != 0.0)) {
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
        if (dt <= 0.0)
          dt = 1e-3;

        error_phi = goal_phi - phi_;
        integral_phi = std::clamp(integral_phi + error_phi * dt, -I_MAX, I_MAX);
        double derivative_phi = (error_phi - error_phi_prev) / dt;
        double PID_phi =
            Kp * error_phi + Ki * integral_phi + Kd * derivative_phi;
        PID_phi = std::clamp(PID_phi, -W_MAX, W_MAX);
        error_phi_prev = error_phi;

        twist.linear.x = 0.0;
        twist.linear.y = 0.0;
        twist.angular.z = PID_phi;

        apply_course_correction(twist);
        pub_->publish(twist);

        RCLCPP_INFO(get_logger(),
                    "angle_error=%.3f -> w_cmd=%.3f | Walls: Front:%.2fm "
                    "Left:%.2fm Right:%.2fm Back:%.2fm",
                    error_phi, twist.angular.z, last_front_dist_,
                    last_left_dist_, last_right_dist_, last_back_dist_);

        rclcpp::spin_some(shared_from_this());
        rclcpp::sleep_for(25ms);
      }

      // Translation control
      while (true) {
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
        if (dt <= 0)
          dt = 1e-3;

        double ex = goal_x - x_;
        double ey = goal_y - y_;
        double dist = std::hypot(ex, ey);
        if (dist <= pos_tol)
          break;

        integral_dist = std::clamp(integral_dist + dist * dt, -I_MAX, I_MAX);
        double derivative_dist = (dist - error_dist_prev) / dt;
        double v = Kp * dist + Ki * integral_dist + Kd * derivative_dist;
        v = std::clamp(v, -V_MAX, V_MAX);
        error_dist_prev = dist;

        double ux = ex / dist;
        double uy = ey / dist;
        double vx_body = std::cos(phi_) * v * ux + std::sin(phi_) * v * uy;
        double vy_body = -std::sin(phi_) * v * ux + std::cos(phi_) * v * uy;

        twist.linear.x = vx_body;
        twist.linear.y = vy_body;
        twist.angular.z = 0.0;

        apply_course_correction(twist);
        pub_->publish(twist);

        RCLCPP_INFO(
            get_logger(),
            "dist=%.2f -> v=%.2f (vx=%.2f, vy=%.2f) | Walls: Front:%.2fm "
            "Left:%.2fm Right:%.2fm Back:%.2fm",
            dist, v, twist.linear.x, twist.linear.y, last_front_dist_,
            last_left_dist_, last_right_dist_, last_back_dist_);

        rclcpp::spin_some(shared_from_this());
        rclcpp::sleep_for(25ms);
      }

      stop();
    }
  }

private:
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  std::vector<std::tuple<double, double, double>> motions_;
  std::vector<float> scan_ranges_;
  sensor_msgs::msg::LaserScan::SharedPtr scan_msg_;

  double x_ = 0, y_ = 0, phi_ = 0;
  float last_front_dist_ = 0.0;
  float last_left_dist_ = 0.0;
  float last_right_dist_ = 0.0;
  float last_back_dist_ = 0.0;
  const double pos_tol = 0.01;
  const double ang_tol = 0.01;

  int scene_number_;
  bool odom_received_ = false;
  bool scan_received_ = false;

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    x_ = msg->pose.pose.position.x;
    y_ = msg->pose.pose.position.y;
    tf2::Quaternion q(
        msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    phi_ = yaw;
    odom_received_ = true;
  }

  void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
    scan_msg_ = msg;
    scan_ranges_ = msg->ranges;
    scan_received_ = true;
  }

  float get_sector_distance(int center_idx, int width) {
    int start = std::max(0, center_idx - width);
    int end =
        std::min(static_cast<int>(scan_ranges_.size()) - 1, center_idx + width);
    return *std::min_element(scan_ranges_.begin() + start,
                             scan_ranges_.begin() + end + 1);
  }

  void apply_course_correction(geometry_msgs::msg::Twist &twist) {
    if (!scan_received_ || scan_ranges_.empty() || !scan_msg_)
      return;

    float angle_min = scan_msg_->angle_min;
    float angle_increment = scan_msg_->angle_increment;

    // Front sector (directly ahead)
    int front_idx = static_cast<int>((M_PI - angle_min) / angle_increment);

    // Right sector (90° right)
    int right_idx = static_cast<int>((M_PI_2 - angle_min) / angle_increment);

    // Left sector (90° left)
    int left_idx = static_cast<int>((-M_PI_2 - angle_min) / angle_increment);

    // Back sector (directly behind)
    int back_idx = static_cast<int>((0.0 - angle_min) / angle_increment);

    int max_idx = static_cast<int>(scan_ranges_.size()) - 1;
    front_idx = std::clamp(front_idx, 0, max_idx);
    right_idx = std::clamp(right_idx, 0, max_idx);
    left_idx = std::clamp(left_idx, 0, max_idx);
    back_idx = std::clamp(back_idx, 0, max_idx);

    const int sector_width = 5;
    last_front_dist_ = get_sector_distance(front_idx, sector_width);
    last_right_dist_ = get_sector_distance(right_idx, sector_width);
    last_left_dist_ = get_sector_distance(left_idx, sector_width);
    last_back_dist_ = get_sector_distance(back_idx, sector_width);

    const float min_safe_dist =
        0.22f; // I chose this distance, because 23-24
               // was the exact middle of the road and I
               // wanted a little more of space to work with
    const float move_distance = 0.1f;

    // Front too close - move backward and turn
    if (last_front_dist_ < min_safe_dist) {
      twist.linear.x = -move_distance;
      twist.linear.y = 0.0;
    }
    // Right too close - move left
    else if (last_right_dist_ < min_safe_dist) {
      twist.linear.x = 0.0;
      twist.linear.y = move_distance;
      twist.angular.z = 0.0;
    }
    // Left too close - move right
    else if (last_left_dist_ < min_safe_dist) {
      twist.linear.x = 0.0;
      twist.linear.y = -move_distance;
      twist.angular.z = 0.0;
    }
    // Back too close - move forward
    else if (last_back_dist_ < min_safe_dist) {
      twist.linear.x = move_distance;
      twist.linear.y = 0.0;
      twist.angular.z = 0.0;
    }
  }

  void stop() {
    geometry_msgs::msg::Twist twist;
    rclcpp::Rate rate(20);
    for (int i = 0; i < 20; ++i) {
      pub_->publish(twist);
      rclcpp::spin_some(shared_from_this());
      rate.sleep();
    }
  }

  std::vector<std::tuple<double, double, double>> readWaypointsYAML() {
    std::vector<std::tuple<double, double, double>> waypoints;
    std::string pkg_share =
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
    default:
      RCLCPP_ERROR(get_logger(), "Invalid scene_number_: %d", scene_number_);
      return waypoints;
    }

    std::string path = pkg_share + "/waypoints/" + waypoint_file_name;

    try {
      YAML::Node config = YAML::LoadFile(path);
      if (!config["waypoints"])
        return waypoints;

      for (std::size_t i = 0; i < config["waypoints"].size(); ++i) {
        const auto &wp = config["waypoints"][i];
        double x = wp[0].as<double>();
        double y = wp[1].as<double>();
        double phi = wp[2].as<double>();
        waypoints.emplace_back(x, y, phi);
      }
    } catch (const YAML::Exception &e) {
      RCLCPP_ERROR(get_logger(), "YAML error: %s", e.what());
    }

    return waypoints;
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node =
      std::make_shared<PIDMazeSolver>(argc > 1 ? std::atoi(argv[1]) : 1);
  node->run();
  rclcpp::shutdown();
  return 0;
}