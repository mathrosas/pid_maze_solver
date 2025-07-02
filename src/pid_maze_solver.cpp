#include <chrono>
#include <cmath>
#include <memory>
#include <thread>
#include <vector>

#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

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
    RCLCPP_INFO(get_logger(), "PID maze solver node.");
    // pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
    //     "wheel_speed", 10);
    pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/filtered", 10,
        std::bind(&PIDMazeSolver::odom_callback, this, std::placeholders::_1));
    // sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    //     "/rosbot_xl_base_controller/odom", 10,
    //     std::bind(&PIDMazeSolver::odom_callback, this,
    //               std::placeholders::_1));

    motions_ = readWaypointsYAML();
    if (motions_.empty()) {
      RCLCPP_ERROR(get_logger(), "No waypoints loaded, shutting down.");
      rclcpp::shutdown();
    }
  }

  void run() {
    // wait until we actually have a valid odom
    RCLCPP_INFO(get_logger(), "Waiting for first odometry…");
    while (!odom_received_ && rclcpp::ok()) {
      rclcpp::spin_some(shared_from_this());
      rclcpp::sleep_for(50ms);
    }
    RCLCPP_INFO(get_logger(), "Got odom. Starting control loops.");

    double error_phi;
    double goal_x = x_, goal_y = y_, goal_phi = phi_;
    double Kp = 0.5, Ki = 0.05, Kd = 0.1;
    double error_phi_prev, error_dist_prev;
    double integral_phi = 0;
    double integral_dist = 0;
    double derivative_phi = 0;
    double PID_phi;
    double I_MAX = 1.0; // integrate clamp [–1,1]
    double V_MAX = 0.4; // max linear m/s
    double W_MAX = 0.4; // max angular rad/s

    geometry_msgs::msg::Twist twist;

    while (pub_->get_subscription_count() == 0) {
      rclcpp::sleep_for(50ms);
    }

    auto t0 = std::chrono::steady_clock::now();

    motions_ = readWaypointsYAML();

    for (auto [rel_x, rel_y, rel_phi] : motions_) {
      goal_phi += rel_phi;
      goal_x += rel_x;
      goal_y += rel_y;

      // reset previous errors/integrals for each segment if you like
      error_dist_prev = error_phi_prev = 0.0;
      integral_dist = integral_phi = 0.0;

      // **2) main control loop**
      // Main PID loop: until we reach the angular tolerance
      while (std::abs(goal_phi - phi_) > ang_tol) {
        // ——— timing ———
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
        if (dt <= 0.0)
          dt = 1e-3;

        // ——— compute yaw error ———
        error_phi = goal_phi - phi_;

        // ——— integrate (with clamp) ———
        integral_phi = std::clamp(integral_phi + error_phi * dt, -I_MAX, I_MAX);

        // ——— derivative ———
        derivative_phi = (error_phi - error_phi_prev) / dt;

        // ——— PID output ———
        PID_phi = Kp * error_phi + Ki * integral_phi + Kd * derivative_phi;

        // ——— clamp command ———
        PID_phi = std::clamp(PID_phi, -W_MAX, W_MAX);

        // ——— save for next step ———
        error_phi_prev = error_phi;

        twist.linear.x = 0.0;
        twist.linear.y = 0.0;
        twist.angular.z = PID_phi;

        // Publish to /cmd_vel
        pub_->publish(twist);

        rclcpp::spin_some(shared_from_this());
        rclcpp::sleep_for(25ms);

        // ——— debug print ———
        RCLCPP_INFO(get_logger(), "angle_error=%.3f -> w_cmd=%.3f", error_phi,
                    PID_phi);
      }

      while (true) {
        // compute dt
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
        if (dt <= 0)
          dt = 1e-3;

        // radial error
        double ex = goal_x - x_;
        double ey = goal_y - y_;
        double dist = std::hypot(ex, ey);
        if (dist <= pos_tol)
          break;

        // PID on distance as before
        integral_dist = std::clamp(integral_dist + dist * dt, -I_MAX, I_MAX);
        double derivative_dist = (dist - error_dist_prev) / dt;
        double v = Kp * dist + Ki * integral_dist + Kd * derivative_dist;
        v = std::clamp(v, -V_MAX, V_MAX);
        error_dist_prev = dist;

        // world-frame unit vector
        double ux = ex / dist;
        double uy = ey / dist;
        double vx = v * ux;
        double vy = v * uy;

        double vx_body = std::cos(phi_) * vx + std::sin(phi_) * vy;
        double vy_body = -std::sin(phi_) * vx + std::cos(phi_) * vy;

        twist.linear.x = vx_body;
        twist.linear.y = vy_body;
        twist.angular.z = 0.0;

        // Publish to /cmd_vel
        pub_->publish(twist);

        rclcpp::spin_some(shared_from_this());
        rclcpp::sleep_for(25ms);

        RCLCPP_INFO(get_logger(),
                    "dist=%.2f -> v=%.2f (vx_body=%.2f, vy_body=%.2f)", dist, v,
                    vx_body, vy_body);
      }

      stop();
    }
  }

private:
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_;
  std::vector<std::tuple<double, double, double>> motions_;
  double x_, y_, phi_;
  double w_, l_, r_;

  double pos_tol = 0.01;
  double ang_tol = 0.01;

  int scene_number_;

  bool odom_received_ = false;

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    x_ = msg->pose.pose.position.x;
    y_ = msg->pose.pose.position.y;

    // Extract yaw (phi) from quaternion
    tf2::Quaternion q(
        msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    phi_ = yaw;
    odom_received_ = true;
  }

  void stop() {
    geometry_msgs::msg::Twist twist;
    rclcpp::Rate rate(20);
    for (int i = 0; i < 20; ++i) {
      pub_->publish(twist);
      rclcpp::spin_some(shared_from_this());
      rate.sleep();
    }
    RCLCPP_INFO(get_logger(), "Stop (zeroed for 0.5 s)");
  }

  std::vector<std::tuple<double, double, double>> readWaypointsYAML() {
    std::vector<std::tuple<double, double, double>> waypoints;

    // 1) locate the YAML file
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
    case 3: // Simulation Reverse
      waypoint_file_name = "reverse_waypoints_sim.yaml";
      break;
    default:
      RCLCPP_ERROR(get_logger(), "Invalid scene_number_: %d", scene_number_);
      return waypoints;
    }
    std::string path = pkg_share + "/waypoints/" + waypoint_file_name;
    RCLCPP_INFO(get_logger(), "Loading waypoints from: %s", path.c_str());

    // 2) parse it
    try {
      YAML::Node config = YAML::LoadFile(path);
      if (!config["waypoints"] || !config["waypoints"].IsSequence()) {
        RCLCPP_ERROR(get_logger(), "No “waypoints” sequence in %s",
                     path.c_str());
        return waypoints;
      }

      for (std::size_t i = 0; i < config["waypoints"].size(); ++i) {
        const auto &wp = config["waypoints"][i];
        double x, y, phi;

        if (wp.IsSequence() && wp.size() == 3) {
          x = wp[0].as<double>();
          y = wp[1].as<double>();
          phi = wp[2].as<double>();
        } else if (wp.IsMap()) {
          x = wp["x"].as<double>();
          y = wp["y"].as<double>();
          phi = wp["phi"].as<double>();
        } else {
          RCLCPP_WARN(get_logger(),
                      "Waypoint %zu has unexpected format; skipping.", i);
          continue;
        }

        waypoints.emplace_back(x, y, phi);
      }

    } catch (const YAML::Exception &e) {
      RCLCPP_ERROR(get_logger(), "Failed to load YAML file %s: %s",
                   path.c_str(), e.what());
    }

    return waypoints;
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  int scene_number = 1;
  if (argc > 1) {
    scene_number = std::atoi(argv[1]);
  }
  auto node = std::make_shared<PIDMazeSolver>(scene_number);
  node->run();
  rclcpp::shutdown();
  return 0;
}