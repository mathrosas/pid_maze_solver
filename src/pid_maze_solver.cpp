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

using namespace std::chrono_literals;

class PIDMazeSolver : public rclcpp::Node {
public:
  PIDMazeSolver(int scene_number)
      : Node("distance_controller"), scene_number_(scene_number) {
    RCLCPP_INFO(get_logger(), "Distance controller node.");
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

    // Robot geometry
    w_ = 0.26969 / 2.0; // half wheelbase
    l_ = 0.17000 / 2.0; // half track width
    r_ = 0.10000 / 2.0; // wheel radius
    select_waypoints();
  }

  void run() {
    double error_phi, error_x, error_y;
    double goal_x = 0.0, goal_y = 0.0, goal_phi = 0.0;
    double Kp = 0.5, Ki = 0.0, Kd = 0.05;
    double error_phi_prev, error_dist_prev;
    double integral_phi = 0;
    double integral_dist = 0;
    double derivative_phi = 0;
    double derivative_dist = 0;
    double PID_phi;
    double I_MAX = 1.0; // integrate clamp [–1,1]
    double V_MAX = 0.5; // max linear m/s
    double W_MAX = 1.0; // max angular rad/s

    while (pub_->get_subscription_count() == 0) {
      rclcpp::sleep_for(100ms);
    }

    auto t0 = std::chrono::steady_clock::now();

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

        // ——— publish via holonomic pipeline ———
        //  1) world->body twist
        auto [wz, vx, vy] = velocity2twist(PID_phi, 0.0, 0.0);
        //  2) twist->wheel speeds
        auto wheels = twist2wheels(wz, 0.0, 0.0);
        //  3) wheel speeds->safe Twist & publish
        wheels2twist(wheels);

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
        double vx_w = v * ux;
        double vy_w = v * uy;

        // ** transform into body frame **
        auto [wz_body, vx_body, vy_body] = velocity2twist(0.0, vx_w, vy_w);

        // holonomic drive -> wheel speeds -> publish
        auto wheels = twist2wheels(wz_body, vx_body, vy_body);
        wheels2twist(wheels);

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

  double pos_tol = 0.1;
  double ang_tol = 0.1;

  int scene_number_;

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    x_ = msg->pose.pose.position.x;
    y_ = msg->pose.pose.position.y;

    // Extract yaw (φ) from quaternion
    tf2::Quaternion q(
        msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    phi_ = yaw;
  }

  std::tuple<double, double, double>
  velocity2twist(double error_phi, double error_x, double error_y) {
    // Build rotation matrix R(φ)
    Eigen::Matrix3d R;
    R << 1, 0, 0, 0, std::cos(phi_), std::sin(phi_), 0, -std::sin(phi_),
        std::cos(phi_);

    Eigen::Vector3d v(error_phi, error_x, error_y);
    Eigen::Vector3d twist = R * v;

    // twist[0]=wz, twist[1]=vx, twist[2]=vy
    return std::make_tuple(twist(0), twist(1), twist(2));
  }

  std::vector<float> twist2wheels(double wz, double vx, double vy) {
    // H matrix (4×3)
    Eigen::Matrix<double, 4, 3> H;
    H << -l_ - w_, 1, -1, l_ + w_, 1, 1, l_ + w_, 1, -1, -l_ - w_, 1, 1;
    H /= r_;

    Eigen::Vector3d twist(wz, vx, vy);
    Eigen::Matrix<double, 4, 1> u = H * twist;

    // cast each wheel speed to float
    return {static_cast<float>(u(0, 0)), static_cast<float>(u(1, 0)),
            static_cast<float>(u(2, 0)), static_cast<float>(u(3, 0))};
  }

  void wheels2twist(std::vector<float> wheels) {
    // Holonomic drive matrix H_ (4x3): maps wheel velocities [ω, vx, vy] to
    // wheel speeds
    Eigen::Matrix<float, 4, 3> H_;
    H_ << -l_ - w_, 1, -1, l_ + w_, 1, 1, l_ + w_, 1, -1, -l_ - w_, 1, 1;
    // Scale by wheel radius
    H_ /= r_;

    // Wheel speeds vector U (4x1)
    Eigen::Matrix<float, 4, 1> U;
    U << wheels[0], wheels[1], wheels[2], wheels[3];

    // Compute pseudoinverse of H_ via SVD: H_pinv (3x4)
    Eigen::JacobiSVD<Eigen::Matrix<float, 4, 3>> svd(
        H_, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto &S = svd.singularValues();
    Eigen::Matrix<float, 3, 4> S_pinv = Eigen::Matrix<float, 3, 4>::Zero();
    const float tol = 1e-6f;
    for (int i = 0; i < S.size(); ++i) {
      if (S(i) > tol) {
        S_pinv(i, i) = 1.0 / S(i);
      }
    }
    Eigen::Matrix<float, 3, 4> H_pinv =
        svd.matrixV() * S_pinv * svd.matrixU().transpose();

    // Compute wheel velocities: [ω, vx, vy] = H_pinv * U
    Eigen::Matrix<float, 3, 1> wheel_vel = H_pinv * U;

    // Convert to Twist message
    geometry_msgs::msg::Twist twist;
    twist.angular.z = wheel_vel(0);
    twist.linear.x = wheel_vel(1);
    twist.linear.y = wheel_vel(2);

    RCLCPP_INFO(get_logger(),
                "Computed wheel velocities w: %.3f, vx: %.3f, vy: %.3f",
                wheel_vel(0), wheel_vel(1), wheel_vel(2));

    // Publish to /cmd_vel
    pub_->publish(twist);
  }

  void stop() {
    geometry_msgs::msg::Twist twist;
    pub_->publish(twist);
    RCLCPP_INFO(get_logger(), "Stop");
  }

  void select_waypoints() {
    switch (scene_number_) {
    case 1: // Simulation
      motions_ = {
          {0.45, 0.0, 0.0},     // Waypoint 1
          {0.11, -0.35, -0.80}, // Waypoint 2
          {0.0, -1.05, -0.80},  // Waypoint 3
          {0.55, 0.0, 1.607},   // Waypoint 4
          {0.0, 0.65, 1.607},   // Waypoint 5
          {0.40, 0.0, 0.0},     // Waypoint 6
          {0.0, 0.55, 0.0},     // Waypoint 7
          {0.55, 0.0, 0.0},     // Waypoint 8
          {0.0, 0.84, 0.0},     // Waypoint 9
          {-0.55, 0.0, 1.607},  // Waypoint 10
          {0.0, -0.44, 0.0},    // Waypoint 11
          {-0.55, 0.0, 0.0},    // Waypoint 12
          {-0.15, 0.36, -0.80}, // Waypoint 13
          {-0.65, 0.0, 0.80},   // Waypoint 14
          {0.0, 0.0, -3.24}     // Waypoint 16
      };
      break;

    case 2: // CyberWorld
      motions_ = {
          {1.0, 0.0, 0}, {0.0, -0.6, 0.0}, {0.0, 0.5, 0.0}, {-1.0, 0.0, 0.0}};
      break;

    default:
      RCLCPP_ERROR(this->get_logger(), "Invalid Scene Number: %d",
                   scene_number_);
    }
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