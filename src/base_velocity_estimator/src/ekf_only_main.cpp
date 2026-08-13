// Minimal launcher for EKFVelocityEstimator alone (no KinematicsVelocityEstimator).
// Use this when no Go2 URDF is available, since the kinematics node needs one
// (-p urdf_path:=...) to build its Pinocchio model. Without /obs_velocity the EKF
// still works using its IMU prediction + foot-contact ZUPT (zero-velocity update
// during stance).
//
// Topics consumed: /imu/data (sensor_msgs/Imu), /contact (Float32MultiArray, 4 forces)
// Topics published: /base_velocity (geometry_msgs/Twist, body-frame linear velocity)

#include "base_velocity_estimator/ekf_velocity_estimator.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EKFVelocityEstimator>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
