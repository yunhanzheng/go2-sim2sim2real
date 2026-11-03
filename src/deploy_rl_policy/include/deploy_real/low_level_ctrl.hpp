#pragma once
#ifndef LOW_LEVEL_CTRL_H
#define LOW_LEVEL_CTRL_H
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/low_state.hpp"
#include "unitree_go/msg/motor_cmd.hpp"
#include "unitree_go/msg/bms_cmd.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "common/motor_crc.h"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/bool.hpp"
#include <chrono>
#include <cstdint>
#include <fstream>

using namespace std;

class LowLevelControl : public rclcpp::Node
{
public:
    LowLevelControl();
    // ~LowLevelControl();

private:
    void init_cmd();
    void state_callback(unitree_go::msg::LowState::SharedPtr msg);
    void joy_callback(sensor_msgs::msg::Joy::SharedPtr msg);
    void target_pos_callback(std_msgs::msg::Float32MultiArray::SharedPtr msg);
    void state_machine();
    // L2+R2 stop ladder (see the comment on hold_ / passive_ below): first
    // press picks HOLD or PASSIVE, later presses only ever reduce authority.
    void enter_stop();
    void enter_passive();
    void hold_tick();
    void passive_tick();
    void state_transform(vector<double> &target_angels);
    void state_obs();
    void run_policy();
    // Publish cmd_msg_ to /lowcmd (after CRC). Single point so every /lowcmd
    // path goes through here.
    void publish_lowcmd();
    static double jointLinearInterpolation(double initPos, double targetPos, double rate);
    rclcpp::TimerBase::SharedPtr timer_;
    // republishes the cached command at keepalive_ms while the policy is
    // engaged — pure watchdog insurance, byte-identical commands, no behavior
    // change. Disabled (keepalive_ms=0) in lockstep sim.
    rclcpp::TimerBase::SharedPtr keepalive_timer_;
    void keepalive_tick();
    rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr cmd_puber_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr target_pos_puber_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr engaged_puber_;
    rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr state_suber_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_suber_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr target_pos_suber_;

    unitree_go::msg::LowCmd cmd_msg_;
    std_msgs::msg::Float32MultiArray pos_data_;
    unitree_go::msg::MotorState motor[12];
    std_msgs::msg::Float32MultiArray rl_target_pos_;

    // Ramp state, owned exclusively by state_transform(): a ramp (re)starts
    // whenever the requested target differs from ramp_target_ or ramp_active_
    // is false (the engaged FSM branch invalidates it while the policy drives
    // the joints). q_init_ is captured once at ramp start; rate_count_ walks
    // 0 -> 100 (~2 s @ 50 Hz) and saturates, so a completed ramp doubles as a
    // steady pose hold. No other code may touch these — scattered resets are
    // what caused the 2026-07 lay-down snap (stale rate_count_ >= 100 made
    // state_transform command the full lay pose in one tick at kp=50).
    bool ramp_active_ = false;
    int rate_count_ = 0;
    vector<double> ramp_target_ = vector<double>(12, 0);
    vector<double> q_init_ = vector<double>(12, 0);
    vector<double> q_des_ = vector<double>(12, 0);

    bool recieved_data_ = false;
    bool have_lowstate_ = false;
    bool is_uncontrolled_ = true;
    bool is_laydown_ = false;
    bool is_standing_ = false;
    bool should_laydown_ = false;
    bool should_stand_ = false;
    bool should_run_policy_ = false;
    bool should_settle_stand_ = false; // dead-man: ramp current pose -> standing (no snap)
    bool momentary_engage_ = false;    // true: policy runs only WHILE L1+R1 held; release -> stand
    // Latched RL-stall fault: set when /rl/target_pos is dead past
    // rl_stall_timeout_s_ while engaged. Blocks re-engagement until BOTH
    // L1+R1 have been seen released, so still-held buttons can't instantly
    // re-engage and cancel the settle ramp the timeout just started.
    bool rl_stall_fault_ = false;
    // L2+R2 stop ladder — two latches, each one strictly weaker than the last,
    // both of which own /lowcmd and keep streaming until the operator presses
    // L2+R2 again. Neither hands the joints back once set.
    //
    // hold_    "stay still". Stop writing cmd_msg_ and republish it byte for
    //          byte: every field (mode/kp/kd/q/tau and the CRC over them) stays
    //          identical to the frame the press interrupted. On a position-
    //          controlled joint that is exactly "keep doing what you were
    //          doing", so a settled robot is motionless — this is the report's
    //          "press it and nothing happens". Only legal when the robot is
    //          settled: freezing a mid-gait target freezes a trot with two feet
    //          in the air, which is how it ends up on its back.
    // passive_ unitree_cpp_deploy's State_Passive: kp -> 0, kd = passive_kd,
    //          dq = tau = 0, q = live measured. Limb goes limp. Second rung from
    //          HOLD, and the only rung for a robot that was already moving.
    //
    // Why not the old rclcpp::shutdown(): the last frame then stayed on the wire
    // with nobody left to overwrite it, so a half-finished standing ramp at
    // kp=50 kept pushing the robot up — the "jerks once, stands up, sometimes
    // on its back" report. A latched state that owns the message cannot do that.
    bool hold_ = false;
    bool passive_ = false;
    // Edge latch for the L2+R2 ladder. The pad repeats /joy at 20 Hz while a
    // trigger is held, so without this one long press would walk the whole
    // ladder (hold -> passive -> exit) in 100 ms. Only a fresh press advances a
    // rung.
    bool stop_btn_held_ = false;
    // Frame + gains caught at the stop press, so the PASSIVE fade starts exactly
    // where the interrupted frame left off (no torque step) and decays to the
    // reference's kp=0 / kd=passive_kd.
    double stop_q0_[12] = {0.0};
    double stop_kp0_[12] = {0.0};
    double stop_kd0_[12] = {0.0};
    rclcpp::Time stop_time_;
    // Per-engage "the actor has produced at least one step" flag: cleared on
    // every fresh disengaged->engaged edge, set by the first validated
    // /rl/target_pos. While it is false the stall timeout must not run — the
    // policy node is gated on its own startup (startup_wait_s / first depth
    // latent) and a 1.5 s timeout there latched the fault, settled the robot
    // and made the sticks look dead (2026-10 real run: 推杆不走).
    bool rl_engaged_steps_ = false;
    bool is_simulation;
    // Torque-aware position clamp in run_policy(): bound |q_cmd - q_meas| to
    // tau_limit_i / policy_kp so the position PD (tau_ff = 0) can never
    // demand torque past the actuator limits (per-joint TAU_LIMIT table in
    // low_level_ctrl.cpp, from resources/go2/go2.xml). Composes with the
    // joint-range clamp; only engages in fault scenarios at normal kp.
    // Off by default (2026-07-24: no torque clamping on real hardware).
    bool torque_clamp_enabled_ = false;
    double policy_kp_;
    double policy_kd_;
    double joy_timeout_s_;          // momentary mode: /joy silence beyond this fires the dead-man (0 disables)
    double rl_stall_timeout_s_;     // engaged: /rl/target_pos dead beyond this -> latched fault + settle (0 disables)
    double passive_kd_;             // L2+R2 Passive state: kp=0 with this kd (reference State_Passive uses 3.0)
    double passive_ramp_s_;         // fade the holding torque out over this long instead of cutting it (0 = instant, the reference's behaviour)
    double stop_tilt_max_;          // rad: |roll| and |pitch| below this count as settled (HOLD is allowed)
    double stop_dq_max_;            // rad/s: max|dq| below this counts as settled
    float imu_rpy_[3] = {0.0f, 0.0f, 0.0f}; // last /lowstate body rpy, for the settled test
    double rl_startup_grace_s_;     // engaged with no step yet: hold the cached pose this long before the timeout may fire (0 disables)
    rclcpp::Time last_rl_msg_time_; // last /rl/target_pos, for the engaged /lowcmd keep-alive
    rclcpp::Time engage_time_;      // start of the current engage, for the startup grace
    rclcpp::Time last_joy_time_;    // last valid /joy, for the joy-loss dead-man

    std::ofstream latency_log_;
    std::uint64_t latest_lowstate_receive_mono_ns_ = 0;
    std::uint64_t last_target_receive_mono_ns_ = 0;
    std::uint64_t last_lowcmd_publish_start_mono_ns_ = 0;
    std::uint64_t last_lowcmd_publish_end_mono_ns_ = 0;
    std::uint32_t latest_lowstate_tick_ = 0;
    std::uint32_t current_target_seq_ = 0;
    std::uint32_t latency_log_rows_ = 0;

    vector<double> kp = vector<double>(12, 0);
    vector<double> kd = vector<double>(12, 0);
    vector<double> standing_angels_ = {-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1, -1.5};
    // Lay pose with hips matched to standing_angels_ so the lay→stand
    // interpolation only moves thigh/calf — no asymmetric hip travel and
    // therefore no hysteretic offset at the standing equilibrium (verified
    // 2026-05-23: the stock Unitree rear hips at ±0.34 left a 0.05-0.08 rad
    // asymmetric proprio offset at engage and the policy collapsed in 3-4 s).
    // Calf targets are clamped to the joint range: the stock -2.80..-2.82
    // values sit past the -2.7227 mechanical stop, so PD ground the calves
    // into the stops at kp=50 forever and the lay pose could never converge
    // (that unreachable error is what originally forced the loose 0.85
    // detection threshold in state_obs). Body settles at base_z ≈ 0.06 m via
    // thigh/calf curl; sphinx-style flat lays remain mechanically impossible.
    vector<double> laydown_angels_ = {
        -0.1,                   // FR_hip — == standing
         1.263495683670044,
        -2.72,                  // FR_calf — was -2.8057, past the -2.7227 stop
         0.1,                   // FL_hip — == standing
         1.248689889907837,
        -2.72,                  // FL_calf — was -2.8048
        -0.1,                   // RR_hip — == standing
         1.2936768531799316,
        -2.72,                  // RR_calf — was -2.8195
         0.1,                   // RL_hip — == standing
         1.2838366031646729,
        -2.72};                 // RL_calf — was -2.8085
};
#endif
