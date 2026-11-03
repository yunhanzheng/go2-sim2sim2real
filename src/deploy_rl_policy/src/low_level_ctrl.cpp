
#include <deploy_real/low_level_ctrl.hpp>
#include <cmath> // std::isfinite — the joint clamp can't reject NaN (min/max pass it through)
#include <ctime>
double get_diff_norm(vector<double> &vec1, vector<double> &vec2)
{
    assert(vec1.size() == vec2.size());
    double sum = 0.0;
    int n = vec1.size();
    for (int i = 0; i < n; i++)
    {
        sum += pow((vec1[i] - vec2[i]), 2);
    }
    return sqrt(sum);
}

double get_norm(vector<double> &vec)
{
    double sum = 0.0;
    for (double num : vec)
        sum += pow(num, 2);
    return sqrt(sum);
}

namespace
{
std::uint64_t monotonic_ns()
{
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    return static_cast<std::uint64_t>(now.tv_sec) * 1000000000ULL +
           static_cast<std::uint64_t>(now.tv_nsec);
}

// Go2 joint position limits in low-level motor order FR,FL,RR,RL ×
// (hip, thigh, calf), from resources/go2/go2.xml <joint range> (== URDF;
// front and rear thighs differ). Every commanded q — RL targets included —
// is clamped here: a target past a mechanical stop can't be reached, only
// ground into the stop at full PD torque.
constexpr double Q_MIN[12] = {
    -1.0472, -1.5708, -2.7227,  // FR
    -1.0472, -1.5708, -2.7227,  // FL
    -1.0472, -0.5236, -2.7227,  // RR
    -1.0472, -0.5236, -2.7227}; // RL
constexpr double Q_MAX[12] = {
     1.0472,  3.4907, -0.83776,
     1.0472,  3.4907, -0.83776,
     1.0472,  4.5379, -0.83776,
     1.0472,  4.5379, -0.83776};

double clamp_joint_pos(double q, int i)
{
    return std::min(std::max(q, Q_MIN[i]), Q_MAX[i]);
}

// Per-joint actuator torque limits |tau| in low-level motor order
// FR,FL,RR,RL × (hip, thigh, calf), from resources/go2/go2.xml <actuator>:
// the go2 default <motor ctrlrange="-23.7 23.7"/> applies to the hip
// (abduction) and thigh motors; the "knee" class overrides the calves with
// <motor ctrlrange="-35.5 35.5"/>. Used by the torque-aware position clamp
// in run_policy(): |q_cmd - q_meas| <= TAU_LIMIT[i] / policy_kp.
constexpr double TAU_LIMIT[12] = {
    23.7, 23.7, 35.5,  // FR hip, thigh, calf
    23.7, 23.7, 35.5,  // FL
    23.7, 23.7, 35.5,  // RR
    23.7, 23.7, 35.5}; // RL
} // namespace

LowLevelControl::LowLevelControl() : Node("finite_state_machine_node")
{
    this->declare_parameter("is_simulation", true);
    this->get_parameter("is_simulation", is_simulation);
    this->declare_parameter("policy_kp", 30.0);
    this->declare_parameter("policy_kd", 0.6);
    this->get_parameter("policy_kp", policy_kp_);
    this->get_parameter("policy_kd", policy_kd_);
    RCLCPP_INFO(this->get_logger(), "policy gains: kp=%.2f kd=%.2f", policy_kp_, policy_kd_);
    // Torque-aware position clamp (run_policy): bound the commanded PD error
    // so kp*(q_cmd - q_meas) can never demand more than the per-joint
    // actuator torque limit (TAU_LIMIT table above run_policy). Guards the
    // stall/snapback fault class documented 2026-06-19 (‖dq‖ 6.76 snapback
    // after a 0.7 s /lowcmd stall). Default FALSE (2026-07-24 decision: no
    // torque clamping on real hardware); opt in with torque_clamp_enabled:=true.
    this->declare_parameter("torque_clamp_enabled", false);
    this->get_parameter("torque_clamp_enabled", torque_clamp_enabled_);
    RCLCPP_INFO(this->get_logger(), "torque_clamp_enabled=%d (|q_cmd-q_meas| <= tau_limit/kp)",
                int(torque_clamp_enabled_));
    // momentary_engage: when true, the policy is active ONLY while L1+R1 are held;
    // releasing them stops inference and returns to the standing pose (dead-man).
    // Default false keeps the original latch behavior (engage stays on until A=lay).
    this->declare_parameter("momentary_engage", false);
    this->get_parameter("momentary_engage", momentary_engage_);
    RCLCPP_INFO(this->get_logger(), "momentary_engage=%d", int(momentary_engage_));
    // Dead-man on joystick LOSS (momentary mode only): joy_callback can only
    // react to messages that arrive, so a dead transmitter / bluetooth drop /
    // crashed joy node would leave the policy running with no way to stop it.
    // Both joy sources stream continuously (joy_node autorepeats at 20 Hz,
    // keyboard_to_joy publishes at 50 Hz), so /joy silence beyond this timeout
    // is treated exactly like an L1+R1 release. 0.0 disables.
    this->declare_parameter("joy_timeout_s", 0.5);
    this->get_parameter("joy_timeout_s", joy_timeout_s_);
    // Hard bound on the engaged /rl/target_pos keep-alive: the 0.1 s
    // republish below covers transient stalls (Mac overload was seen holding
    // /lowcmd for ~0.7 s on 2026-06-19), but past this timeout the actor is
    // treated as dead — disengage and settle to standing instead of holding a
    // mid-gait pose forever. The fault LATCHES (rl_stall_fault_): re-engaging
    // requires releasing L1+R1 first, so held buttons can't cancel the settle
    // ramp the timeout just started. 0.0 disables (keep-alive only).
    this->declare_parameter("rl_stall_timeout_s", 1.5);
    this->get_parameter("rl_stall_timeout_s", rl_stall_timeout_s_);
    // Startup grace, counted from the engage edge: until the actor has produced
    // its FIRST step, the stall timeout above must not run. The policy node is
    // gated on its own startup (ep_rl_policy --startup_wait_s waiting for the
    // viewer, distill waiting for its first depth latent), and on real hardware
    // the 1.5 s timeout fired inside that window: it latched the fault, settled
    // the robot and left the sticks looking dead (2026-10 real run: 推杆不走).
    // While the grace holds, the cached pose is republished so the wire stays
    // fed; once it expires the normal timeout applies. 0.0 disables (HEAD
    // behaviour: first-second silence is already a fault).
    this->declare_parameter("rl_startup_grace_s", 10.0);
    this->get_parameter("rl_startup_grace_s", rl_startup_grace_s_);
    RCLCPP_INFO(this->get_logger(), "rl stall timeout %.2f s, startup grace %.2f s",
                rl_stall_timeout_s_, rl_startup_grace_s_);
    // L2+R2 stop ladder (2026-10-29, after unitree_cpp_deploy's State_Passive).
    // HOLD is the answer to "press it and the robot stays put": the last frame
    // is republished untouched, so the joints keep executing exactly the target
    // they already had. PASSIVE is the reference's damped state for everything
    // HOLD must not touch (a robot that is already moving).
    this->declare_parameter("passive_kd", 3.0);
    this->get_parameter("passive_kd", passive_kd_);
    // The reference slams kp to 0 in one 1 ms tick. On a standing robot that
    // instantly deletes the holding torque and the body drops. Fading kp over
    // this long turns the same end state into a kneel (0 = reference behaviour).
    this->declare_parameter("passive_ramp_s", 0.30);
    this->get_parameter("passive_ramp_s", passive_ramp_s_);
    // "Settled" gate for HOLD: slow and upright. Anything else gets PASSIVE.
    this->declare_parameter("stop_tilt_max", 0.35); // rad (~20 deg)
    this->get_parameter("stop_tilt_max", stop_tilt_max_);
    this->declare_parameter("stop_dq_max", 0.5); // rad/s
    this->get_parameter("stop_dq_max", stop_dq_max_);
    RCLCPP_INFO(this->get_logger(),
                "L2+R2 -> HOLD if settled (|rpy|<=%.2f rad, max|dq|<%.2f) else PASSIVE "
                "(kp -> 0 over %.2f s, kd=%.2f)",
                stop_tilt_max_, stop_dq_max_, passive_ramp_s_, passive_kd_);
    this->declare_parameter<std::string>("latency_log", "");
    const auto latency_log_path = this->get_parameter("latency_log").as_string();
    if (!latency_log_path.empty())
    {
        latency_log_.open(latency_log_path, std::ios::out | std::ios::trunc);
        if (latency_log_)
        {
            latency_log_ << "target_seq\ttarget_receive_mono_ns\tlowstate_tick\t"
                         << "lowstate_receive_mono_ns\tlowcmd_publish_start_mono_ns\t"
                         << "lowcmd_publish_end_mono_ns\n";
            RCLCPP_INFO(this->get_logger(), "latency timing log: %s", latency_log_path.c_str());
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "cannot open latency timing log: %s",
                         latency_log_path.c_str());
        }
    }
    last_rl_msg_time_ = this->get_clock()->now();
    engage_time_ = last_rl_msg_time_;
    last_joy_time_ = this->get_clock()->now();
    if (is_simulation)
    {
        RCLCPP_INFO(this->get_logger(), "Running in simulation mode.");
        cmd_puber_ = this->create_publisher<unitree_go::msg::LowCmd>("/mujoco/lowcmd", 10);
        state_suber_ = this->create_subscription<unitree_go::msg::LowState>(
            "/mujoco/lowstate", 10, std::bind(&LowLevelControl::state_callback, this, std::placeholders::_1)); // 500HZ
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Running in real mode.");
        cmd_puber_ = this->create_publisher<unitree_go::msg::LowCmd>("/lowcmd", 10);
        state_suber_ = this->create_subscription<unitree_go::msg::LowState>(
            "/lowstate", 10, std::bind(&LowLevelControl::state_callback, this, std::placeholders::_1)); // 500HZ
    }
    target_pos_puber_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/pos", 10);
    // FSM gate consumed by rl_distill_policy / depth_encoder_node: true only
    // once we have standing + LB+RB engage, so the RL inference doesn't run
    // (and the yaw arrow doesn't update) while the robot is still on the ground.
    engaged_puber_ = this->create_publisher<std_msgs::msg::Bool>("/policy_engaged", 10);
    init_cmd();
    target_pos_suber_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/rl/target_pos", 10, std::bind(&LowLevelControl::target_pos_callback, this, std::placeholders::_1)); // 50 HZ
    joy_suber_ = this->create_subscription<sensor_msgs::msg::Joy>("/joy", 10, bind(&LowLevelControl::joy_callback, this, placeholders::_1));
    timer_ = this->create_wall_timer(std::chrono::milliseconds(20), std::bind(&LowLevelControl::state_machine, this)); // 50hz
    // 5 ms keep-alive republish while engaged (see header comment). Runs on the
    // same single-threaded executor as every callback, so it never races the
    // fresh-command path; between RL messages it re-sends the identical bytes.
    // keepalive_ms=0 disables it — REQUIRED in lockstep sim, where the
    // simulator advances 4 substeps per received /lowcmd and the extra
    // republishes would run sim time ~5x faster than the 50 Hz policy tick.
    this->declare_parameter<int>("keepalive_ms", 5);
    const int keepalive_ms = this->get_parameter("keepalive_ms").as_int();
    if (keepalive_ms > 0)
        keepalive_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(keepalive_ms), std::bind(&LowLevelControl::keepalive_tick, this));
    else
        RCLCPP_INFO(this->get_logger(), "keepalive timer disabled (keepalive_ms=0)");
}

void LowLevelControl::init_cmd()
{
    kp = vector<double>(12, 50.0);
    kd = vector<double>(12, 1);

    for (int i = 0; i < 20; i++)
    {
        cmd_msg_.motor_cmd[i].mode = 0x01; // Set toque mode, 0x00 is passive mode
        cmd_msg_.motor_cmd[i].q = PosStopF;
        cmd_msg_.motor_cmd[i].kp = 0;
        cmd_msg_.motor_cmd[i].dq = VelStopF;
        cmd_msg_.motor_cmd[i].kd = 0;
        cmd_msg_.motor_cmd[i].tau = 0;
    }
    pos_data_ = std_msgs::msg::Float32MultiArray();
}

void LowLevelControl::keepalive_tick()
{
    // While the policy is engaged and healthy, and unconditionally while a stop
    // owns the joints: pre-engage the FSM states own their own publication
    // cadence, and a latched stall fault must keep its settle-to-stand
    // ownership. HOLD and PASSIVE are the exception — they must stay on the
    // wire at full rate until the operator releases them, otherwise the motors
    // fall back to their own command watchdog (the whole reason the stop states
    // own the message instead of shutting the process down).
    if (hold_ or passive_ or (is_standing_ and should_run_policy_ and not rl_stall_fault_))
        publish_lowcmd();
}

void LowLevelControl::publish_lowcmd()
{
    // CRC then publish (single point so every /lowcmd path goes through here).
    last_lowcmd_publish_start_mono_ns_ = monotonic_ns();
    get_crc(cmd_msg_);
    cmd_puber_->publish(cmd_msg_);
    last_lowcmd_publish_end_mono_ns_ = monotonic_ns();
}

void LowLevelControl::state_callback(unitree_go::msg::LowState::SharedPtr msg)
{
    latest_lowstate_receive_mono_ns_ = monotonic_ns();
    latest_lowstate_tick_ = msg->tick;
    for (int i = 0; i < 12; i++)
    {
        motor[i] = msg->motor_state[i];
    }
    for (int i = 0; i < 3; i++)
        imu_rpy_[i] = msg->imu_state.rpy[i];
    have_lowstate_ = true;
}

void LowLevelControl::target_pos_callback(std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    const auto target_receive_ns = monotonic_ns();
    if (msg->data.size() < 12)
    {
        // Indexing data[0..11] on a shorter vector is UB, not an error — drop it.
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                              "/rl/target_pos has %zu values (need 12) — ignoring",
                              msg->data.size());
        return;
    }
    // Reject the WHOLE message on any non-finite value: clamp_joint_pos
    // (std::min/std::max) passes NaN through, so one NaN here would become a
    // motor target and then be held by the engaged keep-alive. A NaN anywhere
    // means the actor state is corrupt — keep the last good command instead.
    // last_rl_msg_time_ is only refreshed AFTER validation, so a stream of
    // NaNs counts as a stall and eventually trips rl_stall_timeout_s.
    for (int i = 0; i < 12; i++)
    {
        if (!std::isfinite(msg->data[i]))
        {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                  "/rl/target_pos[%d] non-finite — dropping message", i);
            return;
        }
    }
    recieved_data_ = true;
    rl_engaged_steps_ = true;
    last_rl_msg_time_ = this->get_clock()->now();
    last_target_receive_mono_ns_ = target_receive_ns;
    current_target_seq_ = msg->layout.data_offset;
    rl_target_pos_.data = msg->data;
    if (is_standing_ and should_run_policy_)
        run_policy();
}
void LowLevelControl::joy_callback(sensor_msgs::msg::Joy::SharedPtr msg)
{
    // Guard the fixed indices below (buttons[5], axes[5]): a pad with a
    // different mapping or a malformed publisher gives a shorter vector, and
    // operator[] past the end is UB, not an error.
    if (msg->buttons.size() < 6 || msg->axes.size() < 6)
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "/joy too short (%zu buttons / %zu axes, need 6/6) — ignoring",
                             msg->buttons.size(), msg->axes.size());
        return;
    }
    last_joy_time_ = this->get_clock()->now();

    // Clear a latched RL-stall fault only once BOTH engage buttons are seen
    // released — a held L1+R1 must not re-engage straight out of the fault.
    if (rl_stall_fault_ and not (msg->buttons[4] and msg->buttons[5]))
        rl_stall_fault_ = false;

    // L2+R2 stop ladder, handled FIRST so no other button can dirty the FSM
    // flags while a stop is latched. Threshold, not ==-1.0 — analog triggers on
    // some pads never report exactly -1.0 when floored. One rung per PRESS, not
    // per message: /joy autorepeats at 20 Hz while the trigger is held, and
    // acting on every message would run the whole ladder (hold -> passive ->
    // exit) inside 100 ms of one press. See the comment on hold_/passive_ in the
    // header for why the stream keeps running instead of the process exiting.
    const bool stop_pressed = msg->axes[2] < -0.9 && msg->axes[5] < -0.9;
    if (stop_pressed and not stop_btn_held_)
    {
        stop_btn_held_ = true;
        if (hold_)
        {
            // Second press from HOLD: same message, but now the limb is meant
            // to let go. Third press then exits.
            enter_passive();
        }
        else if (passive_)
        {
            RCLCPP_WARN(this->get_logger(), "L2+R2 again — exiting (robot is already passive)");
            rclcpp::shutdown();
        }
        else
        {
            enter_stop();
        }
        return;
    }
    if (not stop_pressed)
        stop_btn_held_ = false;
    // A latched stop ignores every other button until L2+R2 is pressed again.
    if (hold_ or passive_)
        return;

    if (is_laydown_ and msg->buttons[1]) // Button B
    {
        should_stand_ = true;
        should_laydown_ = false;
        should_run_policy_ = false;
        should_settle_stand_ = false;
    }
    else if (is_standing_ and msg->buttons[0]) // Button A
    {
        should_laydown_ = true;
        should_stand_ = false;
        should_run_policy_ = false;
        should_settle_stand_ = false;
    }
    else if (is_standing_ and msg->buttons[4] and msg->buttons[5]
             and not rl_stall_fault_) // Button LB and Button RB (blocked while stall fault latched)
    {
        // Reset the stall clock ONLY on a fresh disengaged->engaged edge —
        // this branch runs on every /joy message while the buttons are held,
        // and refreshing the timestamp continuously would defeat the timeout.
        if (not should_run_policy_)
        {
            last_rl_msg_time_ = this->get_clock()->now();
            engage_time_ = last_rl_msg_time_;
            rl_engaged_steps_ = false;
        }
        should_laydown_ = false;
        should_stand_ = false;
        should_run_policy_ = true;
        should_settle_stand_ = false;   // re-engage cancels an in-progress settle ramp
    }
    else if (momentary_engage_ and should_run_policy_
             and not (msg->buttons[4] and msg->buttons[5])) // L1+R1 released while engaged
    {
        // Dead-man: stop the policy and settle back to standing through the
        // should_settle_stand_ FSM branch instead of snapping. The engaged
        // branch invalidated the ramp, so state_transform starts a fresh one:
        // it captures q_init_ from wherever the gait left the legs and
        // interpolates to standing_angels_ over ~2 s at kp=50/kd=1 — the same
        // smooth profile as lay<->stand.
        // (Guard on the buttons being released so an is_standing_ flicker mid-gait
        // can't drop the policy while they're still held.)
        should_run_policy_ = false;
        should_stand_ = false;
        should_laydown_ = false;
        should_settle_stand_ = true;
    }
}

void LowLevelControl::enter_stop()
{
    // The whole stop decision in one place. Both rungs own the message; the
    // choice is only about whether the operator gets "nothing happens" or "the
    // limb lets go".
    //
    // HOLD is only legal once the robot is already still and upright: freezing
    // the frame is freezing a *target*, and a target captured mid-trot has two
    // feet in the air, so holding it is what tips the robot over. A moving robot
    // therefore goes straight to PASSIVE, where the reference puts it too.
    double max_dq = 0.0;
    if (have_lowstate_)
        for (int i = 0; i < 12; i++)
            max_dq = std::max(max_dq, std::fabs(double(motor[i].dq)));
    const bool tilted = std::fabs(double(imu_rpy_[0])) > stop_tilt_max_ or
                        std::fabs(double(imu_rpy_[1])) > stop_tilt_max_;
    const bool settled = have_lowstate_ and not tilted and max_dq < stop_dq_max_;

    // Whichever rung runs, the policy and every pending FSM transition are off:
    // nothing may write cmd_msg_ behind the stop's back.
    should_run_policy_ = false;
    should_stand_ = false;
    should_laydown_ = false;
    should_settle_stand_ = false;

    if (settled)
    {
        hold_ = true;
        RCLCPP_WARN(this->get_logger(),
                    "L2+R2 — HOLD (settled: max|dq| %.2f < %.2f, |roll| %.2f |pitch| %.2f). "
                    "The last frame is republished unchanged, so the joints keep executing the "
                    "target they already had and the robot does not move. Press L2+R2 again to "
                    "go passive, once more to exit.",
                    max_dq, stop_dq_max_, double(imu_rpy_[0]), double(imu_rpy_[1]));
        hold_tick();
    }
    else
    {
        RCLCPP_WARN(this->get_logger(),
                    "L2+R2 — not settled (max|dq| %.2f, |roll| %.2f |pitch| %.2f) -> PASSIVE: "
                    "freezing a moving target would strand a foot in the air, so the limb lets go "
                    "instead. Press L2+R2 again to exit.",
                    max_dq, double(imu_rpy_[0]), double(imu_rpy_[1]));
        enter_passive();
    }
}

void LowLevelControl::enter_passive()
{
    hold_ = false;
    passive_ = true;
    should_run_policy_ = false;
    should_stand_ = false;
    should_laydown_ = false;
    should_settle_stand_ = false;
    // Snapshot the frame the motors were executing this instant. The fade below
    // starts from exactly these gains, so the first passive frame is torque-
    // identical to the last active one.
    for (int i = 0; i < 12; i++)
    {
        stop_q0_[i] = double(cmd_msg_.motor_cmd[i].q);
        stop_kp0_[i] = double(cmd_msg_.motor_cmd[i].kp);
        stop_kd0_[i] = double(cmd_msg_.motor_cmd[i].kd);
    }
    stop_time_ = this->get_clock()->now();
    RCLCPP_WARN(this->get_logger(),
                "L2+R2 — PASSIVE: kp %.1f -> 0 over %.2f s, kd -> %.2f, q = measured. The "
                "stream keeps running so no stale target is left for the motors; press L2+R2 "
                "again to exit.",
                stop_kp0_[0], passive_ramp_s_, passive_kd_);
    passive_tick(); // apply now, not on the next 20 ms FSM tick
}

void LowLevelControl::hold_tick()
{
    // The entire HOLD state: republish cmd_msg_ untouched. Every field — mode,
    // kp, kd, q, tau and the CRC computed over them — is byte-identical to the
    // frame the press interrupted, so from the motors' side nothing changed at
    // all. That is what "keep still" means on a position-controlled joint: the
    // same position error keeps producing the same holding torque.
    publish_lowcmd();
}

void LowLevelControl::passive_tick()
{
    const double elapsed = (this->get_clock()->now() - stop_time_).seconds();
    const double a = (passive_ramp_s_ > 0.0)
                         ? std::min(1.0, std::max(0.0, elapsed / passive_ramp_s_))
                         : 1.0;
    // Torque = kp*(q_cmd - q_meas) + kd*(0 - dq). Both terms fade along `a`, so
    // the holding torque decays instead of disappearing in one tick. At a == 1
    // this is unitree_cpp_deploy's State_Passive exactly: kp 0, kd passive_kd,
    // dq = tau = 0, q = live measured — but reached continuously, which is why
    // it does not read as "it twitched once".
    for (int i = 0; i < 12; i++)
    {
        cmd_msg_.motor_cmd[i].mode = 0x01; // Set toque mode, 0x00 is passive mode
        cmd_msg_.motor_cmd[i].kp = stop_kp0_[i] * (1.0 - a);
        cmd_msg_.motor_cmd[i].kd = stop_kd0_[i] + (passive_kd_ - stop_kd0_[i]) * a;
        // While kp is still non-zero the target must stay ON the frozen pose —
        // that is what is holding the robot up. Once kp has reached 0 the q
        // field is inert, so hand it to the live measurement and leave no stale
        // pose anywhere in the message.
        cmd_msg_.motor_cmd[i].q =
            (a >= 1.0 and have_lowstate_) ? double(motor[i].q) : stop_q0_[i];
        cmd_msg_.motor_cmd[i].dq = 0;
        cmd_msg_.motor_cmd[i].tau = 0;
    }
    publish_lowcmd();
}

void LowLevelControl::state_machine()
{
    // A latched stop owns everything: it publishes its own command and never
    // hands the joints back, so nothing below (policy, ramps, pose holds) may
    // run. Checked BEFORE the have_lowstate_ gate so L2+R2 pressed before the
    // first /lowstate still does something sensible.
    if (hold_ or passive_)
    {
        // This early return skips the /policy_engaged publish below, which would
        // otherwise leave the RL stack inferring against a stale `true`.
        std_msgs::msg::Bool engaged_msg;
        engaged_msg.data = false;
        engaged_puber_->publish(engaged_msg);
        if (hold_)
            hold_tick();
        else
            passive_tick();
        return;
    }
    // Wait for the first /lowstate (real) or /mujoco/lowstate (sim) before
    // running any FSM logic. Without this guard state_transform records
    // q_init_ from default-zero motor[i].q, then interpolates 0 → lay_angels
    // while the actual joints are already at lay — PD then yanks them around
    // the workspace and the body slides metres before the sim catches up.
    if (!have_lowstate_) return;
    state_obs();
    // Broadcast FSM engagement so the RL stack can gate its own inference.
    std_msgs::msg::Bool engaged_msg;
    engaged_msg.data = is_standing_ && should_run_policy_;
    engaged_puber_->publish(engaged_msg);
    // Dead-man on joystick loss: if /joy itself goes silent while engaged
    // (transmitter off, bluetooth drop, joy node crash), joy_callback never
    // fires again — so treat prolonged silence like an L1+R1 release.
    if (momentary_engage_ and should_run_policy_ and joy_timeout_s_ > 0.0 and
        (this->get_clock()->now() - last_joy_time_).seconds() > joy_timeout_s_)
    {
        RCLCPP_WARN(this->get_logger(),
                    "/joy silent > %.2f s while engaged — dead-man settle to standing",
                    joy_timeout_s_);
        should_run_policy_ = false;
        should_stand_ = false;
        should_laydown_ = false;
        should_settle_stand_ = true;
    }
    // NB: the per-tick `cout << ... << endl` FSM-branch prints were removed — they
    // flushed synchronously at 200 Hz (jitter + log bloat). The FSM state is already
    // reported every 250 ticks by state_obs() (uncontrolled/laydown/standing).
    if (is_uncontrolled_ and !is_laydown_) // init pos: lay down
    {
        state_transform(laydown_angels_);
    }
    else if (is_laydown_ and should_stand_) // from lay down state to stand up state
    {
        state_transform(standing_angels_);
    }
    else if (is_standing_ and should_laydown_) // from stand up state to lay down state
    {
        state_transform(laydown_angels_);
    }
    else if (is_standing_ and should_settle_stand_) // dead-man: ramp back to standing
    {
        // state_transform starts the ramp itself (target change / invalidated
        // ramp) and saturates at rate 1.0; once the full 2 s profile has run,
        // clear the flag so the hold branch takes over at the same kp=50/kd=1.
        state_transform(standing_angels_);
        if (rate_count_ >= 100)
            should_settle_stand_ = false;
    }
    else if (is_standing_ and should_run_policy_) // policy engaged
    {
        // The policy owns the joints now: invalidate the ramp so whatever
        // transition runs next recaptures q_init_ from wherever the gait left
        // the legs, instead of inheriting a stale ramp.
        ramp_active_ = false;
        // Stall handling: if the RL node stalls or dies mid-run, cmd_msg_
        // stops being refreshed. Up to rl_stall_timeout_s that is a benign
        // position hold (the continuous republish below keeps the watchdog
        // fed); past it the actor is dead, and holding a mid-gait pose
        // forever is worse than settling. Latch the fault and ramp back to
        // standing; re-engage requires an L1+R1 release first. Detection
        // keys off last_rl_msg_time_ (message ARRIVAL), never publication,
        // so the per-tick republish below cannot mask a stall.
        double stall_s = (this->get_clock()->now() - last_rl_msg_time_).seconds();
        const double engaged_s = (this->get_clock()->now() - engage_time_).seconds();
        if (not rl_engaged_steps_ and engaged_s <= rl_startup_grace_s_)
        {
            // Not a single step since this engage: that is the NORMAL state
            // while the policy node sits behind its own startup gate. Keep the
            // wire fed with the cached pose (the standing hold built before
            // engage) and stay silent-ish — calling this "stalled" sent the
            // operator hunting for a blocking bug that wasn't there.
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                 "engaged, waiting for the actor's first step (%.1f s of %.1f s "
                                 "grace; the policy node is likely gated on startup_wait_s / the "
                                 "depth latent) — holding the cached pose",
                                 engaged_s, rl_startup_grace_s_);
            publish_lowcmd();
        }
        else if (rl_stall_timeout_s_ > 0.0 and stall_s > rl_stall_timeout_s_)
        {
            RCLCPP_ERROR(this->get_logger(),
                         "/rl/target_pos dead > %.1f s while engaged — latching stall "
                         "fault, settling to standing (release L1+R1 to clear)",
                         rl_stall_timeout_s_);
            rl_stall_fault_ = true;
            should_run_policy_ = false;
            should_stand_ = false;
            should_laydown_ = false;
            should_settle_stand_ = true;
        }
        else
        {
            if (!recieved_data_ or stall_s > 0.1)
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                     "engaged but /rl/target_pos stalled — republishing last cmd");
            // Continuous engaged publication (was: republish only after a
            // detected stall). Unitree's own low-level examples write /lowcmd
            // at a fixed high rate regardless of upstream; previously the
            // watchdog margin while engaged depended entirely on the Python
            // RL node's scheduling (Mac overload held /lowcmd for ~0.7 s on
            // 2026-06-19). Now this branch republishes the cached cmd_msg_
            // EVERY FSM tick, so any gap between RL messages is bridged
            // within one timer period no matter what Python does.
            // Design: dual-owner, callback-first. target_pos_callback still
            // builds + publishes the command immediately on arrival (zero
            // added latency for fresh RL actions — funnelling all publishing
            // through this timer would add up to one timer period of latency
            // to every action, the higher-risk option for a 50 Hz control
            // loop), and this tick republishes the identical cached message
            // in between. No race: the node spins on the default
            // single-threaded executor, so this timer and the callbacks
            // never run concurrently, and an occasional duplicate of an
            // identical position command inside one tick is harmless.
            // Contents are byte-identical to what run_policy last built (or
            // the pre-engage standing hold if no RL data has arrived yet —
            // exactly what the old keep-alive republished in that case).
            publish_lowcmd();
        }
    }
    else
    {
        // Hold the pose of the current state. Routed through state_transform
        // rather than commanding the target directly: if a state flip lands
        // here mid-ramp (state_obs detects lay/stand before the ramp finishes)
        // the unchanged target keeps ramping to completion instead of snapping
        // the remainder, and once rate saturates at 1.0 this IS a plain hold.
        if (is_standing_)
            state_transform(standing_angels_);
        else
            state_transform(laydown_angels_);
    }
}
void LowLevelControl::run_policy()
{
    // Only called from target_pos_callback after the size guard, so
    // rl_target_pos_.data[0..11] is valid. The engaged-branch keep-alive in
    // state_machine covers the no-data / stalled-RL case.
    vector<float> vec;
    for (int i = 0; i < 12; i++)
    {
        double q = clamp_joint_pos(rl_target_pos_.data[i], i);
        // Torque-aware clamp, composed ON TOP of the joint-range clamp above
        // (range first, then torque — this clamp only ever pulls the target
        // TOWARD the measured joint position, so it cannot push a range-legal
        // target back out of range). The engaged command is pure position PD
        // (tau_ff = 0, dq_des = 0), so the implied torque is
        // kp*(q_cmd - q_meas) - kd*dq_meas; nothing on the host bounded the
        // kp term, and after a stall/snapback a large error can demand torque
        // far beyond the actuator (2026-06-19 incident: /lowcmd stalled
        // ~0.7 s → watchdog sag → PD snapback with ‖dq‖ 6.76). Bound it:
        // |q_cmd - q_meas| <= TAU_LIMIT[i] / policy_kp. With the typical
        // kp=35 the clamp only engages in fault scenarios (23.7 Nm / 35 ≈
        // 0.68 rad error; normal gait tracking error is far smaller), so the
        // gait is unaffected. motor[i].q is valid here: run_policy is only
        // reachable once is_standing_, which requires have_lowstate_.
        if (torque_clamp_enabled_ and policy_kp_ > 0.0)
        {
            const double q_meas = motor[i].q;
            const double max_err = TAU_LIMIT[i] / policy_kp_;
            if (q > q_meas + max_err or q < q_meas - max_err)
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                     "torque clamp engaged: joint %d target %.3f is %+.3f rad "
                                     "from measured %.3f, > %.3f rad (%.1f Nm / kp %.1f) — clamping",
                                     i, q, q - q_meas, q_meas, max_err, TAU_LIMIT[i], policy_kp_);
                q = std::min(std::max(q, q_meas - max_err), q_meas + max_err);
            }
        }
        cmd_msg_.motor_cmd[i].mode = 0x01; // Set toque mode, 0x00 is passive mode
        cmd_msg_.motor_cmd[i].q = q;
        cmd_msg_.motor_cmd[i].kp = policy_kp_;
        cmd_msg_.motor_cmd[i].kd = policy_kd_;
        cmd_msg_.motor_cmd[i].dq = 0;
        cmd_msg_.motor_cmd[i].tau = 0;
        vec.push_back(q);
    }
    publish_lowcmd();
    if (latency_log_)
    {
        latency_log_ << current_target_seq_ << '\t'
                     << last_target_receive_mono_ns_ << '\t'
                     << latest_lowstate_tick_ << '\t'
                     << latest_lowstate_receive_mono_ns_ << '\t'
                     << last_lowcmd_publish_start_mono_ns_ << '\t'
                     << last_lowcmd_publish_end_mono_ns_ << '\n';
        if (++latency_log_rows_ % 50 == 0)
            latency_log_.flush();
    }
    pos_data_.data = vec;
    target_pos_puber_->publish(pos_data_);
}

void LowLevelControl::state_obs()
{
    vector<double> q(12, 0);
    vector<double> dq(12, 0);

    for (int i = 0; i < 12; i++)
    {
        q[i] = motor[i].q;
        dq[i] = motor[i].dq;
    }
    static int _dbg_tick = 0;
    if ((++_dbg_tick) % 250 == 0) {
        RCLCPP_INFO(this->get_logger(),
                    "state_obs: lay_diff=%.3f stand_diff=%.3f dq=%.3f  "
                    "uncontrolled=%d laydown=%d standing=%d",
                    get_diff_norm(q, laydown_angels_),
                    get_diff_norm(q, standing_angels_),
                    get_norm(dq),
                    int(is_uncontrolled_), int(is_laydown_), int(is_standing_));
    }
    // Threshold 0.85 (loosened from the original 0.25): gravity-induced
    // steady-state PD error at kp=50/kd=1 leaves the joint-space RSS at
    // ~0.5-0.7 even when the robot is fully settled in either pose (the
    // gravity-pinned hips dominate; the once-unreachable calf lay targets are
    // fixed to -2.72 and no longer contribute). dq gates widened to 0.5 / 0.3
    // for the same reason — small contact jitter keeps it at 0.15-0.30 at rest.
    if (get_diff_norm(q, laydown_angels_) < 0.85 && get_norm(dq) < 0.5)
    {
        is_laydown_ = true;
        is_uncontrolled_ = false;
        is_standing_ = false;
        // Intent is consumed on arrival: a should_* flag left latched after
        // its transition completes would re-fire the transition the moment
        // the opposite state is ever detected again — e.g. a stale
        // should_stand_ from an old B-press would make a robot that is
        // knocked over (landing near the lay pose) stand back up on its own.
        should_laydown_ = false;
    }
    else if (get_diff_norm(q, standing_angels_) < 0.85 && get_norm(dq) < 0.3)
    {
        is_standing_ = true;
        is_laydown_ = false;
        is_uncontrolled_ = false;
        should_stand_ = false; // consumed on arrival (see above)
    }
}
void LowLevelControl::state_transform(vector<double> &target_angels)
{
    // The ramp lifecycle lives HERE and only here: (re)start whenever the
    // requested target changes or no ramp is active (the policy was driving
    // the joints). This guarantees every transition interpolates from the
    // CURRENT pose — counters from a previous ramp can never leak into a new
    // one. (The old scattered resets in state_obs/joy_callback missed the
    // post-settle A-press path: rate_count_ arrived at >= 100 and the very
    // first tick commanded the full lay pose at kp=50 with no interpolation.)
    if (!ramp_active_ or ramp_target_ != target_angels)
    {
        ramp_active_ = true;
        ramp_target_ = target_angels;
        rate_count_ = 0;
        for (int i = 0; i < 12; i++)
            q_init_[i] = motor[i].q; // valid: state_machine gates on have_lowstate_
    }
    else if (rate_count_ < 100)
    {
        rate_count_++;
    }
    // rate ramps 0 -> 1 over 100 ticks ≈ 2 s @ 50 Hz, then saturates (hold).
    // The first tick publishes rate 0 == the captured pose, so /lowcmd never
    // pauses at a ramp start (the old 5-tick capture window left an 80 ms gap).
    double rate = rate_count_ / 100.0;
    vector<float> vec;
    for (int i = 0; i < 12; i++)
    {
        q_des_[i] = jointLinearInterpolation(q_init_[i], target_angels[i], rate);
        q_des_[i] = clamp_joint_pos(q_des_[i], i);
        cmd_msg_.motor_cmd[i].mode = 0x01; // Set toque mode, 0x00 is passive mode
        cmd_msg_.motor_cmd[i].q = q_des_[i];
        cmd_msg_.motor_cmd[i].kp = kp[i];
        cmd_msg_.motor_cmd[i].dq = 0;
        cmd_msg_.motor_cmd[i].kd = kd[i];
        cmd_msg_.motor_cmd[i].tau = 0;
        vec.push_back(q_des_[i]);
    }

    publish_lowcmd();
    pos_data_.data = vec;
    target_pos_puber_->publish(pos_data_);
}

double LowLevelControl::jointLinearInterpolation(double initPos, double targetPos, double rate)
{
    double p;
    rate = std::min(std::max(rate, 0.0), 1.0);
    p = initPos * (1 - rate) + targetPos * rate;
    return p;
}

// LowLevelControl::~LowLevelControl()
// {
// }

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);                        // Initialize rclcpp
    auto node = std::make_shared<LowLevelControl>(); // Create a ROS2 node and make share with low_level_cmd_sender class

    rclcpp::spin(node); // Run ROS2 node
    rclcpp::shutdown(); // Exit
    return 0;
}
