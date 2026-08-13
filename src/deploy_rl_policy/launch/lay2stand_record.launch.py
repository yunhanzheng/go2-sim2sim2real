"""Headless ROS run that captures the full lay-down → stand-up → RL policy
sequence to MP4.

Stack: mujoco_simulator (init_standing=false, headless) + low_level_ctrl
(lay/stand/policy FSM) + depth_sim_node + depth_encoder_node +
rl_distill_policy + joy_sequencer (auto-presses B then LB+RB) +
distill_recorder. No joy_node, no X11.

Phases (timed from sequencer start):
  [0, wait_init)            : idle — let nodes wake up
  [wait_init, +press)       : hold B → triggers low_level_ctrl lay→stand
  [+press, +settle)         : idle — wait for stand-up motion to settle
  [+settle, end)            : hold LB+RB, axes[LY]=speed → policy engaged

Total recorder duration must cover all phases (default wait_init=2,
press=1, settle=4, engage=10 → 17 s).

Usage:
  ros2 launch deploy_rl_policy lay2stand_record.launch.py \\
      ckpt:=/abs/path/to/model.pt \\
      scene_xml:=/abs/path/to/scene.xml \\
      output:=/abs/path/to/out.mp4 \\
      lin_vel_max:=1.5 speed:=1.0 engage_duration:=10.0
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue

# install/<pkg>/share/<pkg> -> 3 up -> the repo (colcon workspace) root.
_WS_ROOT = os.path.normpath(
    os.path.join(get_package_share_directory("deploy_rl_policy"), "..", "..", "..")
)
# The training repo supplies the policy classes (Actor / Estimator /
# RecurrentDepthBackbone): https://github.com/Roman-089/go2-flyball-parkour
# Override with $EP_ROOT or the ep_root: launch argument.
_EP_ROOT = os.environ.get(
    "EP_ROOT", os.path.join(os.path.dirname(_WS_ROOT), "go2-flyball-parkour")
)
_DEFAULT_CKPT = os.path.join(
    _WS_ROOT, "models", "L00-02-flat-baseline-delay-distillation-noised", "model_14500.pt"
)


def generate_launch_description():
    ckpt = LaunchConfiguration("ckpt")
    ep_root = LaunchConfiguration("ep_root")
    scene_xml = LaunchConfiguration("scene_xml")
    output = LaunchConfiguration("output")
    lin_vel_max = LaunchConfiguration("lin_vel_max")
    speed = LaunchConfiguration("speed")
    vel_profile = LaunchConfiguration("vel_profile")
    action_delay = LaunchConfiguration("action_delay")
    policy_kp = LaunchConfiguration("policy_kp")
    policy_kd = LaunchConfiguration("policy_kd")
    fps = LaunchConfiguration("fps")
    width = LaunchConfiguration("width")
    height = LaunchConfiguration("height")
    camera = LaunchConfiguration("camera")
    title = LaunchConfiguration("title")
    metrics_only = LaunchConfiguration("metrics_only")
    clean_frame = LaunchConfiguration("clean_frame")
    depth_mode = LaunchConfiguration("depth_mode")
    depth_noise_scale = LaunchConfiguration("depth_noise_scale")
    depth_cam_roll_deg = LaunchConfiguration("depth_cam_roll_deg")
    depth_stereo = LaunchConfiguration("depth_stereo")
    depth_seed = LaunchConfiguration("depth_seed")
    depth_rate_hz = LaunchConfiguration("depth_rate_hz")
    depth_sim_gate_on_engage = LaunchConfiguration("depth_sim_gate_on_engage")
    depth_latency_s = LaunchConfiguration("depth_latency_s")
    policy_rate_hz = LaunchConfiguration("policy_rate_hz")
    friction_scale = LaunchConfiguration("friction_scale")
    initial_x = LaunchConfiguration("initial_x")
    wait_init = LaunchConfiguration("wait_init")
    press = LaunchConfiguration("press")
    settle = LaunchConfiguration("settle")
    engage_duration = LaunchConfiguration("engage_duration")
    slowdown = LaunchConfiguration("slowdown")
    laydown = LaunchConfiguration("laydown")
    tail = LaunchConfiguration("tail")
    proprio_log = LaunchConfiguration("proprio_log")
    proprio_log_ticks = LaunchConfiguration("proprio_log_ticks")
    viewer = LaunchConfiguration("viewer")
    use_sim_truth_vel = LaunchConfiguration("use_sim_truth_vel")
    zero_student_velocity_obs = LaunchConfiguration("zero_student_velocity_obs")
    vel_log = LaunchConfiguration("vel_log")
    vel_log_ticks = LaunchConfiguration("vel_log_ticks")
    run_ekf = LaunchConfiguration("run_ekf")
    zero_yaw_pred = LaunchConfiguration("zero_yaw_pred")
    use_true_yaw = LaunchConfiguration("use_true_yaw")
    env_class_eq_17 = LaunchConfiguration("env_class_eq_17")
    contact_threshold = LaunchConfiguration("contact_threshold")
    proprio_latency_s = LaunchConfiguration("proprio_latency_s")
    lift_test = LaunchConfiguration("lift_test")
    lockstep = LaunchConfiguration("lockstep")
    lift_height = LaunchConfiguration("lift_height")
    lift_stand_hold_s = LaunchConfiguration("lift_stand_hold_s")
    lift_up_s = LaunchConfiguration("lift_up_s")
    lift_down_s = LaunchConfiguration("lift_down_s")
    lift_ramp_s = LaunchConfiguration("lift_ramp_s")
    snap_qpos_on_engage = LaunchConfiguration("snap_qpos_on_engage")
    depth_gate_on_engage = LaunchConfiguration("depth_gate_on_engage")
    depth_reset_on_engage = LaunchConfiguration("depth_reset_on_engage")
    encoder_use_shared_contact = LaunchConfiguration("encoder_use_shared_contact")
    publish_raw_last_action = LaunchConfiguration("publish_raw_last_action")
    clip_applied_action = LaunchConfiguration("clip_applied_action")
    encoder_trace_log = LaunchConfiguration("encoder_trace_log")
    action_log = LaunchConfiguration("action_log")
    latency_encoder_log = LaunchConfiguration("latency_encoder_log")
    latency_actor_log = LaunchConfiguration("latency_actor_log")
    latency_low_level_log = LaunchConfiguration("latency_low_level_log")
    fused = LaunchConfiguration("fused")

    decls = [
        DeclareLaunchArgument("ckpt", default_value=_DEFAULT_CKPT),
        DeclareLaunchArgument("ep_root", default_value=_EP_ROOT),
        DeclareLaunchArgument(
            "scene_xml",
            # scene_2hurdles.xml has a 2.5 cm noise hfield that the distill
            # student wasn't trained on; on that scene the policy locks into
            # a static crouch. The "smooth" variant drops the hfield and the
            # same student honestly clears the obstacles.
            default_value=os.path.join(
                _WS_ROOT, "resources", "go2", "scene_2hurdles_smooth.xml"
            ),
        ),
        DeclareLaunchArgument(
            "output",
            default_value=os.path.join(_WS_ROOT, "videos", "ros_lay2stand.mp4"),
        ),
        DeclareLaunchArgument("lin_vel_max", default_value="1.5"),
        DeclareLaunchArgument("speed", default_value="1.0"),
        DeclareLaunchArgument("vel_profile", default_value="",
                              description="hard-step engage speed schedule "
                                          "'dur:speed,...' (e.g. '5:1,2:0,8:1'); "
                                          "overrides speed, no ramps."),
        DeclareLaunchArgument("action_delay", default_value="0"),
        DeclareLaunchArgument("proprio_latency_s", default_value="0.016",
                              description="Sleep after inference (s) to enforce fixed "
                                          "sensor→publish latency. Default 0.016=paper §4.1; "
                                          "0.0=off."),
        DeclareLaunchArgument("policy_kp", default_value="35.0"),
        DeclareLaunchArgument("policy_kd", default_value="0.6"),
        DeclareLaunchArgument("fps", default_value="50"),
        DeclareLaunchArgument("width", default_value="1280"),
        DeclareLaunchArgument("height", default_value="720"),
        DeclareLaunchArgument("camera", default_value="follow_cam"),
        DeclareLaunchArgument("title", default_value="",
                              description="optional header text drawn on every "
                                          "recorded frame (policy id + scene)."),
        DeclareLaunchArgument("metrics_only", default_value="false",
                              description="record base-height JSON but skip MP4 rendering."),
        DeclareLaunchArgument("clean_frame", default_value="false",
                              description="render the scene without telemetry, depth PiP, "
                                          "or yaw arrows."),
        DeclareLaunchArgument("depth_mode", default_value="render"),
        DeclareLaunchArgument("depth_noise_scale", default_value="0.0",
                              description="latest training depth-noise scale: "
                                          "0=clean, 0.5=reality-tuned, 1=2x rates."),
        DeclareLaunchArgument("depth_cam_roll_deg", default_value="0.0",
                              description="Rotate front_depth_cam about its optical "
                                          "axis by this many degrees."),
        DeclareLaunchArgument("depth_stereo", default_value="auto",
                              description="D435 stereo geometry: auto enables with "
                                          "depth noise; on/off override."),
        DeclareLaunchArgument("depth_seed", default_value="-1",
                              description="non-negative seed makes ROS depth noise repeatable."),
        DeclareLaunchArgument("depth_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("depth_sim_gate_on_engage", default_value="0"),
        DeclareLaunchArgument("depth_latency_s", default_value="0.08"),
        DeclareLaunchArgument("policy_rate_hz", default_value="50.0"),
        DeclareLaunchArgument("depth_gate_on_engage", default_value="1"),
        DeclareLaunchArgument("depth_reset_on_engage", default_value="1"),
        DeclareLaunchArgument("encoder_use_shared_contact", default_value="1"),
        DeclareLaunchArgument("publish_raw_last_action", default_value="1"),
        DeclareLaunchArgument("clip_applied_action", default_value="1"),
        DeclareLaunchArgument("encoder_trace_log", default_value=""),
        DeclareLaunchArgument("action_log", default_value=""),
        DeclareLaunchArgument("latency_encoder_log", default_value=""),
        DeclareLaunchArgument("latency_actor_log", default_value=""),
        DeclareLaunchArgument("latency_low_level_log", default_value=""),
        DeclareLaunchArgument("friction_scale", default_value="1.0"),
        DeclareLaunchArgument("initial_x", default_value="0.0",
                              description="initial MuJoCo robot base x position (m)"),
        DeclareLaunchArgument("wait_init", default_value="2.0",
                              description="seconds of idle before B (lay→stand) press"),
        DeclareLaunchArgument("press", default_value="1.0",
                              description="seconds to hold B"),
        DeclareLaunchArgument("settle", default_value="4.0",
                              description="seconds to wait after B before engaging policy"),
        DeclareLaunchArgument("engage_duration", default_value="20.0",
                              description="seconds the policy stays engaged"),
        # Return-stop: ON BY DEFAULT since 2026-07-28. Once the robot has been out
        # past return_arm_x and comes back through return_x, joy_sequencer steps the
        # command to zero (staying ENGAGED, so the policy still holds the stance),
        # holds it for return_stop_hold, then lays down. Without this the episode
        # keeps commanding forward after the robot is already home: these students
        # are 32-D no-yaw with prop[6:8] pinned at 0, so nothing anchors heading and
        # a constant forward command turns the tail of every run into a heading
        # random walk that ping-pongs across the course (x 2.4 -> -0.9 -> 2.5) and
        # tells you nothing. Set stop_on_return_x:=false to record the full window.
        DeclareLaunchArgument("stop_on_return_x", default_value="true"),
        DeclareLaunchArgument("end_on_still", default_value="true",
                              description="end the recording once the robot has "
                                          "finished and come to a stop, instead of "
                                          "always capturing duration*fps frames"),
        DeclareLaunchArgument("return_x", default_value="0.0"),
        DeclareLaunchArgument("return_arm_x", default_value="1.0"),
        DeclareLaunchArgument("return_stop_hold", default_value="2.0"),
        DeclareLaunchArgument("slowdown", default_value="2.0",
                              description="seconds to ramp axes[1] from speed → 0 before laydown"),
        DeclareLaunchArgument("laydown", default_value="3.0",
                              description="seconds to hold A after engage (stand → lay)"),
        DeclareLaunchArgument("tail", default_value="2.0",
                              description="idle seconds after laydown before shutdown"),
        DeclareLaunchArgument("proprio_log", default_value="",
                              description="if non-empty, rl_distill_policy writes per-tick 53-D proprio to this tsv path."),
        DeclareLaunchArgument("proprio_log_ticks", default_value="2000",
                              description="number of policy ticks to log (50 Hz → 2000 = 40 s)."),
        DeclareLaunchArgument("viewer", default_value="false",
                              description="open the live X11 MuJoCo viewer; requires DISPLAY."),
        DeclareLaunchArgument("use_sim_truth_vel", default_value="0",
                              description="1 = inject MuJoCo body-frame qvel into "
                                          "priv_explicit[0:3] (bypass learned Estimator); "
                                          "use to test if velocity-estimate noise is the "
                                          "dominant sim2sim gap."),
        DeclareLaunchArgument(
            "zero_student_velocity_obs",
            default_value="0",
            description="1 = force actor priv_explicit[0:3] to zero for students "
                        "trained with zero_student_velocity_obs=True.",
        ),
        DeclareLaunchArgument("vel_log", default_value="",
                              description="If non-empty, write per-tick TSV "
                                          "(t, cmd_vx, gt, ekf, est) to this path."),
        DeclareLaunchArgument("vel_log_ticks", default_value="2000",
                              description="Number of policy ticks to log (50 Hz)."),
        DeclareLaunchArgument("run_ekf", default_value="1",
                              description="1 = also launch ekf_only_node so /base_velocity is published."),
        DeclareLaunchArgument("zero_yaw_pred", default_value="0",
                              description="1 = force prop[6:8] yaw_pred slice to 0 (bypass "
                                          "depth-encoder yaw head); diagnostic only."),
        DeclareLaunchArgument("use_true_yaw", default_value="0",
                              description="1 = inject true delta_yaw (target heading 0) from "
                                          "MuJoCo base quat into prop[6:8]; overrides "
                                          "zero_yaw_pred and the depth-encoder yaw head."),
        DeclareLaunchArgument("env_class_eq_17", default_value="0",
                              description="1 = flip prop[11:13] indicator to (0, 1) "
                                          "matching parkour_flat training (go2_flat / L00-* "
                                          "exptids, env_class==17). Default 0 keeps the "
                                          "(1, 0) indicator used by every other env_class. "
                                          "Passed to BOTH encoder and policy (2026-06-06: "
                                          "previously the encoder never got it — silent "
                                          "prop[11:13] desync, commit-6542128 failure class)."),
        DeclareLaunchArgument("contact_threshold", default_value="2.0",
                              description="Foot-force threshold for proprio[49:53]; passed to "
                                          "BOTH encoder and policy so the gait-phase bits stay "
                                          "in lockstep. 2.0 N = Isaac training value, paired with "
                                          "mujoco_simulator's net contact force (0 in swing)."),
        DeclareLaunchArgument("lockstep", default_value="false",
                              description="true = simulator advances 4 substeps inside the "
                                          "/target_torque callback instead of a wall-clock "
                                          "timer (removes the free-run wall-clock-vs-sim-rate "
                                          "artifact, F15#1). Was previously hardcoded False — "
                                          "lockstep:=true was silently ignored before 2026-07-27."),
        DeclareLaunchArgument("lift_test", default_value="false",
                              description="1/true = after stand-up, pin the base and lift it "
                                          "lift_height m off the floor for lift_up_s, then set it "
                                          "back down for lift_down_s (no RL engage). Logs foot "
                                          "forces with /lift_phase markers."),
        DeclareLaunchArgument("lift_height", default_value="0.10",
                              description="meters to lift the base off the floor in lift_test."),
        DeclareLaunchArgument("lift_stand_hold_s", default_value="1.5",
                              description="standing-baseline seconds before lifting."),
        DeclareLaunchArgument("lift_up_s", default_value="2.0",
                              description="seconds to hold the robot lifted."),
        DeclareLaunchArgument("lift_down_s", default_value="2.0",
                              description="seconds to hold the robot back on the ground."),
        DeclareLaunchArgument("lift_ramp_s", default_value="0.25",
                              description="ramp seconds for the up/down base moves."),
        DeclareLaunchArgument("snap_qpos_on_engage", default_value="true",
                              description="snap qpos to the training init on the /policy_engaged "
                                          "rising edge. Set false for lift_test (no engage)."),
        DeclareLaunchArgument("fused", default_value="true",
                              description="DEFAULT since 2026-07-27. Runs the "
                                          "single-process distill_fused_policy instead of "
                                          "depth_encoder_node + rl_distill_policy, so ONE "
                                          "proprio frame per tick drives both the encoder "
                                          "and the actor (the training contract). Only "
                                          "/rl/depth_image stays on the wire. Set false for "
                                          "the legacy two-node split. See FINDINGS "
                                          "F01-F03/F12 and notes/"
                                          "the fused-vs-embedded A/B: statistically identical."),
    ]

    # Total recorded length = init + stand + settle + engage + slowdown + laydown + tail.
    duration = PythonExpression([
        "str(float(", wait_init, ") + float(", press, ") + float(",
        settle, ") + float(", engage_duration, ") + float(",
        slowdown, ") + float(", laydown, ") + float(", tail, "))"
    ])

    mujoco_sim = Node(
        package="deploy_rl_policy", executable="mujoco_simulator.py",
        name="mujoco_simulator",
        parameters=[{
            "scene_xml": scene_xml,
            "init_standing": False,
            "init_lay": True,              # spawn at lay pose so the FSM trips is_laydown_ immediately
            "viewer": ParameterValue(viewer, value_type=bool),
            "lockstep": ParameterValue(lockstep, value_type=bool),
            "decimation": 4,
            "friction_scale": ParameterValue(friction_scale, value_type=float),
            "initial_x": ParameterValue(initial_x, value_type=float),
            "snap_qpos_on_engage": ParameterValue(snap_qpos_on_engage, value_type=bool),
            "lift_test": ParameterValue(lift_test, value_type=bool),
            "lift_height": ParameterValue(lift_height, value_type=float),
            "lift_stand_hold_s": ParameterValue(lift_stand_hold_s, value_type=float),
            "lift_up_s": ParameterValue(lift_up_s, value_type=float),
            "lift_down_s": ParameterValue(lift_down_s, value_type=float),
            "lift_ramp_s": ParameterValue(lift_ramp_s, value_type=float),
        }],
    )

    low_level = Node(
        package="deploy_rl_policy", executable="low_level_ctrl",
        name="low_level_ctrl",
        parameters=[{
            "is_simulation": True,
            "policy_kp": ParameterValue(policy_kp, value_type=float),
            "policy_kd": ParameterValue(policy_kd, value_type=float),
            "latency_log": latency_low_level_log,
            # keep-alive must be OFF in lockstep: the sim steps per received
            # /lowcmd, so republishes would run sim time ~5x the policy rate.
            "keepalive_ms": ParameterValue(
                PythonExpression(["0 if '", lockstep, "'.lower() == 'true' else 5"]),
                value_type=int),
        }],
    )

    depth_sim = Node(
        package="deploy_rl_policy", executable="depth_sim_node.py",
        name="depth_sim_node",
        arguments=["--scene", scene_xml, "--depth_mode", depth_mode,
                   "--depth_noise_scale", depth_noise_scale,
                   "--cam_roll_deg", depth_cam_roll_deg,
                   "--stereo_shadow", depth_stereo,
                   "--seed", depth_seed,
                   "--rate", depth_rate_hz,
                   "--gate_on_engage", depth_sim_gate_on_engage],
    )

    depth_encoder = Node(
        package="deploy_rl_policy", executable="depth_encoder_node.py",
        name="depth_encoder_node",
        # Paper §4.1: enforce 80 ms constant depth latency (capture → latent
        # publish). Training assumes this constant delay too (depth_buffer
        # update_interval=5 ticks renders depth every 100 ms and the actor
        # reads buffer[-2], i.e. the second-to-last frame). Removing it
        # makes the deploy depth fresher than training and the heading
        # head closes the loop on out-of-distribution latency.
        arguments=["--ckpt", ckpt, "--ep_root", ep_root,
                   "--lin_vel_max", lin_vel_max,
                   "--enforce_latency_s", depth_latency_s,
                   "--contact_threshold", contact_threshold,
                   "--env_class_eq_17", env_class_eq_17,
                   "--gate_on_engage", depth_gate_on_engage,
                   "--reset_on_engage", depth_reset_on_engage,
                   "--use_shared_contact", encoder_use_shared_contact,
                   "--trace_log", encoder_trace_log,
                   "--latency_log", latency_encoder_log],
        condition=UnlessCondition(fused),
    )

    rl_policy = Node(
        package="deploy_rl_policy", executable="rl_distill_policy.py",
        name="rl_distill_policy",
        arguments=["--ckpt", ckpt, "--ep_root", ep_root,
                   "--action_delay", action_delay,
                   "--lin_vel_max", lin_vel_max,
                   "--proprio_log", proprio_log,
                   "--proprio_log_ticks", proprio_log_ticks,
                   "--use_sim_truth_vel", use_sim_truth_vel,
                   "--zero_student_velocity_obs", zero_student_velocity_obs,
                   "--vel_log", vel_log,
                   "--vel_log_ticks", vel_log_ticks,
                   "--zero_yaw_pred", zero_yaw_pred,
                   "--use_true_yaw", use_true_yaw,
                   "--env_class_eq_17", env_class_eq_17,
                   "--contact_threshold", contact_threshold,
                   "--proprio_latency_s", proprio_latency_s,
                   "--publish_raw_last_action", publish_raw_last_action,
                   "--clip_applied_action", clip_applied_action,
                   "--action_log", action_log,
                   "--latency_log", latency_actor_log,
                   "--rate_hz", policy_rate_hz],
        condition=UnlessCondition(fused),
    )

    # fused:=true — one process owns encoder + actor off ONE proprio frame per
    # tick. Replaces depth_encoder + rl_policy above; depth_sim_node is
    # unchanged and /rl/depth_image remains the only in-loop topic between
    # processes (mirroring the EP-on-Go2 deployment). The flags that had to be
    # passed identically to both nodes (contact_threshold, env_class_eq_17,
    # lin_vel_max) are consumed exactly once here, so they cannot desync.
    fused_policy = Node(
        package="deploy_rl_policy", executable="distill_fused_policy.py",
        name="distill_fused_policy",
        arguments=["--ckpt", ckpt, "--ep_root", ep_root,
                   "--action_delay", action_delay,
                   "--lin_vel_max", lin_vel_max,
                   "--contact_threshold", contact_threshold,
                   "--env_class_eq_17", env_class_eq_17,
                   "--depth_latency_s", depth_latency_s,
                   "--proprio_latency_s", proprio_latency_s,
                   "--reset_on_engage", depth_reset_on_engage,
                   "--zero_yaw_pred", zero_yaw_pred,
                   "--use_true_yaw", use_true_yaw,
                   "--use_sim_truth_vel", use_sim_truth_vel,
                   "--zero_student_velocity_obs", zero_student_velocity_obs,
                   "--clip_applied_action", clip_applied_action,
                   "--rate_hz", policy_rate_hz,
                   "--trace_log", encoder_trace_log,
                   "--action_log", action_log,
                   "--latency_log", latency_actor_log,
                   "--proprio_log", proprio_log,
                   "--proprio_log_ticks", proprio_log_ticks,
                   "--vel_log", vel_log,
                   "--vel_log_ticks", vel_log_ticks],
        condition=IfCondition(fused),
    )

    # EKF that fuses MuJoCo /imu/data + /contact → /base_velocity (Twist).
    ekf = Node(
        package="base_velocity_estimator",
        executable="ekf_only_node",
        name="base_velocity_estimator",
        condition=IfCondition(run_ekf),
    )

    joy_seq = Node(
        package="deploy_rl_policy", executable="joy_sequencer.py",
        name="joy_sequencer",
        arguments=["--wait-init", wait_init,
                   "--press", press,
                   "--settle", settle,
                   "--speed", speed,
                   "--vel_profile", vel_profile,
                   "--duration", engage_duration,
                   "--stop-on-return-x", LaunchConfiguration("stop_on_return_x"),
                   "--return-x", LaunchConfiguration("return_x"),
                   "--return-arm-x", LaunchConfiguration("return_arm_x"),
                   "--return-stop-hold", LaunchConfiguration("return_stop_hold"),
                   "--slowdown", slowdown,
                   "--laydown", laydown,
                   "--tail", tail],
    )

    recorder = Node(
        package="deploy_rl_policy", executable="distill_recorder.py",
        name="distill_recorder",
        arguments=["--scene", scene_xml, "--output", output,
                   "--duration", duration, "--fps", fps,
                   "--width", width, "--height", height,
                   "--camera", camera,
                   "--lin_vel_max", lin_vel_max,
                   "--title", title,
                   PythonExpression([
                       "'--metrics-only' if '", metrics_only,
                       "' == 'true' else ''"
                   ]),
                   PythonExpression([
                       "'--clean-frame' if '", clean_frame,
                       "' == 'true' else ''"
                   ]),
                   # End the clip when the robot has finished and stopped, instead
                   # of always filling duration*fps frames (see distill_recorder).
                   PythonExpression([
                       "'--no_end_on_still' if '", LaunchConfiguration("end_on_still"),
                       "' == 'false' else ''"
                   ])],
    )

    return LaunchDescription(decls + [
        mujoco_sim, low_level, depth_sim, depth_encoder, rl_policy, fused_policy,
        joy_seq, recorder, ekf,
    ])
