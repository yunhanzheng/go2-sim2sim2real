"""Launch the extreme-parkour distillation deploy stack.

Sim:
  ros2 launch deploy_rl_policy distill.launch.py is_simulation:=true

Real (with realsense2_camera already running):
  ros2 launch deploy_rl_policy distill.launch.py is_simulation:=false

Common args:
  ckpt:=/abs/path/to/model_6500.pt
  ep_root:=/abs/path/to/go2-flyball-parkour
  depth_mode:=render|zeros          (sim only)
  scene_xml:=/abs/path/to/scene.xml (sim only; depth_sim_node + mujoco_simulator must agree)
  policy_kp:=35.0  policy_kd:=0.6
  action_delay:=0|1
  depth_latency_s:=<auto> proprio_latency_s:=<auto> (sim: 0.080/0.016 s; real: 0/0)
  depth_input_topic:=/camera/depth/image_rect_raw  (real only; Jetson depth topic)
  lin_vel_max:=1.0  binary_vel:=0|1  vel_deadzone:=0.2   (defaults match run_real_remote.sh)
  contact_threshold:=<auto>         (foot-force for proprio[49:53]; default 2.0 in sim
                                     (Newtons, matches training), 20.0 on real robot
                                     (raw sensor units, measured); must match encoder+policy)
  env_class_eq_17:=0|1              (1 for L00-* / parkour_flat checkpoints)
  use_keyboard:=true|false          (true => keyboard_to_joy replaces joy_node:
                                     A=lay, B=stand, Space(hold)=engage,
                                     W/S=fwd, ←/→=strafe, Q/E=yaw, Z+X(hold)=quit)
  kbd_speed:=1.0                    (keyboard stick magnitude in [0,1] when
                                     use_keyboard:=true; scales top speed)

Drive the robot in MuJoCo from the keyboard (deploy stack, no gamepad):
  ros2 launch deploy_rl_policy distill.launch.py is_simulation:=true \\
      use_keyboard:=true ckpt:=/abs/model.pt scene_xml:=/abs/scene.xml
On the Mac use the env+path wrapper: deploy_mac/run/keyboard_control.sh
"""
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    # view_prediction.py lives at <workspace>/deploy_mac/diag/ — resolve from
    # the installed package share dir: install/<pkg>/share/<pkg> -> 3 up -> workspace
    _pkg_share = get_package_share_directory("deploy_rl_policy")
    _ws_root = os.path.normpath(os.path.join(_pkg_share, "..", "..", ".."))
    _view_script = os.path.join(_ws_root, "deploy_mac", "diag", "view_prediction.py")

    # The training repo supplies the policy classes:
    # https://github.com/Roman-089/go2-flyball-parkour
    # Override with $EP_ROOT or the ep_root: launch argument.
    _ep_root = os.environ.get(
        "EP_ROOT", os.path.join(os.path.dirname(_ws_root), "go2-flyball-parkour")
    )
    _default_ckpt = os.path.join(
        _ws_root, "models", "L00-02-flat-baseline-delay-distillation-noised",
        "model_14500.pt",
    )

    is_simulation = LaunchConfiguration("is_simulation")
    ckpt = LaunchConfiguration("ckpt")
    ep_root = LaunchConfiguration("ep_root")
    depth_mode = LaunchConfiguration("depth_mode")
    depth_noise_scale = LaunchConfiguration("depth_noise_scale")
    depth_stereo = LaunchConfiguration("depth_stereo")
    scene_xml = LaunchConfiguration("scene_xml")
    policy_kp = LaunchConfiguration("policy_kp")
    policy_kd = LaunchConfiguration("policy_kd")
    action_delay = LaunchConfiguration("action_delay")
    depth_latency_s = LaunchConfiguration("depth_latency_s")
    proprio_latency_s = LaunchConfiguration("proprio_latency_s")
    lin_vel_max = LaunchConfiguration("lin_vel_max")
    binary_vel = LaunchConfiguration("binary_vel")
    vel_deadzone = LaunchConfiguration("vel_deadzone")
    use_keyboard = LaunchConfiguration("use_keyboard")
    kbd_speed = LaunchConfiguration("kbd_speed")
    contact_threshold = LaunchConfiguration("contact_threshold")
    env_class_eq_17 = LaunchConfiguration("env_class_eq_17")
    depth_input_topic = LaunchConfiguration("depth_input_topic")
    launch_viewer = LaunchConfiguration("launch_viewer")
    sim_prefix = LaunchConfiguration("sim_prefix")
    momentary_engage = LaunchConfiguration("momentary_engage")
    joy_timeout_s = LaunchConfiguration("joy_timeout_s")
    rl_stall_timeout_s = LaunchConfiguration("rl_stall_timeout_s")
    depth_gate_on_engage = LaunchConfiguration("depth_gate_on_engage")
    depth_reset_on_engage = LaunchConfiguration("depth_reset_on_engage")
    encoder_use_shared_contact = LaunchConfiguration("encoder_use_shared_contact")
    publish_raw_last_action = LaunchConfiguration("publish_raw_last_action")
    fused = LaunchConfiguration("fused")
    clip_applied_action = LaunchConfiguration("clip_applied_action")
    zero_student_velocity_obs = LaunchConfiguration("zero_student_velocity_obs")

    decls = [
        DeclareLaunchArgument("is_simulation", default_value="true"),
        DeclareLaunchArgument("ckpt", default_value=_default_ckpt),
        DeclareLaunchArgument("ep_root", default_value=_ep_root),
        DeclareLaunchArgument("depth_mode", default_value="render"),
        DeclareLaunchArgument("depth_noise_scale", default_value="0.0",
                              description="latest training depth-noise scale: "
                                          "0=clean, 0.5=reality-tuned, 1=2x rates."),
        DeclareLaunchArgument("depth_stereo", default_value="auto",
                              description="D435 stereo geometry: auto enables with "
                                          "depth noise; on/off override."),
        DeclareLaunchArgument(
            "scene_xml",
            default_value=os.path.join(_ws_root, "resources", "go2", "scene_terrain.xml"),
        ),
        DeclareLaunchArgument("policy_kp", default_value="35.0"),
        DeclareLaunchArgument("policy_kd", default_value="0.6"),
        DeclareLaunchArgument("action_delay", default_value="0"),
        # Synthetic latency is useful for simulation ablations, but real deploy
        # uses the naturally occurring latency and measures it separately.
        DeclareLaunchArgument(
            "depth_latency_s",
            default_value=PythonExpression(
                ["'0.080' if '", is_simulation, "' == 'true' else '0.0'"]),
        ),
        DeclareLaunchArgument(
            "proprio_latency_s",
            default_value=PythonExpression(
                ["'0.016' if '", is_simulation, "' == 'true' else '0.0'"]),
        ),
        # Match the real wrapper defaults (run_real_remote.sh): trained/deployed
        # command contract is bang-bang 0/1.0 m/s. binary_vel/vel_deadzone go to
        # BOTH nodes so the encoder's embedded proprio copy stays in lockstep
        # with the actor (same desync class as env_class_eq_17/contact_threshold).
        DeclareLaunchArgument("lin_vel_max", default_value="1.0"),
        DeclareLaunchArgument("binary_vel", default_value="1",
                              description="1 = bang-bang: stick forward past "
                                          "vel_deadzone commands lin_vel_max, else 0; "
                                          "0 = proportional stick control."),
        DeclareLaunchArgument("vel_deadzone", default_value="0.2"),
        DeclareLaunchArgument("use_keyboard", default_value="false"),
        DeclareLaunchArgument("kbd_speed", default_value="1.0",
                              description="keyboard stick magnitude in [0,1] when "
                                          "use_keyboard:=true (scales top speed: "
                                          "cmd_vx = lin_vel_max * kbd_speed)."),
        # rl_distill_policy owns the 50 Hz contact filter and publishes the
        # result to depth_encoder_node. The threshold therefore applies once
        # and both proprio copies receive the same 20 ms history sample.
        #
        # Default depends on is_simulation:
        #   sim  -> 2.0  : Isaac training value (contact_forces L2 norm > 2.0 N).
        #                  Paired with mujoco_simulator's NET ground-contact force
        #                  (mj_contactForce over the foot geoms, ~0 swing / ~37 stance).
        #                  NB: the old go2.xml <force> sensor floored at ~2.4 N in
        #                  swing, so 2.0 there read always-in-contact — fixed now.
        #   real -> 20,20,20,20 : the Go2's /lowstate.foot_force is in RAW sensor
        #                  units, so the real threshold is intentionally distinct
        #                  from the 2.0 N MuJoCo/training-domain threshold.
        # Override either with contact_threshold:=X (scalar) or X,X,X,X (per-foot).
        DeclareLaunchArgument(
            "contact_threshold",
            default_value=PythonExpression(
                ["'2.0' if '", is_simulation, "' == 'true' else '20,20,20,20'"]),
        ),
        # L00-* checkpoints (parkour_flat / env_class==17) need this set to 1
        # so prop[11:13] = (0, 1) matches training. Default 0 covers every
        # other env_class (parkour, wall, etc.). Like contact_threshold it is
        # passed to BOTH depth_encoder_node and rl_distill_policy — the encoder
        # embeds its own proprio copy into the depth latent, so a one-sided
        # flag desyncs prop[11:13] between encoder forward and actor input
        # (2026-06-06 real-robot readiness audit, same class as commit 6542128).
        DeclareLaunchArgument("env_class_eq_17", default_value="0"),
        # Real mode only: depth_camera_node's input topic. The Jetson's
        # realsense2_camera publishes RECTIFIED depth on
        # /camera/depth/image_rect_raw (not image_raw). depth_camera_node's own
        # default already matches this, so the bare launch works; this arg makes
        # the dependency explicit and lets you override it — notably for the
        # realsense2_camera v4.x /camera/camera/... double-namespace
        # (check `ros2 topic list | grep image_rect_raw`). Matches
        # run_real_remote.sh / run_dryrun_remote.sh.
        DeclareLaunchArgument(
            "depth_input_topic",
            default_value="/camera/depth/image_rect_raw"),
        # Real mode only: launch the live yaw-prediction viewer window automatically.
        # Set launch_viewer:=false to suppress (e.g. headless / CI runs).
        DeclareLaunchArgument("launch_viewer", default_value="true"),
        # macOS only: the live MuJoCo viewer (mujoco.viewer.launch_passive)
        # must run under `mjpython`, which owns the main thread for the Cocoa
        # event loop. Set sim_prefix:=mjpython on a Mac; leave empty on Linux
        # (regular python3 + GLFW/EGL works there). Applied as the launch
        # prefix of the mujoco_simulator node only.
        DeclareLaunchArgument("sim_prefix", default_value=""),
        # Dead-man engage: when true the policy runs ONLY while LB+RB (Space on
        # the keyboard) are held; releasing them stops inference and returns to
        # the standing pose. Mode-dependent default: sim keeps the latch
        # behaviour (engage stays on until A=lay — hold-Space would break every
        # keyboard workflow), but on the REAL robot the dead-man is on by
        # default, matching run_real_remote.sh — this also arms joy_timeout_s,
        # which is inert without momentary engagement.
        DeclareLaunchArgument(
            "momentary_engage",
            default_value=PythonExpression(
                ["'false' if '", is_simulation, "' == 'true' else 'true'"])),
        # Dead-man on joystick LOSS (momentary mode only): if /joy goes silent
        # for longer than this while the policy is engaged (transmitter off,
        # bluetooth drop, joy node crash), low_level_ctrl settles back to
        # standing exactly as if LB+RB were released. Both joy sources stream
        # continuously (joy_node autorepeat 20 Hz, keyboard_to_joy 50 Hz), so
        # 0.5 s of silence really means the link is gone. 0.0 disables.
        DeclareLaunchArgument("joy_timeout_s", default_value="0.5"),
        # Hard bound on the engaged /rl/target_pos keep-alive: past this the
        # actor is treated as dead — low_level_ctrl latches a stall fault and
        # settles to standing (re-engage needs an LB+RB release first). Longer
        # than the ~0.7 s transient stalls seen under Mac overload 2026-06-19,
        # so benign hiccups still ride the keep-alive. 0.0 disables.
        DeclareLaunchArgument("rl_stall_timeout_s", default_value="1.5"),
        DeclareLaunchArgument("depth_gate_on_engage", default_value="1",
                              description="Only advance the depth GRU while the policy is engaged."),
        DeclareLaunchArgument("depth_reset_on_engage", default_value="1",
                              description="Reset depth-GRU state at every policy engagement."),
        DeclareLaunchArgument("encoder_use_shared_contact", default_value="1",
                              description="Use the actor-owned 50 Hz contact history in the encoder."),
        DeclareLaunchArgument("publish_raw_last_action", default_value="1",
                              description="Publish unclipped previous actor output to the encoder, as in Isaac."),
        DeclareLaunchArgument("clip_applied_action", default_value="1",
                              description="Clamp applied raw action to +/-4.8; keep enabled for deploy."),
        DeclareLaunchArgument(
            "fused", default_value="true",
            description="DEFAULT since 2026-07-27. Runs the single-process distill_fused_policy "
                        "instead of depth_encoder_node + rl_distill_policy, so ONE "
                        "proprio frame per tick drives both the encoder and the "
                        "actor (the training contract; see FINDINGS F01-F03/F12 and "
                        "the fused-vs-two-node A/B). When true, "
                        "depth_gate_on_engage / encoder_use_shared_contact / "
                        "publish_raw_last_action become meaningless (satisfied "
                        "structurally) and are NOT forwarded."),
        DeclareLaunchArgument(
            "zero_student_velocity_obs",
            default_value="0",
            description="1 = force actor priv_explicit[0:3] to zero for students "
                        "trained with zero_student_velocity_obs=True.",
        ),
    ]

    is_sim_cond = IfCondition(is_simulation)
    is_real_cond = UnlessCondition(is_simulation)
    use_kbd_cond = IfCondition(use_keyboard)
    use_joy_cond = UnlessCondition(use_keyboard)
    viewer_cond = IfCondition(PythonExpression(
        ["'", is_simulation, "' == 'false' and '", launch_viewer, "' == 'true'"]))

    joy = Node(
        package="joy", executable="joy_node", name="joy_node",
        condition=use_joy_cond,
    )
    keyboard_joy = Node(
        package="deploy_rl_policy", executable="keyboard_to_joy.py",
        name="keyboard_to_joy",
        arguments=["--speed", kbd_speed],
        condition=use_kbd_cond,
        output="screen",
    )

    low_level = Node(
        package="deploy_rl_policy", executable="low_level_ctrl",
        name="low_level_ctrl",
        parameters=[{
            "is_simulation": is_simulation,
            "policy_kp": policy_kp,
            "policy_kd": policy_kd,
            "momentary_engage": ParameterValue(momentary_engage, value_type=bool),
            "joy_timeout_s": ParameterValue(joy_timeout_s, value_type=float),
            "rl_stall_timeout_s": ParameterValue(rl_stall_timeout_s, value_type=float),
        }],
    )

    mujoco_sim = Node(
        package="deploy_rl_policy", executable="mujoco_simulator.py",
        name="mujoco_simulator",
        parameters=[{"scene_xml": scene_xml}],
        prefix=sim_prefix,
        condition=is_sim_cond,
    )

    depth_sim = Node(
        package="deploy_rl_policy", executable="depth_sim_node.py",
        name="depth_sim_node",
        arguments=["--scene", scene_xml, "--depth_mode", depth_mode,
                   "--depth_noise_scale", depth_noise_scale,
                   "--stereo_shadow", depth_stereo],
        condition=is_sim_cond,
    )

    depth_camera = Node(
        package="deploy_rl_policy", executable="depth_camera_node.py",
        name="depth_camera_node",
        parameters=[{"input_topic": depth_input_topic}],
        condition=is_real_cond,
    )

    depth_encoder = Node(
        package="deploy_rl_policy", executable="depth_encoder_node.py",
        name="depth_encoder_node",
        arguments=["--ckpt", ckpt, "--ep_root", ep_root,
                   "--lin_vel_max", lin_vel_max,
                   "--binary_vel", binary_vel,
                   "--vel_deadzone", vel_deadzone,
                   "--enforce_latency_s", depth_latency_s,
                   "--contact_threshold", contact_threshold,
                   "--env_class_eq_17", env_class_eq_17,
                   "--gate_on_engage", depth_gate_on_engage,
                   "--reset_on_engage", depth_reset_on_engage,
                   "--use_shared_contact", encoder_use_shared_contact],
        condition=UnlessCondition(fused),
    )

    rl_policy = Node(
        package="deploy_rl_policy", executable="rl_distill_policy.py",
        name="rl_distill_policy",
        arguments=["--ckpt", ckpt, "--ep_root", ep_root,
                   "--action_delay", action_delay,
                   "--proprio_latency_s", proprio_latency_s,
                   "--lin_vel_max", lin_vel_max,
                   "--binary_vel", binary_vel,
                   "--vel_deadzone", vel_deadzone,
                   "--contact_threshold", contact_threshold,
                   "--env_class_eq_17", env_class_eq_17,
                   "--publish_raw_last_action", publish_raw_last_action,
                   "--clip_applied_action", clip_applied_action,
                   "--zero_student_velocity_obs", zero_student_velocity_obs],
        condition=UnlessCondition(fused),
    )

    # fused:=true — one process owns encoder + actor off ONE proprio frame per
    # tick. Same topic contract as the pair it replaces (/lowstate +
    # /rl/depth_image in, /rl/target_pos out), so depth_sim_node /
    # depth_camera_node / low_level_ctrl are untouched. On the real robot the
    # 80 ms depth budget is enforced by an age gate over timestamped frames
    # (clock-offset corrected via /rl/depth_clock_offset) rather than a sleep,
    # because a fused tick cannot block without missing the 20 ms actor deadline.
    fused_policy = Node(
        package="deploy_rl_policy", executable="distill_fused_policy.py",
        name="distill_fused_policy",
        arguments=["--ckpt", ckpt, "--ep_root", ep_root,
                   "--action_delay", action_delay,
                   "--proprio_latency_s", proprio_latency_s,
                   "--depth_latency_s", depth_latency_s,
                   "--lin_vel_max", lin_vel_max,
                   "--binary_vel", binary_vel,
                   "--vel_deadzone", vel_deadzone,
                   "--contact_threshold", contact_threshold,
                   "--env_class_eq_17", env_class_eq_17,
                   "--reset_on_engage", depth_reset_on_engage,
                   "--clip_applied_action", clip_applied_action,
                   "--zero_student_velocity_obs", zero_student_velocity_obs],
        condition=IfCondition(fused),
    )

    viewer = ExecuteProcess(
        cmd=[sys.executable, _view_script],
        output="screen",
        condition=viewer_cond,
    )

    return LaunchDescription(decls + [
        joy, keyboard_joy, low_level, mujoco_sim, depth_sim, depth_camera, depth_encoder, rl_policy,
        fused_policy, viewer,
    ])
