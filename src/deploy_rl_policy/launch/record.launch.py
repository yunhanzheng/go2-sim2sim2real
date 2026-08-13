"""Headless ROS dry-run that produces an MP4.

Stack: mujoco_simulator (headless, init_standing) + depth_sim_node + depth_encoder_node
+ rl_distill_policy (force_lin_vel_x) + target_pos_bridge (replaces low_level_ctrl)
+ distill_recorder. No joy_node, no low_level_ctrl, no X11.

Usage:
  ros2 launch deploy_rl_policy record.launch.py \\
      ckpt:=/abs/path/to/model.pt \\
      scene_xml:=/abs/path/to/scene.xml \\
      output:=/abs/path/to/out.mp4 \\
      duration:=8.0 \\
      lin_vel_x:=0.5 \\
      action_delay:=0
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue

# install/<pkg>/share/<pkg> -> 3 up -> the repo (colcon workspace) root.
_WS_ROOT = os.path.normpath(
    os.path.join(get_package_share_directory("deploy_rl_policy"), "..", "..", "..")
)
# The training repo supplies the policy classes:
# https://github.com/Roman-089/go2-flyball-parkour
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
    duration = LaunchConfiguration("duration")
    lin_vel_x = LaunchConfiguration("lin_vel_x")
    action_delay = LaunchConfiguration("action_delay")
    policy_kp = LaunchConfiguration("policy_kp")
    policy_kd = LaunchConfiguration("policy_kd")
    fps = LaunchConfiguration("fps")
    width = LaunchConfiguration("width")
    height = LaunchConfiguration("height")
    camera = LaunchConfiguration("camera")
    depth_mode = LaunchConfiguration("depth_mode")
    depth_noise_scale = LaunchConfiguration("depth_noise_scale")
    depth_stereo = LaunchConfiguration("depth_stereo")
    viewer = LaunchConfiguration("viewer")
    record = LaunchConfiguration("record")
    friction_scale = LaunchConfiguration("friction_scale")
    contact_threshold = LaunchConfiguration("contact_threshold")
    env_class_eq_17 = LaunchConfiguration("env_class_eq_17")

    decls = [
        DeclareLaunchArgument("ckpt", default_value=_DEFAULT_CKPT),
        DeclareLaunchArgument("ep_root", default_value=_EP_ROOT),
        DeclareLaunchArgument(
            "scene_xml",
            default_value=os.path.join(_WS_ROOT, "resources", "go2", "scene_terrain.xml"),
        ),
        DeclareLaunchArgument(
            "output",
            default_value=os.path.join(_WS_ROOT, "videos", "ros_distill_dryrun.mp4"),
        ),
        DeclareLaunchArgument("duration", default_value="8.0"),
        DeclareLaunchArgument("lin_vel_x", default_value="0.5"),
        DeclareLaunchArgument("action_delay", default_value="0"),
        DeclareLaunchArgument("policy_kp", default_value="35.0"),
        DeclareLaunchArgument("policy_kd", default_value="0.6"),
        DeclareLaunchArgument("fps", default_value="50"),
        DeclareLaunchArgument("width", default_value="1280"),
        DeclareLaunchArgument("height", default_value="720"),
        DeclareLaunchArgument("camera", default_value="follow_cam"),
        DeclareLaunchArgument("depth_mode", default_value="render"),
        DeclareLaunchArgument("depth_noise_scale", default_value="0.0",
                              description="latest training depth-noise scale: "
                                          "0=clean, 0.5=reality-tuned, 1=2x rates."),
        DeclareLaunchArgument("depth_stereo", default_value="auto",
                              description="D435 stereo geometry: auto enables with "
                                          "depth noise; on/off override."),
        DeclareLaunchArgument("enforce_latency_s", default_value="0.080"),
        DeclareLaunchArgument("viewer", default_value="false",
                              description="If 'true', open the MuJoCo X11 viewer (requires DISPLAY; over SSH use vglrun)."),
        DeclareLaunchArgument("record", default_value="true",
                              description="If 'false', omit the distill_recorder node (live-view only, no MP4)."),
        DeclareLaunchArgument("friction_scale", default_value="1.0",
                              description="Multiplier on geom sliding friction. Use 0.5 to match Isaac base µ=1.0 against the scene's µ=2.0 default."),
        DeclareLaunchArgument("proprio_log", default_value="",
                              description="If non-empty, rl_distill_policy writes per-tick 53-D proprio to this txt path."),
        DeclareLaunchArgument("proprio_log_ticks", default_value="100",
                              description="Number of policy ticks to log to proprio_log (50 Hz → 100 = 2 s)."),
        # Both args reach depth_encoder_node AND rl_distill_policy — each
        # builds its own proprio copy, so a one-sided value silently desyncs
        # prop[11:13] / prop[49:53] between encoder forward and actor input
        # (commit 6542128; 2026-06-06 readiness audit). Until 2026-06-06 this
        # launch passed neither to the encoder.
        DeclareLaunchArgument("contact_threshold", default_value="2.0",
                              description="Foot-force threshold for proprio[49:53]; 2.0 N matches Isaac when using net MuJoCo foot contact."),
        DeclareLaunchArgument("env_class_eq_17", default_value="0",
                              description="1 for parkour_flat checkpoints (go2_flat / L00-*) so prop[11:13] = (0, 1)."),
    ]

    mujoco_sim_x11 = Node(
        package="deploy_rl_policy", executable="mujoco_simulator.py",
        name="mujoco_simulator",
        parameters=[{
            "scene_xml": scene_xml,
            "init_standing": True,
            "viewer": True,
            "lockstep": True,
            "decimation": 4,
            "friction_scale": ParameterValue(friction_scale, value_type=float),
        }],
        condition=IfCondition(viewer),
    )
    mujoco_sim = Node(
        package="deploy_rl_policy", executable="mujoco_simulator.py",
        name="mujoco_simulator",
        parameters=[{
            "scene_xml": scene_xml,
            "init_standing": True,
            "viewer": False,
            "lockstep": True,
            "decimation": 4,
            "friction_scale": ParameterValue(friction_scale, value_type=float),
        }],
        condition=UnlessCondition(viewer),
    )

    depth_sim = Node(
        package="deploy_rl_policy", executable="depth_sim_node.py",
        name="depth_sim_node",
        arguments=["--scene", scene_xml, "--depth_mode", depth_mode,
                   "--depth_noise_scale", depth_noise_scale,
                   "--stereo_shadow", depth_stereo],
    )

    depth_encoder = Node(
        package="deploy_rl_policy", executable="depth_encoder_node.py",
        name="depth_encoder_node",
        arguments=["--ckpt", ckpt, "--ep_root", ep_root,
                   "--force_lin_vel_x", lin_vel_x,
                   "--enforce_latency_s", LaunchConfiguration("enforce_latency_s"),
                   "--contact_threshold", contact_threshold,
                   "--env_class_eq_17", env_class_eq_17,
                   "--gate_on_engage", "1",
                   "--reset_on_engage", "1",
                   "--use_shared_contact", "1"],
    )

    rl_policy = Node(
        package="deploy_rl_policy", executable="rl_distill_policy.py",
        name="rl_distill_policy",
        arguments=["--ckpt", ckpt, "--ep_root", ep_root,
                   "--action_delay", action_delay,
                   "--force_lin_vel_x", lin_vel_x,
                   "--proprio_log", LaunchConfiguration("proprio_log"),
                   "--proprio_log_ticks", LaunchConfiguration("proprio_log_ticks"),
                   "--contact_threshold", contact_threshold,
                   "--env_class_eq_17", env_class_eq_17,
                   "--publish_raw_last_action", "1",
                   "--clip_applied_action", "1"],
    )

    bridge = Node(
        package="deploy_rl_policy", executable="target_pos_bridge.py",
        name="target_pos_bridge",
        arguments=["--kp", policy_kp, "--kd", policy_kd],
    )

    recorder = Node(
        package="deploy_rl_policy", executable="distill_recorder.py",
        name="distill_recorder",
        arguments=["--scene", scene_xml, "--output", output,
                   "--duration", duration, "--fps", fps,
                   "--width", width, "--height", height,
                   "--camera", camera],
        condition=IfCondition(record),
    )

    return LaunchDescription(decls + [
        mujoco_sim_x11, mujoco_sim, depth_sim, depth_encoder, rl_policy, bridge, recorder,
    ])
