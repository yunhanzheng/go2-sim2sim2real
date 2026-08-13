"""Launch all four sim2sim nodes from a single terminal.

Usage:
    ros2 launch deploy_rl_policy sim2sim.launch.py
    ros2 launch deploy_rl_policy sim2sim.launch.py scene:=scene_terrain.xml

Each node's stdout/stderr is multiplexed into the launch terminal, prefixed
with the node name. Ctrl-C cleanly shuts everything down.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    scene_arg = DeclareLaunchArgument(
        "scene",
        default_value="scene_2hurdles_20cm_8m.xml",
        description="MuJoCo scene XML filename in resources/go2/",
    )
    viewer_arg = DeclareLaunchArgument(
        "viewer",
        default_value="true",
        description="Open MuJoCo viewer window. Set false for SSH/headless.",
    )
    follow_robot_arg = DeclareLaunchArgument(
        "follow_robot",
        default_value="false",
        description="Lock the MuJoCo camera to body 'base_link' so it tracks "
                    "the robot. Equivalent to double-clicking the robot in "
                    "the viewer. Defaults off (free-orbit).",
    )
    fake_joy_arg = DeclareLaunchArgument(
        "fake_joy",
        default_value="false",
        description="Use the scripted fake_joy publisher instead of joy_node. "
                    "Stand at t=1s, engage policy at t=8s.",
    )
    vgl_arg = DeclareLaunchArgument(
        "vgl",
        default_value="false",
        description="Run the simulator under VirtualGL (vglrun). "
                    "Use over SSH X11 forwarding so the viewer can render on "
                    "the local NVIDIA GPU and stream pixels via X11.",
    )
    vgl_compress_arg = DeclareLaunchArgument(
        "vgl_compress",
        default_value="proxy",
        description="VirtualGL transport: 'proxy' (uncompressed, default — "
                    "works everywhere but slow), 'xv' (XVideo extension, "
                    "needs adaptor on local X server, often unavailable on "
                    "Mac XQuartz / VS Code SSH / VcXsrv), 'jpeg' (fastest, "
                    "needs vglclient running on local laptop).",
    )
    viewer_fps_arg = DeclareLaunchArgument(
        "viewer_fps",
        default_value="30.0",
        description="Render rate cap for mujoco viewer (Hz). Decouples viewer "
                    "from 200 Hz physics so VGL/X11 doesn't bottleneck physics.",
    )
    low_quality_arg = DeclareLaunchArgument(
        "low_quality",
        default_value="false",
        description="Disable shadow maps and MSAA in the MuJoCo renderer. "
                    "Big framerate win over slow remote displays. UI/geometry "
                    "are unchanged — only shadow casting and edge anti-alias "
                    "are turned off.",
    )
    match_isaac_arg = DeclareLaunchArgument(
        "match_isaac",
        default_value="true",
        description="Zero MuJoCo joint armature + frictionloss to match the "
                    "Isaac Gym training environment. Set false to keep go2.xml "
                    "values (more realistic but unused during training, hurts "
                    "policy fidelity).",
    )
    forward_vel_cmd_arg = DeclareLaunchArgument(
        "forward_vel_cmd",
        default_value="0.0",
        description="Forward velocity command (m/s) the policy receives via "
                    "obs[10]. 0 = stand in place (learned standing). 1.5 = "
                    "walk forward toward the wall.",
    )
    dump_obs_arg = DeclareLaunchArgument(
        "dump_obs",
        default_value="0",
        description="If >0, dump the first N obs/action vectors to "
                    "/tmp/deploy_obs_<i>.npz for comparison against the "
                    "extreme-parkour training observation.",
    )
    use_sandbox_actor_arg = DeclareLaunchArgument(
        "use_sandbox_actor",
        default_value="true",
        description="If true, ep_rl_policy loads the raw checkpoint via "
                    "ep_actor_sandbox.build_base_actor (Actor + Estimator "
                    "instantiated directly, no JIT). If false, falls back "
                    "to torch.jit.load on the traced policy.",
    )
    use_ground_truth_priv_arg = DeclareLaunchArgument(
        "use_ground_truth_priv",
        default_value="false",
        description="DEBUG: bypass the estimator and feed obs[priv_explicit] "
                    "directly from MuJoCo's body-frame base_lin_vel "
                    "(/mujoco/base_lin_vel). Matches play.py's act_inference "
                    "path so sim2sim obs comparisons isolate the physics gap "
                    "from the estimator gap. Sim-only.",
    )
    track_goal_arg = DeclareLaunchArgument(
        "track_goal",
        default_value="false",
        description="Anchor a single tracking goal `goal_dist` m along the "
                    "robot's initial body-forward axis (captured from first "
                    "/mujoco/pos). proprio[6] / proprio[7] then carry the "
                    "running delta_yaw to that goal instead of the default 0, "
                    "so the policy gets a real heading-error signal.",
    )
    goal_dist_arg = DeclareLaunchArgument(
        "goal_dist",
        default_value="5.0",
        description="Distance (m) of the tracking goal along initial "
                    "body-forward when track_goal=true.",
    )
    startup_wait_s_arg = DeclareLaunchArgument(
        "startup_wait_s",
        default_value="0.0",
        description="Hold the policy idle (no /rl/target_pos publishes) for "
                    "this many seconds after the first /mujoco/lowstate, so "
                    "the MuJoCo viewer has time to come up over X11/VGL "
                    "before any motion. While idle the simulator's physics "
                    "thread also stays parked, so the robot is frozen at "
                    "the post-presettle standing pose. 0 disables.",
    )
    stand_for_steps_arg = DeclareLaunchArgument(
        "stand_for_steps",
        default_value="0",
        description="If > 0, force forward_vel_cmd=0 for the first N policy "
                    "steps (each step = 20 ms), letting the robot stabilise "
                    "in standing pose before walking. After N steps the "
                    "configured forward_vel_cmd takes over. e.g. 250 = 5 s "
                    "stand. Ignored when det_capture=true (which has its own "
                    "pre/post schedule).",
    )
    match_physx_arg = DeclareLaunchArgument(
        "match_physx",
        default_value="default",
        description="MuJoCo solver tweaks aimed at PhysX parity. Choices: "
                    "'off' (default Euler + pyramidal cone), "
                    "'elliptic' (Euler + elliptic cone), "
                    "'implicitfast' (IMPLICITFAST integrator + elliptic), "
                    "'rk4' (RK4 + elliptic).",
    )
    mj_timestep_arg = DeclareLaunchArgument(
        "mj_timestep",
        default_value="0.005",
        description="MuJoCo physics timestep in seconds (default 0.005 s "
                    "matches Isaac sim_dt). Substep count auto-scales to "
                    "keep the 20 ms policy period.",
    )
    presettle_steps_arg = DeclareLaunchArgument(
        "presettle_steps",
        default_value="0",
        description="If > 0, run that many extra mj_steps of PD-controlled "
                    "settling AFTER the match_isaac 1-step drop and BEFORE "
                    "the policy starts. The robot lands + stabilises in "
                    "standing pose, so the viewer shows it on the floor "
                    "from frame 0. Default 0 keeps the Isaac-aligned "
                    "free-fall start needed by the deterministic harness. "
                    "200 = 1 s of physics at 5 ms/step, usually enough.",
    )

    joy = Node(
        package="joy",
        executable="joy_node",
        name="joy_node",
        output="screen",
        condition=UnlessCondition(LaunchConfiguration("fake_joy")),
    )
    fake_joy = Node(
        package="deploy_rl_policy",
        executable="fake_joy.py",
        name="fake_joy",
        output="screen",
        condition=IfCondition(LaunchConfiguration("fake_joy")),
    )

    # Prepend `vglrun -c <compress> -fps <fps>` to the simulator command when
    # `vgl:=true`. The -fps caps how often VGL itself pushes frames to the
    # client, complementing the simulator-side viewer_fps cap.
    vgl_prefix = PythonExpression([
        "'vglrun -c ", LaunchConfiguration("vgl_compress"),
        " -fps ", LaunchConfiguration("viewer_fps"),
        "' if '", LaunchConfiguration("vgl"), "'.lower() == 'true' else ''",
    ])
    mujoco_sim = Node(
        package="deploy_rl_policy",
        executable="mujoco_simulator.py",
        name="mujoco_simulator",
        output="screen",
        prefix=vgl_prefix,
        parameters=[
            {"scene": LaunchConfiguration("scene")},
            {"viewer": LaunchConfiguration("viewer")},
            {"viewer_fps": LaunchConfiguration("viewer_fps")},
            {"low_quality": LaunchConfiguration("low_quality")},
            {"match_isaac": LaunchConfiguration("match_isaac")},
            {"match_physx": LaunchConfiguration("match_physx")},
            {"mj_timestep": LaunchConfiguration("mj_timestep")},
            {"follow_robot": LaunchConfiguration("follow_robot")},
            {"presettle_steps": LaunchConfiguration("presettle_steps")},
        ],
    )

    low_level_ctrl = Node(
        package="deploy_rl_policy",
        executable="low_level_ctrl",
        name="finite_state_machine_node",
        output="screen",
        parameters=[{"is_simulation": True}],
    )

    ep_policy = Node(
        package="deploy_rl_policy",
        executable="ep_rl_policy.py",
        name="ep_rl_policy",
        output="screen",
        arguments=["--is_simulation", "True"],
        parameters=[
            {"forward_vel_cmd":       LaunchConfiguration("forward_vel_cmd")},
            {"dump_obs":              LaunchConfiguration("dump_obs")},
            {"use_sandbox_actor":     LaunchConfiguration("use_sandbox_actor")},
            {"use_ground_truth_priv": LaunchConfiguration("use_ground_truth_priv")},
            {"track_goal":            LaunchConfiguration("track_goal")},
            {"goal_dist":             LaunchConfiguration("goal_dist")},
            {"stand_for_steps":       LaunchConfiguration("stand_for_steps")},
            {"startup_wait_s":        LaunchConfiguration("startup_wait_s")},
        ],
    )

    # When vgl=true, ensure VGL_DISPLAY points at the local NVIDIA-backed
    # X server (:1 on this machine). User can override before launch.
    set_vgl_display = SetEnvironmentVariable(
        name="VGL_DISPLAY",
        value=os.environ.get("VGL_DISPLAY", ":1"),
    )

    return LaunchDescription([
        scene_arg, viewer_arg, follow_robot_arg, fake_joy_arg,
        vgl_arg, vgl_compress_arg, viewer_fps_arg, low_quality_arg,
        match_isaac_arg, forward_vel_cmd_arg, dump_obs_arg,
        use_sandbox_actor_arg, use_ground_truth_priv_arg,
        track_goal_arg, goal_dist_arg, stand_for_steps_arg, startup_wait_s_arg,
        match_physx_arg, mj_timestep_arg, presettle_steps_arg,
        set_vgl_display,
        joy, fake_joy, mujoco_sim, low_level_ctrl, ep_policy,
    ])
