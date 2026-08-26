# notes/

Written analyses and the standing deploy procedure. `deploy/` holds procedures,
`analysis/` the investigations that produced the current parameter choices.

These notes cite the raw sweep rows and render directories they were derived from
(`studies/`, `renders/`, `run_logs/`). Those artifacts are regenerable output and are
not shipped — the notes carry the analysed result.

## deploy/ — Standing deploy procedure

- [`RUNBOOK_real_deploy.md`](deploy/RUNBOOK_real_deploy.md) — Go2 真机部署手册（Mac / fused 栈）
- [`RUNBOOK_real_deploy_ubuntu.md`](deploy/RUNBOOK_real_deploy_ubuntu.md) — Ubuntu（系统 ROS Humble）中文版

## analysis/ — Investigations and measurements

- [`contact_threshold_calibration_2026-07-24.md`](analysis/contact_threshold_calibration_2026-07-24.md) — Per-foot contact-threshold calibration; why sim and real differ
- [`debug_2026-07-24_depth_camera_fov.md`](analysis/debug_2026-07-24_depth_camera_fov.md) — Depth-camera near-clip and FOV alignment
- [`depth_image_identity_proof.md`](analysis/depth_image_identity_proof.md) — Proof that the deploy depth image matches the training render
- [`drivetrain_armature_friction_sweep_2026-07-28.md`](analysis/drivetrain_armature_friction_sweep_2026-07-28.md) — Reflected inertia (armature) vs. dry friction (frictionloss): the tolerance cliff
- [`sim2real_divergence_camera_pitch_2026-07-27.md`](analysis/sim2real_divergence_camera_pitch_2026-07-27.md) — Why real runs failed while sim2sim passed: camera pitch
- [`standalone_vs_ros_latency_divergence_2026-07-28.md`](analysis/standalone_vs_ros_latency_divergence_2026-07-28.md) — Why the ROS stack outperformed the standalone runner on the same student
- [`unitree_go2_lowlevel_limits_research_2026-07-24.md`](analysis/unitree_go2_lowlevel_limits_research_2026-07-24.md) — Go2 low-level control: torque limits, command rate, foot-force sensors
