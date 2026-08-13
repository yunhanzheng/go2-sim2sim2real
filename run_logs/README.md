# runs/ — logged deploy launches

Each launch writes a timestamped subfolder with all its data.

- `dryrun/<YYYY-MM-DD_HH-MM-SS>/` — inference dry-runs (NO `low_level_ctrl`, no `/lowcmd`,
  robot does not move). Produced by `deploy_mac/run_dryrun_remote.sh`.
- `real/<YYYY-MM-DD_HH-MM-SS>/`   — real runs that send `/lowcmd` (robot moves; hoisted).
  Produced by `deploy_mac/run_deploy.sh`.

Typical files per launch:
- `meta.txt`            — timestamp, mode, gains, interface, and git provenance
  (`git_commit`/`git_branch`/`git_state` clean|dirty + any untracked files)
- `git_diff.patch`      — `git diff HEAD` at launch: the exact uncommitted changes the
  run used. Empty when the tree was clean. Reproduce a run with:
  `git checkout <git_commit> && git apply runs/.../git_diff.patch`
- `low_level_ctrl.log`  — FSM node stdout (real runs)
- `wireless_to_joy.log` — remote bridge
- `lowstate.csv`        — joint q/dq + engaged + cmd_kp over time (real runs)
- `policy_log.csv`      — /rl/* policy outputs over time (dry-runs)
- `rl_policy.log` / `depth_encoder.log` / `depth_camera.log` — node logs (dry-runs)
