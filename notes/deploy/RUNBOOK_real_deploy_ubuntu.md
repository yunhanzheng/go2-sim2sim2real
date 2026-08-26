# Go2 真机部署手册 — Ubuntu（系统 ROS 2 Humble）

对应 Mac / RoboStack 版 [`RUNBOOK_real_deploy.md`](RUNBOOK_real_deploy.md)。
栈与安全规则相同，环境和启动命令按 Ubuntu 书写。

默认按仓库 README 的 **Option A**：系统 `/opt/ros/humble` + 已编译的
`unitree_ros2`（本机示例路径：`~/go2/ros2_ws`）。若路径不同请自行替换。

> ⚠️ 真机务必 **吊起 / 系绳**。没有真正的看门狗，急停只是脆弱的手柄组合键。
> 详见 [`DEPLOY.md`](../../DEPLOY.md)。

---

## 0. 一次性准备（核对即可）

- [ ] 已装 ROS 2 Humble（`ls /opt/ros/humble/setup.bash`）
- [ ] 已编译并 source `unitree_go` / `unitree_api`（如 `~/go2/ros2_ws`）
- [ ] 本仓库已编译：`cd ~/data/go2-sim2sim2real && colcon build --symlink-install`
- [ ] 训练仓库在兄弟目录：`~/data/go2-flyball-parkour`（或设置 `$EP_ROOT`）
- [ ] **系统 Python 3.10** 上有 Torch + MuJoCo（Humble 的 `rclpy` 依赖 3.10）。
      **不要**在 `parkour` conda 环境（Python 3.8）里起 ROS — 见 §9
- [ ] 已确认连 Go2 的有线网卡名（`ip -br link` / `ip addr`）
- [ ] 可选：Jetson RealSense 服务（见 `deploy_mac/jetson/INSTALL.md`；
      SSH 主机常见 `192.168.123.18` 或 `.222`）

---

## 1. 现场与机体

- [ ] 机器人吊起 / 系绳，周围空旷，手不离电源
- [ ] 深度相机线插牢（若跑深度策略）
- [ ] 相机俯仰接近训练安装角（画面里地面大约在 2/3 处淡出 ≈ 20°）

---

## 2. 网络（Ubuntu）

宇树官方做法：网线直连，电脑设 **123 网段**静态 IP。

| 设备 | 常见 IP |
| --- | --- |
| Go2 运控板 | `192.168.123.161` |
| 机载 Jetson（若有） | 常见 `192.168.123.18` |
| 本机 | 如 `192.168.123.99` 或 `.222`（**不要**用 `.161`） |

```bash
# 示例：有线网卡设静态 IPv4（把 enpXsY 换成实际网卡名）
# 图形界面：设置 → 网络 → 有线 → IPv4 → 手动
#   地址 192.168.123.99   掩码 255.255.255.0   网关留空

ping -c 3 192.168.123.161
ip -br addr | grep 192.168.123    # 记下网卡名，例如 enp3s0
```

文档：[宇树开发者中心](https://support.unitree.com/home/zh/developer)、
[`unitree_ros2` README](https://github.com/unitreerobotics/unitree_ros2)。

---

## 3. 关闭 Sport 模式

Sport 必须 **关掉**，否则出厂控制器会和 `/lowcmd` 抢控制权。

先用宇树原厂遥控器把狗放倒，再释放底层（思路与 Mac 手册相同）。Ubuntu 上用你本地的
`unitree_sdk2py` / `disable_sport` 脚本，例如：

```bash
# 示例 — 按本机 sdk2py 路径修改：
# PYTHONPATH=$HOME/go2/unitree_sdk2_python $HOME/go2/venv/bin/python \
#   $HOME/go2/disable_sport.py
```

成功标志大致为 `CheckMode: code=0` 及 release 确认。每次上电后都要再确认。

---

## 4. 环境（每个终端都要）

用**干净 shell**（不要有 `(parkour)`）。系统 Python 须为 3.10：

```bash
conda deactivate 2>/dev/null; conda deactivate 2>/dev/null
export PATH="/usr/bin:$PATH"
which python3 && python3 --version    # 应为 /usr/bin/python3 ，3.10.x

REPO=/home/ubuntu22/data/go2-sim2sim2real
cd "$REPO"

source /opt/ros/humble/setup.bash
source ~/go2/ros2_ws/install/setup.bash
source "$REPO/install/setup.bash"

export EP_ROOT=/home/ubuntu22/data/go2-flyball-parkour
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# CycloneDDS 绑到能通 192.168.123.x 的网卡（把 enp3s0 换成实际名字）：
export GO2_IFACE=enp3s0
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces>
  <NetworkInterface name=\"${GO2_IFACE}\" priority=\"default\" multicast=\"default\"/>
</Interfaces></General></Domain></CycloneDDS>"
```

若缺包：`sudo apt install ros-humble-rmw-cyclonedds-cpp`。

自检：

```bash
ros2 topic hz /lowstate                          # 约 500 Hz
ros2 topic hz /camera/depth/image_rect_raw       # 若 Jetson 深度已起，约 30 Hz
```

相机无数据 → 按 Mac/Jetson 笔记重启 RealSense 服务，等约 15 s。

---

## 5. 干跑 / 仿真预演（本机仿真，不是真机走路）

Ubuntu 没有 `run_dryrun_remote.sh`。在本机先跑通 ROS 栈可用仿真：

```bash
ros2 launch deploy_rl_policy distill.launch.py is_simulation:=true \
  use_keyboard:=true \
  ckpt:=models/student/model_10000.pt \
  scene_xml:=resources/go2/scene_flyball30_2h_h13_noffset.xml
```

真机在未吊起前：**不要**按 LB+RB / 空格 engage。

---

## 6. 真机先做 lay ↔ stand（最安全的一步）

先验证 FSM + PD，**再**加载行走策略。吊起。此步 **不要 engage**（不要 LB+RB / 空格）。

```bash
# 完成 §4 source 之后 — 示例：只起 low_level_ctrl + 手柄桥
# 用宇树遥控器 → /joy（wireless_to_joy 或 joy_node），然后：
ros2 run deploy_rl_policy low_level_ctrl --ros-args \
  -p is_simulation:=false \
  -p policy_kp:=35.0 -p policy_kd:=0.6
```

遥控 / 键盘映射（与 README 一致）：

| 输入 | 动作 |
| --- | --- |
| **B** | 趴下 → 站起 |
| **A** | 趴下 |
| 按住 **LB+RB**（键盘：按住 **空格**） | engage 策略 |
| 摇杆前进 / **W** | 走（仅 engage 后） |
| **L2+R2** / 退出组合 | 停控制器 |

若用 `use_keyboard:=true`：焦点放在 **launch 终端**，不要点 MuJoCo 窗口
（viewer 会抢走 W，画面容易「变黑」）。需安装：
`sudo apt install python3-pynput`。

---

## 7. 真机策略栈（`is_simulation:=false`）

吊起 / 系绳。Sport 已关。若学生策略需要深度，确认深度话题在发。

**示例 — flyball / 走廊学生（非平地）：**

```bash
ros2 launch deploy_rl_policy distill.launch.py is_simulation:=false \
  ckpt:=models/student/model_10000.pt \
  env_class_eq_17:=0 \
  action_delay:=1 \
  binary_vel:=1 \
  lin_vel_max:=1.0 \
  policy_kp:=35.0 \
  policy_kd:=0.6
```

**示例 — L00-02 平地 baseline（标志不同）：**

```bash
ros2 launch deploy_rl_policy distill.launch.py is_simulation:=false \
  ckpt:=models/L00-02-flat-baseline-delay-distillation-noised/model_14500.pt \
  env_class_eq_17:=1 \
  action_delay:=1
```

真机默认会提高 `contact_threshold`（按脚的原始传感器单位）。**不要**把仿真里的
`2.0` 原样拷到真机，先读
[`contact_threshold_calibration_2026-07-24.md`](../analysis/contact_threshold_calibration_2026-07-24.md)。

操作顺序：

1. 唤醒宇树遥控器（休眠则没有 `/wirelesscontroller`）
2. **B** → 站起；站稳约 2 s
3. 按住 **LB+RB** → engage
4. 摇杆前进 → 走
5. 松开 LB+RB → 站；**A** → 趴；硬停：断电 / L2+R2

---

## 8. 两次运行之间清理

```bash
bash "$REPO/deploy_mac/kill_ros.sh"
```

残留节点会占着 DDS topic。每次 Ctrl-C 或启动失败后都跑一遍。

---

## 9. Ubuntu 常见坑（本机 bring-up 实录）

| 现象 | 处理 |
| --- | --- |
| `No module named 'rclpy._rclpy_pybind11'` | 还在 `(parkour)` / PATH 上是 Python 3.8。退出 conda；`PATH=/usr/bin:$PATH` |
| `keyboard_to_joy` 立刻退出 | `sudo apt install python3-pynput` |
| B 能站、W 画面变黑 | 焦点在 MuJoCo，W 被当成相机键。点回 launch 终端；先按住空格再按 W |
| README 里 `τ_peak < 35 Nm` | 平地走路冒烟标准，不是 flyball 成功标准 / 硬件绝对上限 |
| checkpoint 标志 | 错的 `env_class_eq_17` / contact / yaw **不会报错**，只会默默走坏 |

`deploy_mac/run/` 下的 Mac 脚本在此可选；优先直接 `ros2 launch` + §4 环境。
Jetson 校时 / RealSense 服务仍可参考 `deploy_mac/jetson/` 与 Mac 手册中机器人侧章节。

---

## 10. 禁止事项

- 地面未确认 lay↔stand、场地未清空前，不要 engage
- Sport 开着时不要发 `/lowcmd`
- 真机 DDS 不要用 `GO2_IFACE=lo`
- 不要在 `parkour`（py3.8）里跑 ROS launch
- 两次会话之间不要跳过 `kill_ros.sh`
