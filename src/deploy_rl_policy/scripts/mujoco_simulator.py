#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import mujoco.viewer
import mujoco
import numpy as np
from unitree_go.msg import LowState,LowCmd
from pathlib import Path
from xbox_command import XboxController
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
import threading
import time
from std_msgs.msg import Float32MultiArray
project_root=Path(__file__).parents[4]
class MujocoSimulator(Node):
    def __init__(self):
        super().__init__("mujoco_simulator")
        self.declare_parameter("scene_xml", "")
        self.declare_parameter("init_standing", False)
        self.declare_parameter("init_lay", False)
        self.declare_parameter("viewer", True)
        self.declare_parameter("lockstep", False)
        self.declare_parameter("decimation", 4)
        self.declare_parameter("friction_scale", 1.0)
        self.declare_parameter("initial_x", 0.0)
        scene_param = self.get_parameter("scene_xml").get_parameter_value().string_value
        self.init_standing = bool(self.get_parameter("init_standing").value)
        self.init_lay = bool(self.get_parameter("init_lay").value)
        self.use_viewer = bool(self.get_parameter("viewer").value)
        self.lockstep = bool(self.get_parameter("lockstep").value)
        self.get_logger().info(f"lockstep={self.lockstep} "
                               f"({'4 substeps per /target_torque' if self.lockstep else 'free-running wall-clock thread'})")
        self.decimation = int(self.get_parameter("decimation").value)
        self.friction_scale = float(self.get_parameter("friction_scale").value)
        self.initial_x = float(self.get_parameter("initial_x").value)

        self.cmd_sub = XboxController(self)
        # Resets qpos to the standalone distill_runner.py initial conditions
        # (z=0.42 in air, joints at DEFAULT_DOF_POLICY, qvel=0) on the rising
        # edge of /policy_engaged. Why: with FSM lay→stand→engage the robot
        # arrives at engagement at the kp=50 standing PD equilibrium —
        # qpos[7:19] is offset 0.05–0.15 rad from DEFAULT_DOF_POLICY and all
        # four feet are planted. The L00-02 policy was trained on episodes
        # starting from in-air drop (qpos exactly = DEFAULT, no contact yet),
        # not from this settled-standing transition, and produces divergent
        # commands when given the off-distribution startup proprio. Snapping
        # at the engage edge puts the policy in its training initial-state
        # distribution. Sim-only — for the real robot this needs a proper
        # warm-up phase or retraining on standing→walking starts.
        self.declare_parameter("snap_qpos_on_engage", True)
        self._snap_qpos_on_engage = bool(self.get_parameter("snap_qpos_on_engage").value)
        self.declare_parameter("snap_qpos_mode", "air")  # "air" or "joints"
        self._snap_qpos_mode = str(self.get_parameter("snap_qpos_mode").value)
        self._was_engaged = False

        # --- lift test ------------------------------------------------------
        # Sequence (no RL engage; robot just holds the FSM standing pose):
        #   stand up -> standing baseline -> ramp base +lift_height -> hold
        #   lifted -> ramp back down -> hold grounded -> release. The base is
        #   pinned (qpos[0:7] overwritten, qvel[0:6]=0) during the up/down/hold
        #   phases while the legs stay under low_level_ctrl's standing PD, so
        #   the feet leave the floor (foot force -> 0) and return on the way
        #   down. Phase index is published on /lift_phase for the recorder.
        self.declare_parameter("lift_test", False)
        self.declare_parameter("lift_height", 0.10)          # meters off the floor
        self.declare_parameter("lift_stand_z_thresh", 0.26)  # base_z above this => stood
        self.declare_parameter("lift_stand_hold_s", 1.5)     # standing baseline before lift
        self.declare_parameter("lift_up_s", 2.0)             # seconds held lifted
        self.declare_parameter("lift_down_s", 2.0)           # seconds held back on the ground
        self.declare_parameter("lift_ramp_s", 0.25)          # ramp up/down duration
        self.lift_test = bool(self.get_parameter("lift_test").value)
        self.lift_height = float(self.get_parameter("lift_height").value)
        self.lift_stand_z_thresh = float(self.get_parameter("lift_stand_z_thresh").value)
        self.lift_stand_hold_s = float(self.get_parameter("lift_stand_hold_s").value)
        self.lift_up_s = float(self.get_parameter("lift_up_s").value)
        self.lift_down_s = float(self.get_parameter("lift_down_s").value)
        self.lift_ramp_s = float(self.get_parameter("lift_ramp_s").value)
        # phase: 0 pre, 1 stand_hold, 2 lifting, 3 lifted, 4 lowering, 5 grounded, 6 done
        self._lift_phase = 0
        self._lift_phase_steps = 0
        self._lift_stable_count = 0
        self._lift_pin = None    # (x, y, z, quat[4]) captured when the robot stands
        self.low_state_puber=self.create_publisher(LowState,"/mujoco/lowstate",10)
        self.pos_pub=self.create_publisher(Float32MultiArray,"/mujoco/pos",10)
        self.force_pub=self.create_publisher(Float32MultiArray,"/mujoco/force",10)
        self.torque_pub=self.create_publisher(Float32MultiArray,"/mujoco/torque",10)
        # Body-frame ground-truth base linear velocity for sim2sim probing —
        # subscribers can replace the learned Estimator output on the priv
        # slice to isolate whether the velocity input is the dominant gap.
        self.sim_truth_vel_pub=self.create_publisher(Twist,"/sim_truth/base_velocity",10)
        # IMU + per-foot contact force magnitudes — feeds the standalone
        # base_velocity_estimator (EKF) node so we can compare EKF vs learned
        # estimator vs sim-truth.
        self.imu_pub=self.create_publisher(Imu,"/imu/data",10)
        self.contact_force_pub=self.create_publisher(Float32MultiArray,"/contact",10)
        self.lift_phase_pub=self.create_publisher(Float32MultiArray,"/lift_phase",10)
        self.target_torque_suber=self.create_subscription(LowCmd,"/mujoco/lowcmd",self.target_torque_callback,10)
        from std_msgs.msg import Bool
        self.create_subscription(Bool, "/policy_engaged", self._on_engaged, 10)

        self.step_counter = 0
        self.xml_path = Path(scene_param) if scene_param else (project_root/"resources"/"go2"/"scene_terrain.xml")
        # Initialize Mujoco
        self.init_mujoco()
        # Shared state read by sim_thread; updated atomically (np.array ref swap)
        # in target_torque_callback to avoid the multi-thread races that previously
        # destabilised the integration when running in the distillation deploy loop.
        self.target_dof_pos = np.zeros(12, dtype=np.float64)
        self.tau = np.zeros(12, dtype=np.float64)
        self.kps = np.full(12, 30.0)
        self.kds = np.full(12, 0.75)
        # Lock protecting mj_step + qpos mutation. The on_engaged callback runs
        # a tight snap-and-settle loop; without the lock, sim_thread can call
        # mj_step concurrently on the same MjData while the GIL is released
        # inside MuJoCo C, corrupting data and segfaulting.
        self._sim_lock = threading.Lock()
        self._reset_wall_clock = threading.Event()
        # Load params
        self.timer = self.create_timer(0.002, self.publish_sensor_data)
        self.running=True
        self.recieve_data=False
        if self.lockstep:
            # In lockstep mode, the simulator advances inside target_torque_callback;
            # no free-running thread.
            self.sim_thread = None
        else:
            self.sim_thread=threading.Thread(target=self.step_simulation)
            self.sim_thread.start()

    def init_mujoco(self):
        """Initialize Mujoco model and data"""

        self.m = mujoco.MjModel.from_xml_path(str(self.xml_path))
        if self.friction_scale != 1.0:
            self.m.geom_friction[:, 0] *= self.friction_scale
            self.get_logger().info(
                f"[mj] friction_scale={self.friction_scale} → sliding friction × {self.friction_scale}"
            )
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = 0.005

        # Foot collision-geom ids (go2.xml: <geom name="FR" class="foot"/> etc.)
        # in policy/SDK order FR,FL,RR,RL, for computing NET ground-contact force
        # (see _foot_contact_forces) — matches Isaac's contact_forces[feet].
        self._foot_order = ("FR", "FL", "RR", "RL")
        self._foot_geom_ids = []
        for nm in self._foot_order:
            gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, nm)
            if gid < 0:
                self.get_logger().warning(f"[mj] foot geom '{nm}' not found in scene")
            self._foot_geom_ids.append(gid)
        self._foot_contact_cache = np.zeros(4, dtype=np.float64)  # FR,FL,RR,RL net contact (N)
        self._foot_force_int16_warned = False

        if self.init_standing:
            # Standing pose with identity quat at training spawn height (0.42 m).
            # Default joint angles in MJ actuator order (FR,FL,RR,RL × hip,thigh,calf).
            default_dof_mj = np.array(
                [-0.1, 0.8, -1.5,   # FR
                  0.1, 0.8, -1.5,   # FL
                 -0.1, 1.0, -1.5,   # RR
                  0.1, 1.0, -1.5],  # RL
                dtype=np.float64,
            )
            self.d.qpos[:3] = [self.initial_x, 0.0, 0.42]  # configurable x; Isaac z=0.42
            self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.d.qpos[7:7+12] = default_dof_mj
            self.d.qvel[:] = 0.0
            mujoco.mj_forward(self.m, self.d)
        elif self.init_lay:
            # Curled lay pose, clamped within the URDF joint ranges so the
            # calves don't try to over-bend past -2.7227. Matches the
            # laydown_angels_ in low_level_ctrl.hpp (hips at ±0.1 == standing,
            # not the stock Unitree ±0.34 rear hips) so the FSM detects "lay"
            # immediately and starts holding the pose instead of dragging the
            # joints elsewhere. MJ/SDK order: FR,FL,RR,RL × (hip, thigh, calf).
            lay_dof_mj = np.array(
                [-0.1,  1.26, -2.72,
                  0.1,  1.25, -2.72,
                 -0.1,  1.29, -2.72,
                  0.1,  1.28, -2.72],
                dtype=np.float64,
            )
            # Spawn just above the natural lay settle (~0.06 m). 0.05 puts the
            # body collision box slightly under the floor and the contact
            # recovery impulse leaves the joints in a state PD can't escape;
            # 0.10 visibly drops; 0.07 lands cleanly. The pre-settle below
            # absorbs the tiny remaining vertical impulse so the recorder's
            # first frame already shows a still robot in the lay pose.
            self.d.qpos[:3] = [self.initial_x, 0.0, 0.07]
            self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.d.qpos[7:7+12] = lay_dof_mj
            self.d.qvel[:] = 0.0
            mujoco.mj_forward(self.m, self.d)
            # Pre-settle 0.5 s with PD holding the lay pose, so the contact
            # transient is gone before sim_thread / publishers spin up.
            kp_lay = np.full(12, 50.0)
            kd_lay = np.full(12, 1.0)
            for _ in range(100):  # 100 × 5 ms = 0.5 s
                tau = kp_lay * (lay_dof_mj - self.d.qpos[7:7+12]) - kd_lay * self.d.qvel[6:6+12]
                self.d.ctrl[:] = tau
                mujoco.mj_step(self.m, self.d)
            self.get_logger().info(
                f"[init_lay] pre-settled at base_x={float(self.d.qpos[0]):.3f} "
                f"base_z={float(self.d.qpos[2]):.3f}"
            )

        if self.use_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
            cam_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "follow_cam")
            if cam_id >= 0:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.viewer.cam.fixedcamid = cam_id
        else:
            self.viewer = None
        print("Number of qpos:", self.m.nq)
        print("Joint order:")
        for i in range(self.m.njnt):
            print(f"{i}: {self.m.joint(i).name}")
   
        
    def _on_engaged(self, msg):
        engaged = bool(msg.data)
        if engaged and not self._was_engaged and self._snap_qpos_on_engage:
            # Snap mode controlled by snap_qpos_mode param:
            #   "air"   : full standalone init — z=0.42 in air, joints=DEFAULT, qvel=0
            #             (robot drops then walks; matches distill_runner.py:300-305)
            #   "joints": joints snapped to DEFAULT, qvel=0, but base z kept where it
            #             is (robot stays planted but proprio[13:25] = 0 exactly).
            #             Isolates whether the dpos asymmetry from FSM stand-up is
            #             the actual bug — should give the policy training-matched
            #             tick-0 proprio without lifting the body.
            default_dof_mj = np.array(
                [-0.1, 0.8, -1.5,   0.1, 0.8, -1.5,
                 -0.1, 1.0, -1.5,   0.1, 1.0, -1.5],
                dtype=np.float64,
            )
            mode = self._snap_qpos_mode
            if mode == "air":
                # Hold the lock for the snap + settle so sim_thread doesn't
                # call mj_step concurrently (corrupted MjData → segfault).
                with self._sim_lock:
                    # Clear contact-solver warm-start so the post-snap
                    # equilibrium is determined fresh (otherwise the lay→stand
                    # path's solution biases toward a non-symmetric config).
                    self.d.qacc_warmstart[:] = 0.0
                    self.d.qpos[:3] = [float(self.d.qpos[0]), float(self.d.qpos[1]), 0.42]
                    self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
                    self.d.qpos[7:7+12] = default_dof_mj
                    self.d.qvel[:] = 0.0
                    mujoco.mj_forward(self.m, self.d)
                    # Post-snap settle: hold standing PD for 1 s so the body
                    # drops from z=0.42, lands, and reaches equilibrium BEFORE
                    # the policy starts reading /lowstate. Mirrors standalone
                    # distill_runner's --pre_settle_seconds path.
                    kp_air = np.full(12, 50.0)
                    kd_air = np.full(12, 1.0)
                    n_settle = int(1.0 / self.m.opt.timestep)  # 1 s = 200 substeps
                    for _ in range(n_settle):
                        tau = kp_air * (default_dof_mj - self.d.qpos[7:7+12]) - kd_air * self.d.qvel[6:6+12]
                        self.d.ctrl[:] = tau
                        mujoco.mj_step(self.m, self.d)
                self.get_logger().info(
                    f"[mj_sim] /policy_engaged rising edge — air snap + 1s settle "
                    f"(base_z={float(self.d.qpos[2]):.3f}, |qvel|≈{float(np.linalg.norm(self.d.qvel)):.4f})"
                )
            elif mode == "joints":
                with self._sim_lock:
                    self.d.qpos[7:7+12] = default_dof_mj
                    self.d.qvel[:] = 0.0
                    mujoco.mj_forward(self.m, self.d)
                self.get_logger().info(
                    f"[mj_sim] /policy_engaged rising edge — joints snap "
                    f"(qpos[7:19]=DEFAULT, qvel=0; base_z={float(self.d.qpos[2]):.3f} kept)"
                )
            else:
                self.get_logger().warning(f"unknown snap_qpos_mode={mode}; skip")
        if engaged and not self._was_engaged:
            self._reset_wall_clock.set()
        self._was_engaged = engaged

    def target_torque_callback(self, msg):
        # Atomic ref swap so sim_thread never sees a half-updated target/kp/kd.
        target = np.fromiter((msg.motor_cmd[i].q for i in range(12)), dtype=np.float64, count=12)
        kp = np.fromiter((msg.motor_cmd[i].kp for i in range(12)), dtype=np.float64, count=12)
        kd = np.fromiter((msg.motor_cmd[i].kd for i in range(12)), dtype=np.float64, count=12)
        self.target_dof_pos = target
        self.kps = kp
        self.kds = kd
        self.recieve_data = True
        if self.lockstep:
            # Mirror distill_runner.py:443-447: 4 PD-recomputed substeps per policy tick.
            # Locks sim cadence to policy cadence (50 Hz × 4 = 200 Hz sim).
            self._step_decimation()

    def _step_decimation(self):
        for _ in range(self.decimation):
            tau = self.kps * (self.target_dof_pos - self.d.qpos[7:7+12]) - self.kds * self.d.qvel[6:6+12]
            self.d.ctrl[:] = tau
            mujoco.mj_step(self.m, self.d)
        self.tau = tau
        self._foot_contact_cache = self._foot_contact_forces()
        if self.viewer is not None:
            self.viewer.sync()

    def _foot_contact_forces(self):
        """Per-foot NET ground-contact force magnitude (N) in policy order
        FR,FL,RR,RL. Sums the world-frame contact-force vectors over each foot
        collision geom, matching Isaac's torch.norm(contact_forces[feet]):
        ~0 in swing, body weight in stance. Replaces the go2.xml <force> sensor,
        which measured the knee-joint reaction (calf self-weight floor ~2.4 N
        that never drops to 0 in swing — corrupts the contact threshold)."""
        net = np.zeros((4, 3), dtype=np.float64)
        buf = np.zeros(6, dtype=np.float64)
        for ci in range(self.d.ncon):
            c = self.d.contact[ci]
            for fi, gid in enumerate(self._foot_geom_ids):
                if gid >= 0 and (c.geom1 == gid or c.geom2 == gid):
                    mujoco.mj_contactForce(self.m, self.d, ci, buf)
                    # buf[:3] is the force in the contact frame; rotate to world
                    # (frame rows are the world-frame contact axes) so multiple
                    # contacts on one foot sum correctly.
                    net[fi] += c.frame.reshape(3, 3).T @ buf[:3]
                    break
        return np.linalg.norm(net, axis=1)

    def step_simulation(self):
        first_step_walltime = None
        next_step_deadline = None
        step_count = 0
        while self.running and (self.viewer is None or self.viewer.is_running()):
            if not self.recieve_data:
                next_step_deadline = None
                time.sleep(0.001)
                continue
            if self._reset_wall_clock.is_set():
                self._reset_wall_clock.clear()
                next_step_deadline = None
                first_step_walltime = None
                step_count = 0

            now = time.monotonic()
            if next_step_deadline is None:
                next_step_deadline = now
            elif now < next_step_deadline:
                time.sleep(next_step_deadline - now)
            step_start = time.monotonic()
            # PD compute + ctrl write + mj_step are atomic in this thread,
            # mirroring distill_runner.py's decimation loop. Eliminates the race
            # where update_tau / target_torque_callback ran on the executor
            # thread mid-step.
            with self._sim_lock:
                target = self.target_dof_pos
                kps = self.kps
                kds = self.kds
                tau = kps * (target - self.d.qpos[7:7+12]) - kds * self.d.qvel[6:6+12]
                self.tau = tau
                self.d.ctrl[:] = tau
                mujoco.mj_step(self.m, self.d)
                if self.lift_test:
                    self._update_lift()
                # Net per-foot contact force (after any lift pin + mj_forward),
                # cached under the lock so publish_sensor_data reads a consistent
                # snapshot instead of racing mj_contactForce against mj_step.
                self._foot_contact_cache = self._foot_contact_forces()
            Torque = Float32MultiArray()
            Torque.data = tau.tolist()
            self.torque_pub.publish(Torque)
            if self.viewer is not None:
                self.viewer.sync()
            step_count += 1
            if first_step_walltime is None:
                first_step_walltime = step_start
            next_step_deadline += float(self.m.opt.timestep)
            if step_count % 200 == 0:
                wall = time.monotonic() - first_step_walltime
                sim = step_count * self.m.opt.timestep
                self.get_logger().info(
                    f"[mj_sim] sim_t={sim:5.2f}s  wall_t={wall:5.2f}s  ratio={sim/max(wall,1e-6):.2f}  base_z={float(self.d.qpos[2]):.3f}"
                )
            
    def _set_lift_phase(self, ph):
        self._lift_phase = ph
        self._lift_phase_steps = 0

    def _update_lift(self):
        """Drive the lift-test state machine. Called once per mj_step, inside
        the sim lock. Pins the base pose during the up/down/hold phases and
        re-runs mj_forward so the foot-force sensors reflect the pinned state.
        """
        ts = float(self.m.opt.timestep)
        z = float(self.d.qpos[2])
        ph = self._lift_phase

        if ph == 0:  # pre: wait until the robot has stood and is stable
            vel = float(np.linalg.norm(self.d.qvel[:6]))
            if z > self.lift_stand_z_thresh and vel < 0.15:
                self._lift_stable_count += 1
            else:
                self._lift_stable_count = 0
            if self._lift_stable_count >= int(round(0.3 / ts)):  # 0.3 s stable
                self._lift_pin = (float(self.d.qpos[0]), float(self.d.qpos[1]),
                                  z, self.d.qpos[3:7].copy())
                self._set_lift_phase(1)
                self.get_logger().info(
                    f"[lift] stood at base_z={z:.3f}; holding standing baseline "
                    f"{self.lift_stand_hold_s:.1f}s")
            return

        self._lift_phase_steps += 1
        t = self._lift_phase_steps * ts

        if ph == 1:  # standing baseline, no pin
            if t >= self.lift_stand_hold_s:
                self._set_lift_phase(2)
                self.get_logger().info(
                    f"[lift] lifting base +{self.lift_height*100:.0f}cm off the floor")
            return

        px, py, pz, pq = self._lift_pin
        H, R = self.lift_height, self.lift_ramp_s
        if ph == 2:      # ramp up
            z_t = pz + H * (min(1.0, t / R) if R > 0 else 1.0)
            if t >= R:
                self._set_lift_phase(3)
        elif ph == 3:    # hold lifted
            z_t = pz + H
            if t >= self.lift_up_s:
                self._set_lift_phase(4)
                self.get_logger().info("[lift] lowering base back to the ground")
        elif ph == 4:    # ramp down
            z_t = pz + H * (1.0 - (min(1.0, t / R) if R > 0 else 1.0))
            if t >= R:
                self._set_lift_phase(5)
        elif ph == 5:    # hold grounded
            z_t = pz
            if t >= self.lift_down_s:
                self._set_lift_phase(6)
                self.get_logger().info("[lift] done; releasing base")
                return
        else:            # ph == 6: released, free dynamics
            return

        # Pin the base pose and refresh contacts/sensors for this step.
        self.d.qpos[0] = px
        self.d.qpos[1] = py
        self.d.qpos[2] = z_t
        self.d.qpos[3:7] = pq
        self.d.qvel[:6] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def stop_simulation(self):
        self.running=False
        if self.sim_thread is not None:
            self.sim_thread.join()


    def publish_sensor_data(self):
        # ROS publishers run on the executor while physics runs on a worker.
        # Snapshot one complete MuJoCo state so a message never mixes values
        # from opposite sides of mj_step.
        with self._sim_lock:
            qpos = self.d.qpos.copy()
            qvel = self.d.qvel.copy()
            sensordata = self.d.sensordata.copy()
            tau = self.tau.copy()
            foot_contact = self._foot_contact_cache.copy()
            lift_phase = self._lift_phase

        low_state_msg=LowState()
        for i in range(12):
            low_state_msg.motor_state[i].q=qpos[7+i]
            low_state_msg.motor_state[i].dq=qvel[6+i]
            # Go2 LowState exposes MotorState.tau_est for every actuator.
            low_state_msg.motor_state[i].tau_est=float(tau[i])
        low_state_msg.imu_state.quaternion=qpos[3:7].astype(np.float32)
        low_state_msg.imu_state.gyroscope=sensordata[40:43].astype(np.float32)
        # Go2 LowState provides scalar int16 foot_force and foot_force_est
        # values, not a 3-D contact vector. Publish the quantized resultant
        # contact load in SDK order. MuJoCo has no separate onboard estimator,
        # so both fields carry the same proxy.
        ff_mj = foot_contact
        for i in range(4):
            raw_value = int(round(float(ff_mj[i])))
            value = min(max(raw_value, 0), np.iinfo(np.int16).max)
            if value != raw_value and not self._foot_force_int16_warned:
                self.get_logger().warning(
                    f"foot force {raw_value} N exceeds LowState int16 range; "
                    f"publishing {value} while retaining full precision on /sim_truth/contact_forces")
                self._foot_force_int16_warned = True
            low_state_msg.foot_force[i] = value
            low_state_msg.foot_force_est[i] = value
        self.low_state_puber.publish(low_state_msg)
        pos=Float32MultiArray()
        pos.data=qpos[:19].tolist()
        self.pos_pub.publish(pos)

        # /sim_truth/base_velocity: MuJoCo qvel[0:3] is world-frame linear vel
        # of the free joint. Rotate to body frame via R^T (where R is body→world
        # from the base quat qpos[3:7] = [w,x,y,z]) so subscribers can compare
        # apples to apples with training's `self.base_lin_vel`.
        vel_world = qvel[0:3]
        q = qpos[3:7]  # [w, x, y, z]
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        # R^T columns are the world-frame basis vectors of the body frame.
        r00 = 1 - 2*(y*y + z*z); r01 = 2*(x*y + z*w); r02 = 2*(x*z - y*w)
        r10 = 2*(x*y - z*w); r11 = 1 - 2*(x*x + z*z); r12 = 2*(y*z + x*w)
        r20 = 2*(x*z + y*w); r21 = 2*(y*z - x*w); r22 = 1 - 2*(x*x + y*y)
        vx = r00*vel_world[0] + r01*vel_world[1] + r02*vel_world[2]
        vy = r10*vel_world[0] + r11*vel_world[1] + r12*vel_world[2]
        vz = r20*vel_world[0] + r21*vel_world[1] + r22*vel_world[2]
        twist = Twist()
        twist.linear.x = float(vx); twist.linear.y = float(vy); twist.linear.z = float(vz)
        self.sim_truth_vel_pub.publish(twist)

        # /imu/data: feed the EKF the body-frame linear acceleration *with*
        # gravity restored — the EKF subtracts its own bias (averaged over
        # 1000 stationary samples) which absorbs the gravity term. MuJoCo
        # accelerometer (sensordata[43:46]) is specific force in body frame
        # (≈ -g_body when stationary), so a + g_body cancels gravity, leaving
        # true body-frame linear acceleration. Replicates the convention used
        # in base_velocity_estimator/scripts/mujoco_simulator.py.
        accel_body = sensordata[43:46]
        # Body-frame gravity acceleration vector = 9.81 * projected_gravity
        # where projected_gravity = R^T @ [0,0,-1] = (-r02, -r12, -r22). When
        # the robot is upright (R=I) this gives [0,0,-9.81] as expected.
        gx = -9.81 * r02
        gy = -9.81 * r12
        gz = -9.81 * r22
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.linear_acceleration.x = float(accel_body[0] + gx)
        imu_msg.linear_acceleration.y = float(accel_body[1] + gy)
        imu_msg.linear_acceleration.z = float(accel_body[2] + gz)
        # Gyro (body-frame angular velocity) lives in sensordata[40:43].
        gyro = sensordata[40:43]
        imu_msg.angular_velocity.x = float(gyro[0])
        imu_msg.angular_velocity.y = float(gyro[1])
        imu_msg.angular_velocity.z = float(gyro[2])
        self.imu_pub.publish(imu_msg)

        # /contact: 4-vec of per-foot NET contact-force magnitudes in FL,FR,RL,RR
        # order (matches go2.xml sensor declaration order, which is also what the
        # base_velocity_estimator EKF and distill_recorder assume). Reorder the
        # FR,FL,RR,RL cache via [1,0,3,2].
        cmsg = Float32MultiArray()
        cmsg.data = foot_contact[[1, 0, 3, 2]].astype(np.float32).tolist()
        self.contact_force_pub.publish(cmsg)

        # /lift_phase: current lift-test phase index (0 when lift_test off).
        lp = Float32MultiArray()
        lp.data = [float(lift_phase)]
        self.lift_phase_pub.publish(lp)
    
    @staticmethod
    def pd_control(target_q, q, kp, dq, kd):
        """Calculates torques from position commands"""
        torques=(target_q - q) * kp -  dq * kd
        return torques
    
def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_simulation()
        if node.viewer is not None:
            node.viewer.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
