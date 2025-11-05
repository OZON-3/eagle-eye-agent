import numpy as np
import pybullet as p
from collections import deque
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class TargetAviary(BaseRLAviary):
    """
    Single-agent RL: follow a moving target (position + yaw alignment).
    - Obs, Act spaces identical to BaseRLAviary (KIN/RGB, RPM/VEL/etc.).
    - Target follows a simple parametric trajectory; you can switch to static.
    """

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui: bool = False,
                 record: bool = False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.VEL,
                 # Target/episode params
                 episode_len_sec: int = 60,
                 target_static: bool = True,
                 # replace these three defaults in __init__ signature
                 target_center=(3, 1.0, 1.0),  # ~1.5 m in front of the origin
                 target_radius: float = 0.4,  # small circle
                 hold_time_sec: float = 0.3,  # how long to stay inside radius
                 near_speed_mps: float = 0.15,  # must be slow to count as 'stopped'
                 takeoff_alt_m: float = 0.6,  # small early reward once above this
                 target_omega_hz: float = 0.0,  # slow: ~1 revolution every ~33 s

                 target_height_wave: bool = False,
                 yaw_align_weight: float = 0.3,
                 smooth_action_weight: float = 0.02,
                 success_radius: float = 0.25,
                 show_target: bool = True,
                 camera_mode: str = "follow"):
        """
        Args mirror HoverAviary + a few extras to control the target & reward.
        """
        # --- Target config ---
        self.EPISODE_LEN_SEC = int(episode_len_sec)
        self.TARGET_STATIC = bool(target_static)
        self.TARGET_CENTER = np.array(target_center, dtype=float)
        self.TARGET_RADIUS = float(target_radius)
        self.TARGET_OMEGA = 0.0
        # self.TARGET_OMEGA = 2 * np.pi * float(target_omega_hz)  # rad/s
        self.TARGET_HEIGHT_WAVE = bool(target_height_wave)
        self.YAW_ALIGN_W = float(yaw_align_weight)
        self.SMOOTH_W = float(smooth_action_weight)
        self.SUCCESS_R = float(success_radius)
        self.SHOW_TARGET = bool(show_target)
        self.CAMERA_MODE = str(camera_mode).lower()
        # ---- action smoothing state (for vel actions) ----
        self._act_prev = np.zeros(4, dtype=np.float32)  # same as act_dim
        self._act_tau = 0.08  # first-order lag [s] (0.05–0.12 is fine)
        self._act_rate = 6.0  # max change per second in action units (slew cap)
        self._enable_binning = True  # toggle if you want to disable later
        self._bin_cfg = {
            "binned_channels": (0, 1),  # vx, vy
            "bins_per_channel": (20, 20),  # 20 levels each
            "ranges": ((-0.6, 0.6), (-0.6, 0.6)),  # same range as clip_action
        }
        # --- Reward shaping (bounded & small) ---
        self.ALIVE_BONUS = 0.01
        self.DIST_SCALE = 2.0  # meters, for exp-shaped distance
        self.W_DIST = 1.0
        self.W_YAW = 0.10  # set 0.0 if you don't want yaw reward
        self.SMOOTH_W = 0.005
        self.EFFORT_W = 0.002
        self.SUCCESS_HOLD_STEPS = 10  # must stay inside radius for K steps
        self._crash_penalty = 0.0
        self.YAW_ERR_BETA = 2.0  # sharpness of yaw penalty/bonus
        self.W_YAW_ERR = 0.12  # weight of yaw alignment reward

        # --- Episode/curriculum bookkeeping ---
        self._inside_counter = 0
        self._episodes_done = 0
        self._last_success = False
        self._success_window = deque(maxlen=200)  # moving window of last 200 eps

        self.HOLD_STEPS = int(hold_time_sec * ctrl_freq)
        self.NEAR_SPEED = float(near_speed_mps)
        self.TAKEOFF_ALT = float(takeoff_alt_m)
        self._inside_counter = 0
        self._last_d = None
        self._takeoff_given = False

        # jitter limits (max range we’ll allow later)
        self._JITTER_MAX_XY = 2.5  # meters
        self._JITTER_MAX_Z = 1.0  # meters
        self._last_dists = []  # for stagnation
        self._stagn_window = int(2.0 * ctrl_freq)  # 2 seconds
        self._stagn_eps = 0.01
        self._prev_action = np.zeros(4, dtype=float)  # last action for smooth/effort



        # will be set in reset/housekeeping
        self._target_uid = None
        self._last_action_for_smooth = np.zeros(4, dtype=float)

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)

    # --- Build base obs space from parent ---
        base_obs_space = self._observationSpace()
        base_low = np.asarray(base_obs_space.low, dtype=np.float32).reshape(-1)
        base_high = np.asarray(base_obs_space.high, dtype=np.float32).reshape(-1)

        # --- Extend with [bearing, yaw_err, dist] ---
        aug_low = np.concatenate([base_low, np.array([-np.pi, -np.pi, 0.0], dtype=np.float32)])
        aug_high = np.concatenate([base_high, np.array([np.pi, np.pi, 10.0], dtype=np.float32)])

        self.observation_space = spaces.Box(
            low=aug_low,
            high=aug_high,
            shape=(aug_low.shape[0],),
            dtype=np.float32
        )
    # --------------------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------------------
    def _sim_time(self) -> float:
        return self.step_counter * self.PYB_TIMESTEP

    # def _target_traj(self, t: float) -> np.ndarray:
    #     """Parametric target trajectory (circle in XY; optional slow Z wave)."""
    #     if self.TARGET_STATIC:
    #         return self.TARGET_CENTER.copy()
    #     cx, cy, cz = self.TARGET_CENTER
    #     x = cx + self.TARGET_RADIUS * np.cos(self.TARGET_OMEGA * t)
    #     y = cy + self.TARGET_RADIUS * np.sin(self.TARGET_OMEGA * t)
    #     if self.TARGET_HEIGHT_WAVE:
    #         z = cz + 0.2 * np.sin(self.TARGET_OMEGA * 0.5 * t)
    #     else:
    #         z = cz
    #     return np.array([x, y, z], dtype=float)

    def _target_traj(self, t: float) -> np.ndarray:
        # Always static
        return self.TARGET_CENTER.copy()

    def _ensure_target_body(self):
        """Spawn/refresh a visible target sphere (if SHOW_TARGET)."""
        if self._target_uid is None and self.SHOW_TARGET:
            # bigger, easier to see
            vis = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.25,  # was tiny; now ~15 cm
                rgbaColor=[1, 0, 0, 1],
                physicsClientId=self.CLIENT
            )
            col = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=0.3,
                physicsClientId=self.CLIENT
            )
            pos = self._target_traj(0.0)
            self._target_uid = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos.tolist(),
                physicsClientId=self.CLIENT
            )

    def _quantize_action(self, action,
                         binned_channels=(0,1),
                         bins_per_channel=(20,20),
                         ranges=((-0.6,0.6), (-0.6,0.6))):
        """Snap selected channels to nearest bin center; safe for (4,) or (1,4)."""
        a = np.asarray(action, dtype=np.float32)
        orig_nd = a.ndim
        a2 = np.atleast_2d(a).copy()  # (N, act_dim)

        for ch, nbin, (mn, mx) in zip(binned_channels, bins_per_channel, ranges):
            centers = np.linspace(mn, mx, nbin, dtype=np.float32)
            vals = np.clip(a2[:, ch], mn, mx)
            idx = np.abs(vals[:, None] - centers[None, :]).argmin(axis=1)
            a2[:, ch] = centers[idx]
        return a2[0] if orig_nd == 1 else a2

    def _clip_action(self, action):
        # Force to float32 and 2-D [n, 4]
        a = np.asarray(action, dtype=np.float32).reshape(-1, 4)

        # Column-wise clip: [vx, vy, vz, yaw_rate]
        a[:, 0] = np.clip(a[:, 0], -0.6, 0.6)  # vx
        a[:, 1] = np.clip(a[:, 1], -0.6, 0.6)  # vy
        a[:, 2] = np.clip(a[:, 2], -0.4, 0.4)  # vz
        a[:, 3] = np.clip(a[:, 3], -0.6, 0.6)  # yaw_rate

        # Ensure row count matches NUM_DRONES
        if a.shape[0] != self.NUM_DRONES:
            if a.shape[0] == 1 and self.NUM_DRONES > 1:
                a = np.repeat(a, self.NUM_DRONES, axis=0)
            else:
                a = a[:self.NUM_DRONES]

        return a  # ALWAYS 2-D: (NUM_DRONES, 4)

    def _as_1d(self, action: np.ndarray) -> np.ndarray:
        """Return (act_dim,) no matter if input is (4,) or (1,4) or batched."""
        a = np.asarray(action, dtype=np.float32)
        if a.ndim == 2:
            a = a[0]
        return a

    def _smooth_and_limit(self, a_des_1d: np.ndarray, dt: float) -> np.ndarray:
        """First-order low-pass + per-step slew limit, both in action units."""
        # low-pass
        alpha = dt / (self._act_tau + dt)
        u = self._act_prev + alpha * (a_des_1d - self._act_prev)
        # slew cap
        max_delta = self._act_rate * dt
        u = np.clip(u, self._act_prev - max_delta, self._act_prev + max_delta)
        # commit
        self._act_prev = u
        return u

    # def _update_target_pose(self):
    #     """Move target each step."""
    #     self._ensure_target_body()
    #     pos = self._target_traj(self._sim_time())
    #     p.resetBasePositionAndOrientation(self._target_uid, pos, [0, 0, 0, 1], physicsClientId=self.CLIENT)
    # def _update_target_pose(self):
    #     """Move target each step and keep it obvious in the GUI."""
    #     self._ensure_target_body()
    #     pos = self._target_traj(self._sim_time())
    #
    #     if self._target_uid is not None:
    #         p.resetBasePositionAndOrientation(
    #             self._target_uid, pos, [0, 0, 0, 1],
    #             physicsClientId=self.CLIENT
    #         )
    #
    #     # draw a tall red marker (reused each step)
    #     if not hasattr(self, "_tgt_marker_id"):
    #         self._tgt_marker_id = None
    #         print( self._tgt_marker_id)
    #     self._tgt_marker_id = p.addUserDebugLine(
    #         pos, (pos + np.array([0, 0, 0.3])),
    #         lineColorRGB=[1, 0, 0], lineWidth=8,
    #         lifeTime=0,
    #         replaceItemUniqueId=(self._tgt_marker_id or -1),
    #         physicsClientId=self.CLIENT
    #     )
    #
    #     # auto-aim the GUI camera at the target (optional)
    #     if  self.CAMERA_MODE == "follow" and self.GUI:
    #         p.resetDebugVisualizerCamera(
    #             cameraDistance=2.2, cameraYaw=45, cameraPitch=-30,
    #             cameraTargetPosition=pos.tolist(),
    #             physicsClientId=self.CLIENT
    #         )
    #     elif self.CAMERA_MODE == "drone" and self.GUI:
    #         drone_pos = self._getDroneStateVector(0)[0:3]
    #         p.resetDebugVisualizerCamera(2.2, 45, -30, drone_pos.tolist(), physicsClientId=self.CLIENT)

    def _update_target_pose(self):
        """Ensure a visible, static target marker once."""
        self._ensure_target_body()
        if self._target_uid is not None:
            pos = self.TARGET_CENTER.copy()
            p.resetBasePositionAndOrientation(
                self._target_uid, pos, [0, 0, 0, 1], physicsClientId=self.CLIENT
            )
        # Optional: a short marker line so it's obvious in GUI
        if not hasattr(self, "_tgt_marker_id"):
            self._tgt_marker_id = None
        self._tgt_marker_id = p.addUserDebugLine(
            self.TARGET_CENTER, (self.TARGET_CENTER + np.array([0, 0, 0.3])),
            lineColorRGB=[1, 0, 0], lineWidth=8, lifeTime=0,
            replaceItemUniqueId=(self._tgt_marker_id or -1),
            physicsClientId=self.CLIENT
        )
        #
    # --------------------------------------------------------------------------
    # RL API
    # --------------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        # book-keeping from the episode that just ended
        if self._episodes_done > 0:
            self._success_window.append(1.0 if self._last_success else 0.0)

        obs, info = super().reset(seed=seed, options=options)

        # compute moving success rate
        succ_rate = (sum(self._success_window) / len(self._success_window)) if len(self._success_window) > 0 else 0.0

        # curriculum: if we are consistently succeeding, expand jitter; else keep it modest
        # map success rate -> scale in [0.0, 1.0]
        # 0% success -> scale ~0.0 ; 70%+ success -> scale ~1.0
        scale = np.clip((succ_rate - 0.2) / (0.7 - 0.2), 0.0, 1.0)

        max_xy = 0.5 + scale * (self._JITTER_MAX_XY - 0.5)  # grows from 0.5 → 2.5 m
        max_z = 0.2 + scale * (self._JITTER_MAX_Z - 0.2)  # grows from 0.2 → 1.0 m

        base = np.array([1.5, 1.0, 1.0])  # your nominal center; adjust if you like
        jitter = np.array([
            np.random.uniform(-max_xy, max_xy),
            np.random.uniform(-max_xy, max_xy),
            np.random.uniform(-max_z, max_z)
        ])
        self.TARGET_CENTER = base + jitter

        self._inside_counter = 0
        self._last_d = None
        self._takeoff_given = False
        self._crash_penalty = 0.0

        # (Re)spawn/refresh the visual marker if you show it
        self._ensure_target_body()
        if hasattr(self, "_update_target_pose"):
            self._update_target_pose()

        # reset per-episode flags
        self._last_success = False
        self._episodes_done += 1
        self._act_prev[:] = 0.0  # re-arm smoother each episode
        self._last_action_for_rw = None  # for your reward’s smoothness term

        return obs, info


    # def reset(self, seed: int | None = None, options: dict | None = None):
    #     obs, info = super().reset(seed=seed, options=options)
    #     # Drop the default last action smoother
    #     self._last_action_for_smooth = np.zeros_like(self._last_action_for_smooth)
    #
    #     # (Re)spawn & place target
    #     self._ensure_target_body()
    #     self._update_target_pose()
    #     return obs, info

    # def step(self, action):
    #     # cache action if you use smooth/effort terms (optional)
    #     try:
    #         a_np = np.array(action).reshape(-1)
    #         if a_np.size == 4:
    #             self._prev_action = a_np.copy()
    #     except Exception:
    #         pass
    #
    #     # target is static; just ensure it exists
    #     self._update_target_pose()
    #
    #     obs, reward, terminated, truncated, info = super().step(action)
    #     if terminated or truncated:
    #         self._episodes_done += 1
    #
    #     if terminated:
    #         pos = self._getDroneStateVector(0)[0:3]
    #         self._last_success = (np.linalg.norm(self.TARGET_CENTER - pos) < self.SUCCESS_R)
    #     return obs, reward, terminated, truncated, info

    def step(self, action):
        # 1. Enforce shape & clip
        a = self._clip_action(action)  # (NUM_DRONES, 4)

        # 2. Quantize selected channels (vx, vy)
        if getattr(self, "_enable_binning", False):
            a = self._quantize_action(a, **self._bin_cfg)

        # 2b. Optional smoother (reference low-pass)
        #     This prevents violent RPM flips between bins
        if not hasattr(self, "_last_vel_ref"):
            self._last_vel_ref = np.zeros(4, dtype=np.float32)
        ref_alpha = 0.35  # 0.2–0.4 is typical; lower = smoother
        a = self._last_vel_ref + ref_alpha * (a - self._last_vel_ref)
        self._last_vel_ref = a.copy()

        # 3. Keep last action for smoothness term
        try:
            self._prev_action = a[0].copy()  # single-drone case
        except Exception:
            self._prev_action = None

        # 4. Target marker maintenance
        self._ensure_target_body()

        # 5. Step physics
        obs, reward, terminated, truncated, info = super().step(a)

        # 6. Success bookkeeping (unchanged)
        s = self._getDroneStateVector(0)
        pos = s[0:3]
        vel = s[10:13]
        d = float(np.linalg.norm(self.TARGET_CENTER - pos))
        speed = float(np.linalg.norm(vel))
        if (d < self.SUCCESS_R) and (speed < self.NEAR_SPEED):
            self._inside_counter += 1
        else:
            self._inside_counter = 0
        if self._inside_counter >= self.HOLD_STEPS:
            self._last_success = True
            terminated = True
        if truncated:
            self._last_success = False

        # For debugging (see discrete vx/vy early on)
        if self.step_counter < 5:
            print("quantized+smoothed action:", a)

        return obs, reward, terminated, truncated, info

    # def step(self, action):
    #     # advance target before physics
    #     self._ensure_target_body()
    #
    #     obs, reward, terminated, truncated, info = super().step(action)
    #     return obs, reward, terminated, truncated, info

    # def step(self, action):
    #     # keep a copy for smoothness penalty
    #     if action is not None:
    #         if isinstance(action, (list, tuple, np.ndarray)):
    #             a_np = np.array(action).reshape(-1)
    #             # store per-motor when using RPM-like actions; otherwise store flat
    #             if a_np.size == 4:
    #                 self._prev_action = a_np.copy()
    #             else:
    #                 self._prev_action = a_np.copy()
    #         else:
    #             self._prev_action = None
    #     else:
    #         self._prev_action = None
    #
    #     # advance target before physics
    #     self._update_target_pose()
    #
    #     print("action is:", action)
    #
    #     obs, reward, terminated, truncated, info = super().step(action)
    #
    #     # store smoothed action reference if possible
    #     if self._prev_action is not None:
    #         # If env used PID/VEL internally, reward smoothing still uses provided action vector
    #         if self._last_action_for_smooth.shape == self._prev_action.shape:
    #             self._last_action_for_smooth = self._prev_action
    #     return obs, reward, terminated, truncated, info

    # --------------------------------------------------------------------------
    # Rewards / Termination
    # --------------------------------------------------------------------------
    # def _computeReward(self):
    #     """
    #     Minimal shaping:
    #       - Alive bonus every step
    #       - Linear distance shaping capped at D_MAX
    #       - Big success bonus inside SUCCESS_R
    #       - Optional small smoothness penalty
    #     """
    #     ALIVE = 0.02
    #     D_MAX = 10.0
    #     W_DIST = 1.0
    #
    #     s = self._getDroneStateVector(0)
    #     pos = s[0:3]
    #     # tgt = self._target_traj(self._sim_time())
    #     # dist = float(np.linalg.norm(tgt - pos))
    #
    #     dist = max(0, 2 - np.linalg.norm(self.TARGET_CENTER - pos) ** 4)
    #     reward = dist
    #
    #     # Sparse success bonus
    #     if dist < self.SUCCESS_R:
    #         reward += 1.0
    #
    #     if getattr(self, "_crash_penalty", 0):
    #         reward += self._crash_penalty
    #         self._crash_penalty = 0.0
    #
    #     # print("tgt is:", tgt)
    #     # print("pos is:", pos)
    #     # print("reward is:", reward)
    #     # print("r distance is:", r_dist)
    #     # print("distance is:", dist)
    #
    #
    #     return float(reward)

    def _computeReward(self):
        ALIVE = 0.01
        TERM_BONUS = 1.0
        TAKEOFF_RW = 0.2
        PROG_CLAMP = 0.05

        s = self._getDroneStateVector(0)
        pos = s[0:3]
        vel = s[10:13]
        w = s[13:16]  # body rates (check indices)
        yaw = s[9]
        vec = self.TARGET_CENTER - pos
        bearing = np.arctan2(vec[1], vec[0])
        yaw_err = np.arctan2(np.sin(bearing - yaw), np.cos(bearing - yaw))
        r_yaw_err = np.exp(-self.YAW_ERR_BETA * np.abs(yaw_err))  # (0,1]

        vec = self.TARGET_CENTER - pos
        d = float(np.linalg.norm(vec))
        r_hat = vec / (d + 1e-6)

        # distance shaping
        r_dist = np.exp(-d / 2.0)

        # velocity & components
        v = vel
        speed = float(np.linalg.norm(v)) + 1e-9
        v_r = float(np.dot(v, r_hat))
        v_t = float(np.linalg.norm(v - v_r * r_hat))

        # alignment & approach
        cos_th = float(np.dot(v, r_hat) / speed)
        r_dir = 0.2 * (0.5 * (cos_th + 1.0))
        r_approach = 0.03 * np.clip(v_r, 0.0, 0.8)

        # anti-orbit (distance-aware)
        k_far, k_near = 0.15, 0.03
        alpha = np.clip(d / 2.5, 0.0, 1.0)
        k_t = k_near + (k_far - k_near) * alpha
        r_anti_orbit = -k_t * v_t

        # progress
        if self._last_d is None:
            r_prog = 0.0
        else:
            r_prog = np.clip(self._last_d - d, -PROG_CLAMP, PROG_CLAMP)
        self._last_d = d

        # stop incentive
        speed_mag = float(np.linalg.norm(v))
        r_stop = 0.2 if (d < self.SUCCESS_R and speed_mag < self.NEAR_SPEED) else 0.0

        # takeoff one-off
        r_takeoff = 0.0
        if (not self._takeoff_given) and (pos[2] > self.TAKEOFF_ALT):
            r_takeoff = TAKEOFF_RW
            self._takeoff_given = True

        # angular-rate damping
        r_rates = -0.002 * float(np.dot(w, w))

        # small action smoothness
        r_smooth = 0.0
        if getattr(self, "_last_action_for_rw", None) is not None and self._prev_action is not None:
            da = self._prev_action - self._last_action_for_rw
            r_smooth = -0.001 * float(np.dot(da, da))
        self._last_action_for_rw = None if self._prev_action is None else self._prev_action.copy()

        reward = (ALIVE + r_dist + r_dir + r_approach + r_anti_orbit +
                  self.W_YAW_ERR * r_yaw_err +  # new term
                  r_prog + r_stop + r_takeoff + r_rates + r_smooth)

        # crash penalty passthrough
        if getattr(self, "_crash_penalty", 0):
            reward += self._crash_penalty
            self._crash_penalty = 0.0

        if d < self.SUCCESS_R:
            reward += TERM_BONUS

        return float(reward)

    # def _computeReward(self):
    #     """
    #     Bounded shaping + progress + stop incentive.
    #     Keeps per-step reward ~[0, ~1.6] to stay PPO-friendly.
    #     """
    #     ALIVE = 0.01
    #     DIST_SCALE = 2.0  # meters for exp shaping
    #     PROG_CLAMP = 0.05  # clamp progress reward per step
    #     NEAR_BONUS = 0.2  # small bonus when close AND slow
    #     TERM_BONUS = 1.0  # one-off when inside SUCCESS_R
    #     TAKEOFF_RW = 0.2  # one-off when first above TAKEOFF_ALT
    #
    #     s = self._getDroneStateVector(0)
    #     pos = s[0:3]
    #     vel = s[10:13]
    #
    #     d = float(np.linalg.norm(self.TARGET_CENTER - pos))
    #     r_dist = np.exp(-d / DIST_SCALE)  # (0,1]
    #
    #     # Progress (distance decrease): positive if moving closer
    #     if self._last_d is None:
    #         r_prog = 0.0
    #     else:
    #         r_prog = np.clip(self._last_d - d, -PROG_CLAMP, PROG_CLAMP)
    #     self._last_d = d
    #
    #     # Stop incentive only when close and slow
    #     speed = float(np.linalg.norm(vel))
    #     r_stop = NEAR_BONUS if (d < self.SUCCESS_R and speed < self.NEAR_SPEED) else 0.0
    #
    #     # One-off takeoff reward to prevent lying flat
    #     r_takeoff = 0.0
    #     if (not self._takeoff_given) and (pos[2] > self.TAKEOFF_ALT):
    #         r_takeoff = TAKEOFF_RW
    #         self._takeoff_given = True
    #
    #     reward = ALIVE + r_dist + r_prog + r_stop + r_takeoff
    #
    #     # One-off truncation/crash penalty (if set in _computeTruncated)
    #     if getattr(self, "_crash_penalty", 0):
    #         reward += self._crash_penalty
    #         self._crash_penalty = 0.0
    #
    #     # Terminal (success) bump when inside radius (bounded)
    #     if d < self.SUCCESS_R:
    #         reward += TERM_BONUS
    #
    #     return float(reward)

    def _computeObs(self):
        """
        Extend BaseRLAviary observation with bearing, yaw_err, and distance.
        """
        obs = np.asarray(super()._computeObs(), dtype=np.float32).reshape(-1)
        s = self._getDroneStateVector(0)
        pos, yaw = s[0:3], s[9]

        # Bearing and yaw error
        vec = self.TARGET_CENTER - pos
        bearing = np.arctan2(vec[1], vec[0])
        yaw_err = np.arctan2(np.sin(bearing - yaw), np.cos(bearing - yaw))
        dist = float(np.linalg.norm(vec))

        extra = np.array([bearing, yaw_err, dist], dtype=np.float32)
        return np.concatenate([obs, extra], axis=0)

    def _computeTerminated(self):
        """
        Success if within SUCCESS_R of the target.
        """
        # return False
        s = self._getDroneStateVector(0)
        pos = s[0:3]
        return bool(np.linalg.norm(self.TARGET_CENTER - pos) < self.SUCCESS_R)

    # def _computeTruncated(self):
    #     """
    #     Truncate on flight envelope violations or time limit.
    #     """
    #     s = self._getDroneStateVector(0)
    #     x, y, z = s[0], s[1], s[2]
    #     roll, pitch = s[7], s[8]
    #
    #     # widen workspace so the drone can travel to a distant target
    #     too_far = (abs(x) > 5.0) or (abs(y) > 5.0) or (z < 0.0) or (z > 5.0)
    #     # keep the tilt guardrail, but 0.9 rad (~52°) is a bit more forgiving
    #     too_tilted = (abs(roll) > 0.4) or (abs(pitch) > 0.4)
    #
    #     # timeout = False
    #     timeout = (self.step_counter / self.PYB_FREQ) > self.EPISODE_LEN_SEC
    #     self._crash_penalty = -1 if too_far or too_tilted or timeout else None
    #     # print("too far", too_far)
    #     # print("too too_tilted", too_tilted)
    #     # print("timeout", timeout)
    #     return bool(too_far or too_tilted or timeout)

    def _computeTruncated(self):
        s = self._getDroneStateVector(0)
        x, y, z = float(s[0]), float(s[1]), float(s[2])
        roll, pitch = float(s[7]), float(s[8])

        t = self.step_counter / self.PYB_FREQ
        grace = (t < 2.0)

        # Give room to maneuver
        too_far = (abs(x) > 8.0) or (abs(y) > 8.0) or (z < 0.0) or (z > 6.0)
        # Be a bit more tolerant on tilt early in training
        too_tilted = ((abs(roll) > 0.6) or (abs(pitch) > 0.6)) and (not grace)

        timeout = t > self.EPISODE_LEN_SEC

        # Small one-time penalty on crash/truncate; don't spam negatives
        self._crash_penalty = -0.2 if (too_far or too_tilted) else 0.0
        return bool(too_far or too_tilted or timeout)

    def _computeInfo(self):
        # Helpful diagnostics for logging
        s = self._getDroneStateVector(0)
        pos = s[0:3]
        yaw = s[9]
        tgt = self._target_traj(self._sim_time())
        vec = tgt - pos
        bearing = np.arctan2(vec[1], vec[0])
        yaw_err = np.arctan2(np.sin(bearing - yaw), np.cos(bearing - yaw))
        d = float(np.linalg.norm(self.TARGET_CENTER - pos))
        return {
            "target_x": float(tgt[0]),
            "target_y": float(tgt[1]),
            "target_z": float(tgt[2]),
            "yaw_err": float(yaw_err),
            "dist_to_target": d,
            "success": d < self.SUCCESS_R
        }
