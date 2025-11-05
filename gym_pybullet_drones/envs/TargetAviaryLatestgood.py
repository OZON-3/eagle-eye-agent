import numpy as np
import pybullet as p

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
                 act: ActionType = ActionType.RPM,
                 # Target/episode params
                 episode_len_sec: int = 60,
                 target_static: bool = True,
                 # replace these three defaults in __init__ signature
                 target_center=(1.5, 1.0, 1.0),  # ~1.5 m in front of the origin
                 target_radius: float = 0.4,  # small circle
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
        self.TARGET_OMEGA = 2 * np.pi * float(target_omega_hz)  # rad/s
        self.TARGET_HEIGHT_WAVE = bool(target_height_wave)
        self.YAW_ALIGN_W = float(yaw_align_weight)
        self.SMOOTH_W = float(smooth_action_weight)
        self.SUCCESS_R = float(success_radius)
        self.SHOW_TARGET = bool(show_target)
        self.CAMERA_MODE = str(camera_mode).lower()

        # --- Reward shaping (bounded & small) ---
        self.ALIVE_BONUS = 0.01
        self.DIST_SCALE = 2.0  # meters, for exp-shaped distance
        self.W_DIST = 1.0
        self.W_YAW = 0.10  # set 0.0 if you don't want yaw reward
        self.SMOOTH_W = 0.005
        self.EFFORT_W = 0.002
        self.SUCCESS_HOLD_STEPS = 10  # must stay inside radius for K steps

        # --- Episode/curriculum bookkeeping ---
        self._inside_counter = 0
        self._episodes_done = 0
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

    # --------------------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------------------
    def _sim_time(self) -> float:
        return self.step_counter * self.PYB_TIMESTEP

    def _target_traj(self, t: float) -> np.ndarray:
        """Parametric target trajectory (circle in XY; optional slow Z wave)."""
        if self.TARGET_STATIC:
            return self.TARGET_CENTER.copy()
        cx, cy, cz = self.TARGET_CENTER
        x = cx + self.TARGET_RADIUS * np.cos(self.TARGET_OMEGA * t)
        y = cy + self.TARGET_RADIUS * np.sin(self.TARGET_OMEGA * t)
        if self.TARGET_HEIGHT_WAVE:
            z = cz + 0.2 * np.sin(self.TARGET_OMEGA * 0.5 * t)
        else:
            z = cz
        return np.array([x, y, z], dtype=float)

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
    # def _update_target_pose(self):
    #     """Move target each step."""
    #     self._ensure_target_body()
    #     pos = self._target_traj(self._sim_time())
    #     p.resetBasePositionAndOrientation(self._target_uid, pos, [0, 0, 0, 1], physicsClientId=self.CLIENT)
    def _update_target_pose(self):
        """Move target each step and keep it obvious in the GUI."""
        self._ensure_target_body()
        pos = self._target_traj(self._sim_time())

        if self._target_uid is not None:
            p.resetBasePositionAndOrientation(
                self._target_uid, pos, [0, 0, 0, 1],
                physicsClientId=self.CLIENT
            )

        # draw a tall red marker (reused each step)
        if not hasattr(self, "_tgt_marker_id"):
            self._tgt_marker_id = None
            print( self._tgt_marker_id)
        self._tgt_marker_id = p.addUserDebugLine(
            pos, (pos + np.array([0, 0, 0.3])),
            lineColorRGB=[1, 0, 0], lineWidth=8,
            lifeTime=0,
            replaceItemUniqueId=(self._tgt_marker_id or -1),
            physicsClientId=self.CLIENT
        )

        # auto-aim the GUI camera at the target (optional)
        if  self.CAMERA_MODE == "follow" and self.GUI:
            p.resetDebugVisualizerCamera(
                cameraDistance=2.2, cameraYaw=45, cameraPitch=-30,
                cameraTargetPosition=pos.tolist(),
                physicsClientId=self.CLIENT
            )
        elif self.CAMERA_MODE == "drone" and self.GUI:
            drone_pos = self._getDroneStateVector(0)[0:3]
            p.resetDebugVisualizerCamera(2.2, 45, -30, drone_pos.tolist(), physicsClientId=self.CLIENT)
        #
    # --------------------------------------------------------------------------
    # RL API
    # --------------------------------------------------------------------------
    # def reset(self, seed: int | None = None, options: dict | None = None):
    #     obs, info = super().reset(seed=seed, options=options)
    #     # Drop the default last action smoother
    #     self._last_action_for_smooth = np.zeros_like(self._last_action_for_smooth)
    #
    #     # (Re)spawn & place target
    #     self._ensure_target_body()
    #     self._update_target_pose()
    #     return obs, info

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)

        # Episode/curriculum bookkeeping
        self._inside_counter = 0
        self._last_dists.clear()
        self._prev_action[:] = 0.0
        self._last_action_for_smooth = np.zeros_like(self._prev_action)

        # (Re)spawn & place target
        self._ensure_target_body()
        self._update_target_pose()

        # --------- Simple curriculum by episode count ---------
        ep = self._episodes_done
        if ep < 300:
            self.TARGET_STATIC = True
            self.SUCCESS_R = max(self.SUCCESS_R, 0.60)
            self.TARGET_OMEGA = 0.0
        elif ep < 900:
            self.TARGET_STATIC = True
            self.SUCCESS_R = max(0.40, min(self.SUCCESS_R, 0.60))
            self.TARGET_OMEGA = 0.0
        elif ep < 2000:
            self.TARGET_STATIC = False
            self.TARGET_OMEGA = 2 * np.pi * 0.05
            self.SUCCESS_R = max(0.35, min(self.SUCCESS_R, 0.45))
        elif ep < 3500:
            self.TARGET_STATIC = False
            self.TARGET_OMEGA = 2 * np.pi * 0.10
            self.SUCCESS_R = max(0.30, min(self.SUCCESS_R, 0.40))
        else:
            self.TARGET_STATIC = False
            self.TARGET_OMEGA = 2 * np.pi * np.random.uniform(0.05, 0.12)
            self.SUCCESS_R = np.random.uniform(0.25, 0.32)
            # mild randomization of target center
            jitter = np.random.uniform(-0.5, 0.5, size=3)
            jitter[2] *= 0.3
            self.TARGET_CENTER = np.array(self.TARGET_CENTER) + jitter
        # -------------------------------------------------------

        return obs, info

    def step(self, action):
        # advance target before physics
        self._ensure_target_body()

        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info

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
    def _computeReward(self):
        """
        Minimal shaping:
          - Alive bonus every step
          - Linear distance shaping capped at D_MAX
          - Big success bonus inside SUCCESS_R
          - Optional small smoothness penalty
        """
        ALIVE = 0.02
        D_MAX = 10.0
        W_DIST = 1.0

        s = self._getDroneStateVector(0)
        pos = s[0:3]
        # tgt = self._target_traj(self._sim_time())
        # dist = float(np.linalg.norm(tgt - pos))

        dist = max(0, 2 - np.linalg.norm(self.TARGET_CENTER - pos) ** 4)
        reward = dist

        # Sparse success bonus
        if dist < self.SUCCESS_R:
            reward += 1.0

        if getattr(self, "_crash_penalty", 0):
            reward += self._crash_penalty
            self._crash_penalty = 0.0

        # print("tgt is:", tgt)
        # print("pos is:", pos)
        # print("reward is:", reward)
        # print("r distance is:", r_dist)
        # print("distance is:", dist)


        return float(reward)

    def _computeTerminated(self):
        """
        Success if within SUCCESS_R of the target.
        """
        # return False
        s = self._getDroneStateVector(0)
        pos = s[0:3]
        return bool(np.linalg.norm(self.TARGET_CENTER - pos) < self.SUCCESS_R)

    def _computeTruncated(self):
        """
        Truncate on flight envelope violations or time limit.
        """
        s = self._getDroneStateVector(0)
        x, y, z = s[0], s[1], s[2]
        roll, pitch = s[7], s[8]

        # widen workspace so the drone can travel to a distant target
        too_far = (abs(x) > 5.0) or (abs(y) > 5.0) or (z < 0.0) or (z > 5.0)
        # keep the tilt guardrail, but 0.9 rad (~52°) is a bit more forgiving
        too_tilted = (abs(roll) > 0.4) or (abs(pitch) > 0.4)

        # timeout = False
        timeout = (self.step_counter / self.PYB_FREQ) > self.EPISODE_LEN_SEC
        self._crash_penalty = -1 if too_far or too_tilted or timeout else None
        # print("too far", too_far)
        # print("too too_tilted", too_tilted)
        # print("timeout", timeout)
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
        return {
            "target_x": float(tgt[0]),
            "target_y": float(tgt[1]),
            "target_z": float(tgt[2]),
            "dist_to_target": float(np.linalg.norm(vec)),
            "yaw_err": float(yaw_err)
        }
