# play_target_vec.py
import os
import time
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from gym_pybullet_drones.envs.TargetAviary import TargetAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.CustomActionWrapper import CustomActionWrapper


# -------- Defaults (match training) --------
DEFAULT_MODEL_PATH = "results/best_model.zip"
DEFAULT_STATS_PATH = "results/vecnormalize.pkl"   # path saved by train_env.save(...)
DEFAULT_GUI = True
DEFAULT_OBS = ObservationType("kin")
DEFAULT_ACT = ActionType("vel")

# TargetAviary params (adjust as needed, but keep consistent with training)
DEFAULT_EPISODE_LEN_SEC = 30
DEFAULT_TARGET_STATIC = True
DEFAULT_TARGET_CENTER = (-1.5, -4.0, -3.0)
DEFAULT_TARGET_RADIUS = 0.4
DEFAULT_TARGET_OMEGA_HZ = 0.0
DEFAULT_TARGET_HEIGHT_WAVE = False
DEFAULT_SHOW_TARGET = True
DEFAULT_CAMERA_MODE = "follow"

# Wrapper settings (must match training)
BINNED_CHANNELS = [1, 3]   # roll, yaw
BINS_PER_CHANNEL = 5       # 5 bins each


def make_wrapped_env_gui(gui, obs, act,
                         episode_len_sec,
                         target_static, target_center, target_radius,
                         target_omega_hz, target_height_wave,
                         show_target, camera_mode):
    """Factory for a single CustomActionWrapper-wrapped TargetAviary with GUI flag."""
    def _init(**_):
        base = TargetAviary(
            gui=gui,
            record=False,
            obs=obs,
            act=act,
            episode_len_sec=episode_len_sec,
            target_static=target_static,
            target_center=target_center,
            target_radius=target_radius,
            target_omega_hz=target_omega_hz,
            target_height_wave=target_height_wave,
            show_target=show_target,
            camera_mode=camera_mode,
        )
        return CustomActionWrapper(base,
                                   binned_channels=BINNED_CHANNELS,
                                   bins_per_channel=BINS_PER_CHANNEL)
    return _init


def play_vec(
    model_path: str,
    stats_path: str,
    gui: bool,
    obs: ObservationType,
    act: ActionType,
    episode_len_sec: int,
    target_static: bool,
    target_center: tuple,
    target_radius: float,
    target_omega_hz: float,
    target_height_wave: bool,
    show_target: bool,
    camera_mode: str,
):
    # ---- Load model ----
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        return
    model = PPO.load(model_path, device="cpu")
    print(f"[INFO] Loaded model from {model_path}")

    # ---- Build VecEnv (Wrapper + VecNormalize) ----
    vec = make_vec_env(
        make_wrapped_env_gui(gui, obs, act,
                             episode_len_sec,
                             target_static, target_center, target_radius,
                             target_omega_hz, target_height_wave,
                             show_target, camera_mode),
        n_envs=1, seed=42
    )

    if not os.path.isfile(stats_path):
        print(f"[WARN] VecNormalize stats not found at: {stats_path}. Running unnormalized.")
        eval_env = vec
    else:
        eval_env = VecNormalize.load(stats_path, vec)
        eval_env.training = False
        eval_env.norm_reward = False
        print(f"[INFO] Loaded VecNormalize stats from {stats_path}")

    # Unwrap chain to reach the base TargetAviary for logging/render/meta
    # DummyVecEnv -> VecNormalize -> CustomActionWrapper -> TargetAviary
    under = eval_env.venv.envs[0]      # CustomActionWrapper
    try:
        base_env = under.env           # TargetAviary
    except AttributeError:
        base_env = under               # if wrapper forwards attributes

    # ---- Logger (only meaningful for KIN obs) ----
    logger = Logger(
        logging_freq_hz=int(base_env.CTRL_FREQ),
        num_drones=1,
        output_folder="logs_playback_target_vec/",
        colab=False,
    )

    # ---- Rollout ----
    obs_ = eval_env.reset()  # batched observation
    start = time.time()
    horizon = int((base_env.EPISODE_LEN_SEC + 2) * base_env.CTRL_FREQ)

    # Track distances if available
    dist_list = []

    for i in range(horizon):
        action, _ = model.predict(obs_, deterministic=True)  # already batched
        obs_, rewards, dones, infos = eval_env.step(action)

        # Compute/collect distance to target safely
        try:
            # Prefer env info if your env populates it
            if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
                if "dist_to_target" in infos[0]:
                    dist_list.append(float(infos[0]["dist_to_target"]))
                else:
                    # Fallback: compute directly from base_env state
                    s = base_env._getDroneStateVector(0)
                    pos = s[0:3]
                    d = float(np.linalg.norm(np.array(base_env.TARGET_CENTER) - pos))
                    dist_list.append(d)
        except Exception:
            pass

        # Optional logging (obs_ here is normalized & batched)
        if obs == ObservationType.KIN:
            try:
                o = np.array(obs_).squeeze()
                a = np.array(action).squeeze()
                logger.log(
                    drone=0,
                    timestamp=i / base_env.CTRL_FREQ,
                    state=np.hstack([o[0:3], np.zeros(4), o[3:15], a if a.ndim == 0 else a]),
                    control=np.zeros(12),
                )
            except Exception:
                pass

        if gui:
            try:
                base_env.render()
            except Exception:
                pass

        sync(i, start, base_env.CTRL_TIMESTEP)

        if dones[0]:
            break

    # ---- Report & close ----
    if dist_list:
        print(f"[INFO] Mean distance to target: {np.mean(dist_list):.3f} m "
              f"(min {np.min(dist_list):.3f}, max {np.max(dist_list):.3f})")

    try:
        base_env.close()
    except Exception:
        pass

    if obs == ObservationType.KIN:
        logger.plot()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Play a saved PPO policy with wrapper + VecNormalize.")
    p.add_argument("--model_path",  type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--stats_path",  type=str, default=DEFAULT_STATS_PATH)
    p.add_argument("--gui",         type=lambda s: s.lower() == "true", default=DEFAULT_GUI)

    # Obs/Act (must match training)
    p.add_argument("--obs",         type=str, default="kin", help="kin|rgb")
    p.add_argument("--act",         type=str, default="rpm",
                   help="rpm|pid|vel|one_d_rpm|one_d_pid")

    # TargetAviary params
    p.add_argument("--episode_len_sec", type=int, default=DEFAULT_EPISODE_LEN_SEC)
    p.add_argument("--target_static",   type=lambda s: s.lower() == "true", default=DEFAULT_TARGET_STATIC)
    p.add_argument("--target_center",   type=str, default="-3.0,1.0,1.0",
                   help="comma-separated triple, e.g. 1.5,1.0,1.0")
    p.add_argument("--target_radius",   type=float, default=DEFAULT_TARGET_RADIUS)
    p.add_argument("--target_omega_hz", type=float, default=DEFAULT_TARGET_OMEGA_HZ)
    p.add_argument("--target_height_wave", type=lambda s: s.lower() == "true", default=DEFAULT_TARGET_HEIGHT_WAVE)
    p.add_argument("--show_target",     type=lambda s: s.lower() == "true", default=DEFAULT_SHOW_TARGET)
    p.add_argument("--camera_mode",     type=str, default=DEFAULT_CAMERA_MODE, help="follow|drone|static")

    args = p.parse_args()

    obs_enum = ObservationType(args.obs)
    act_enum = ActionType(args.act)
    tc = tuple(float(x.strip()) for x in args.target_center.split(","))

    play_vec(
        model_path=args.model_path,
        stats_path=args.stats_path,
        gui=args.gui,
        obs=obs_enum,
        act=act_enum,
        episode_len_sec=args.episode_len_sec,
        target_static=args.target_static,
        target_center=tc,
        target_radius=args.target_radius,
        target_omega_hz=args.target_omega_hz,
        target_height_wave=args.target_height_wave,
        show_target=args.show_target,
        camera_mode=args.camera_mode,
    )
