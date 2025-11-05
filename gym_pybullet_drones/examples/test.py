# play_target.py
import os
import time
import argparse
import numpy as np
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.TargetAviary import TargetAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger


# ---------- Defaults ----------
DEFAULT_MODEL_PATH = "results/best_model.zip"
DEFAULT_GUI = True
DEFAULT_OBS = ObservationType("kin")     # 'kin' or 'rgb'
DEFAULT_ACT = ActionType("rpm")          # 'rpm' | 'pid' | 'vel' | 'one_d_rpm' | 'one_d_pid'

# TargetAviary params (override via CLI if needed)
DEFAULT_EPISODE_LEN_SEC = 6000
DEFAULT_TARGET_STATIC = True
DEFAULT_TARGET_CENTER = (-1.5, 1.0, 1.0)
DEFAULT_TARGET_RADIUS = 0.4
DEFAULT_TARGET_OMEGA_HZ = 0.0
DEFAULT_TARGET_HEIGHT_WAVE = False
DEFAULT_SHOW_TARGET = True
DEFAULT_CAMERA_MODE = "follow"


def play(
    model_path: str = DEFAULT_MODEL_PATH,
    gui: bool = DEFAULT_GUI,
    # env/obs/act
    obs: ObservationType = DEFAULT_OBS,
    act: ActionType = DEFAULT_ACT,
    # TargetAviary tuning
    episode_len_sec: int = DEFAULT_EPISODE_LEN_SEC,
    target_static: bool = DEFAULT_TARGET_STATIC,
    target_center: tuple = DEFAULT_TARGET_CENTER,
    target_radius: float = DEFAULT_TARGET_RADIUS,
    target_omega_hz: float = DEFAULT_TARGET_OMEGA_HZ,
    target_height_wave: bool = DEFAULT_TARGET_HEIGHT_WAVE,
    show_target: bool = DEFAULT_SHOW_TARGET,
    camera_mode: str = DEFAULT_CAMERA_MODE,
):
    # ---- Load model ----
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        return

    model = PPO.load(model_path, device="cpu")
    print(f"[INFO] Loaded model from {model_path}")

    # ---- Build env (single-agent) ----
    env = TargetAviary(
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



    logger = Logger(
        logging_freq_hz=int(env.CTRL_FREQ),
        num_drones=1,
        output_folder="logs_playback_target/",
        colab=False,
    )

    # ---- Run episode ----
    obs_, _ = env.reset(seed=42, options={})
    start = time.time()

    # One episode (+ a couple seconds for smooth ending)
    horizon = int((env.EPISODE_LEN_SEC + 2) * env.CTRL_FREQ)

    for i in range(horizon):
        action, _ = model.predict(obs_, deterministic=True)
        obs_, reward, terminated, truncated, info = env.step(action)

        # Logging for KIN obs (shape matches BaseRLAviary convention)
        if obs == ObservationType.KIN:
            obs2 = obs_.squeeze()
            act2 = np.array(action).squeeze()
            logger.log(
                drone=0,
                timestamp=i / env.CTRL_FREQ,
                state=np.hstack([
                    obs2[0:3],           # position
                    np.zeros(4),         # fill (q)
                    obs2[3:15],          # rest of KIN obs
                    act2 if act2.ndim == 0 else act2
                ]),
                control=np.zeros(12),
            )

        if gui:
            env.render()

        sync(i, start, env.CTRL_TIMESTEP)
        if terminated or truncated:
            print("rterminatefd: ", terminated)
            print("truncated: ", truncated)
            break

    env.close()
    if obs == ObservationType.KIN:
        logger.plot()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Play a saved PPO policy in TargetAviary.")
    p.add_argument("--model_path",      type=str,  default=DEFAULT_MODEL_PATH)
    p.add_argument("--gui",             type=lambda s: s.lower() == "true", default=DEFAULT_GUI)

    # Obs/Act (keep consistent with how the model was trained)
    p.add_argument("--obs",             type=str,  default="kin", help="kin|rgb")
    p.add_argument("--act",             type=str,  default="rpm",
                   help="rpm|pid|vel|one_d_rpm|one_d_pid")

    # TargetAviary params
    p.add_argument("--episode_len_sec", type=int,  default=DEFAULT_EPISODE_LEN_SEC)
    p.add_argument("--target_static",   type=lambda s: s.lower() == "true", default=DEFAULT_TARGET_STATIC)
    p.add_argument("--target_center",   type=str,  default="2.0,2.0,2.0",
                   help="comma-separated triple, e.g. 3,0.0,1.0")
    p.add_argument("--target_radius",   type=float, default=DEFAULT_TARGET_RADIUS)
    p.add_argument("--target_omega_hz", type=float, default=DEFAULT_TARGET_OMEGA_HZ)
    p.add_argument("--target_height_wave", type=lambda s: s.lower() == "true", default=DEFAULT_TARGET_HEIGHT_WAVE)
    p.add_argument("--show_target",     type=lambda s: s.lower() == "true", default=DEFAULT_SHOW_TARGET)
    p.add_argument("--camera_mode",     type=str,  default=DEFAULT_CAMERA_MODE, help="follow|drone|static")

    args = p.parse_args()

    # Parse enums safely from strings
    obs_enum = ObservationType(args.obs)
    act_enum = ActionType(args.act)

    # Parse target_center "x,y,z" -> tuple(float,float,float)
    tc = tuple(float(x.strip()) for x in args.target_center.split(","))

    play(
        model_path=args.model_path,
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
