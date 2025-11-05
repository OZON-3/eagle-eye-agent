"""
Minimal PPO training script for gym_pybullet_drones TargetAviary.

Example
-------
In a terminal, run as:

    $ python learn_target_min.py --gui true
    $ python learn_target_min.py --gui false
    $ python learn_target_min.py --gui true --episode_len_sec 90 --target_static false --target_radius 0.6 --target_omega_hz 0.05

Notes
-----
This is a single-environment example integrating TargetAviary with
the reinforcement-learning library Stable-Baselines3 (PPO).
"""

import os
import time
from datetime import datetime
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.envs.TargetAviary import TargetAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


# ---------------- Defaults ----------------
DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False
DEFAULT_LOCAL = True   # If False -> very short training (for CI/tests)

DEFAULT_OBS = ObservationType("kin")      # 'kin' or 'rgb'
DEFAULT_ACT = ActionType("rpm")           # 'rpm' | 'pid' | 'vel' | 'one_d_rpm' | 'one_d_pid'

# TargetAviary parameters
DEFAULT_EPISODE_LEN_SEC = 30
DEFAULT_TARGET_STATIC = True
DEFAULT_TARGET_CENTER = (1.5, 1.0, 1.0)
DEFAULT_TARGET_RADIUS = 0.4
DEFAULT_TARGET_OMEGA_HZ = 0.0
DEFAULT_TARGET_HEIGHT_WAVE = False
DEFAULT_SHOW_TARGET = True
DEFAULT_CAMERA_MODE = "follow"


def run(gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VIDEO,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        plot=True,
        colab=DEFAULT_COLAB,
        local=DEFAULT_LOCAL):
    """
    PPO training, evaluation, and test run for TargetAviary.
    """

    filename = os.path.join(output_folder, "save-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    os.makedirs(filename, exist_ok=True)

    # ---------- Train/Eval environments (headless for stability) ----------
    env_kwargs = dict(
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        episode_len_sec=DEFAULT_EPISODE_LEN_SEC,
        target_static=DEFAULT_TARGET_STATIC,
        target_center=DEFAULT_TARGET_CENTER,
        target_radius=DEFAULT_TARGET_RADIUS,
        target_omega_hz=DEFAULT_TARGET_OMEGA_HZ,
        target_height_wave=DEFAULT_TARGET_HEIGHT_WAVE,
        show_target=DEFAULT_SHOW_TARGET,
        camera_mode=DEFAULT_CAMERA_MODE,
        gui=False,
        record=False,
    )

    train_env = make_vec_env(TargetAviary, env_kwargs=env_kwargs, n_envs=1, seed=0)
    eval_env  = TargetAviary(**env_kwargs)

    print("[INFO] Action space:", train_env.action_space)
    print("[INFO] Observation space:", train_env.observation_space)

    # ---------- Model ----------
    model = PPO(
        "MlpPolicy" if DEFAULT_OBS == ObservationType.KIN else "CnnPolicy",
        train_env,
        verbose=1,
        # device="cuda",
        tensorboard_log=filename + "/",
        device="cpu",  # safer with your GPU setup
    )

    # ---------- Callback ----------
    target_reward = 50000.0
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path=filename + "/",
        log_path=filename + "/",
        eval_freq=1000,
        deterministic=True,
        render=False,
    )

    # ---------- Train ----------
    total_steps = int(1e6) if local else int(1e3)
    print("about to train steps:", total_steps)
    model.learn(total_timesteps=total_steps, callback=eval_callback, log_interval=100)
    model.save(filename + "/final_model.zip")
    print("[INFO] Training done. Model saved to:", filename)

    # ---------- Print progression ----------
    with np.load(filename + "/evaluations.npz") as data:
        for j in range(data["timesteps"].shape[0]):
            print(str(data["timesteps"][j]) + "," + str(data["results"][j][0]))

    if local:
        input("Press Enter to continue...")

    # ---------- Load best model ----------
    path = filename + "/best_model.zip"
    if not os.path.isfile(path):
        alt = filename + "/final_model.zip"
        if os.path.isfile(alt):
            path = alt
        else:
            print("[ERROR] No model found under", filename)
            return
    model = PPO.load(path, device="cpu")

    # ---------- Showcase ----------
    test_env = TargetAviary(**{**env_kwargs, "gui": gui, "record": record_video})
    test_env_nogui = TargetAviary(**env_kwargs)

    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=1,
        output_folder=output_folder,
        colab=colab,
    )

    if local:
        input("Press Enter to continue...")

    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
    print(f"\n[INFO] Mean eval reward: {mean_reward:.2f} ± {std_reward:.2f}\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()

        # Log state
        if DEFAULT_OBS == ObservationType.KIN:
            logger.log(
                drone=0,
                timestamp=i / test_env.CTRL_FREQ,
                state=np.hstack([
                    obs2[0:3], np.zeros(4), obs2[3:15],
                    act2 if act2.ndim == 0 else act2
                ]),
                control=np.zeros(12),
            )

        if gui:
            test_env.render()

        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs, info = test_env.reset(seed=42, options={})

    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TargetAviary PPO Example")
    parser.add_argument("--gui",                default=DEFAULT_GUI,           type=str2bool, help="Enable PyBullet GUI")
    parser.add_argument("--record_video",       default=DEFAULT_RECORD_VIDEO,  type=str2bool, help="Record simulation video")
    parser.add_argument("--output_folder",      default=DEFAULT_OUTPUT_FOLDER, type=str,      help="Results folder")
    parser.add_argument("--colab",              default=DEFAULT_COLAB,         type=bool,     help="Running in Colab")
    parser.add_argument("--local",              default=DEFAULT_LOCAL,         type=str2bool, help="Shorter training for tests")
    ARGS = parser.parse_args()

    run(**vars(ARGS))
