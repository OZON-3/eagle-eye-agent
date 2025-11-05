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
from stable_baselines3.common.vec_env import VecNormalize

from gym_pybullet_drones.envs.TargetAviary import TargetAviary
from gym_pybullet_drones.utils.CustomActionWrapper import CustomActionWrapper
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

    binned_channels = [1, 3]  # roll and yaw
    bins_per_channel = 5  # 5 bins each

    # Use a factory wrapper for make_vec_env so each created env is wrapped
    def make_wrapped_env():
        def _init(**kwargs):
            base = TargetAviary(**env_kwargs)
            wrapped = CustomActionWrapper(base, binned_channels=binned_channels, bins_per_channel=bins_per_channel)
            return wrapped

        return _init

    train_env = make_vec_env(make_wrapped_env(), env_kwargs=env_kwargs, n_envs=4, seed=0)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ---------- Build eval env (normalized, sharing stats with train_env) ----------
    eval_vec_cb = make_vec_env(make_wrapped_env(), env_kwargs=env_kwargs, n_envs=1, seed=999)
    eval_env_cb = VecNormalize(eval_vec_cb, norm_obs=True, norm_reward=False, training=False)
    # share normalization stats with train_env
    eval_env_cb.obs_rms = train_env.obs_rms
    eval_env_cb.ret_rms = train_env.ret_rms

    print("[INFO] Action space:", train_env.action_space)
    print("[INFO] Observation space:", train_env.observation_space)

    def linear_schedule(start, end):
        def f(progress):  # progress goes 1.0 -> 0.0 over training
            return end + (start - end) * progress

        return f

    # ---------- Model ----------
    model = PPO(
        "MlpPolicy",
        train_env,  # your VecNormalize(make_vec_env(...)) env

        # Rollout collection
        # implicit via train_env
        n_steps=4096,  # per-env steps -> 4 * 4096 = 16384 rollout batch
        batch_size=1024,  # divides 16384 evenly (8 minibatches/epoch)
        n_epochs=6,  # enough optimization per batch without overfitting

        # Credit assignment
        gamma=0.99,
        gae_lambda=0.95,

        # Optimization
        learning_rate=linear_schedule(3e-4, 1e-5),  # gentle LR anneal over 10M steps
        clip_range=linear_schedule(0.20, 0.10),  # smaller clips late = stabler policy
        max_grad_norm=0.5,
        vf_coef=0.5,
        ent_coef=0.008,  # slight exploration; binned actions don’t need much

        # Safety guard (prevents bad updates)
        target_kl=0.03,  # stop overly large policy jumps

        # Logging / device
        tensorboard_log=filename + "/",
        verbose=1,
        device="cpu"  # switch to "cuda" if GPU is stable
    )

    # ---------- Callback ----------
    target_reward = 50000.0
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
    eval_callback = EvalCallback(
        eval_env_cb,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path=filename + "/",
        log_path=filename + "/",
        eval_freq=50000,
        deterministic=True,
        render=False,
    )

    # ---------- Train ----------
    total_steps = int(1e7) if local else int(1e3)
    print("about to train steps:", total_steps)
    model.learn(total_timesteps=total_steps, callback=eval_callback, log_interval=100)
    model.save(filename + "/final_model.zip")

    train_env.save(filename + "/vecnormalize.pkl")
    print("[INFO] Saved VecNormalize stats.")

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

    def make_wrapped_env_gui():
        def _init(**_):
            base = TargetAviary(**{**env_kwargs, "gui": gui, "record": record_video})
            return CustomActionWrapper(base, binned_channels=binned_channels, bins_per_channel=bins_per_channel)

        return _init

    # Build a vec env with wrapper and load VecNormalize stats
    show_vec = make_vec_env(make_wrapped_env_gui(), n_envs=1, seed=42)
    show_env = VecNormalize.load(filename + "/vecnormalize.pkl", show_vec)
    show_env.training = False
    show_env.norm_reward = False

    # Access underlying TargetAviary for logging/rendering metadata
    # Wrapper stack: DummyVecEnv -> VecNormalize -> CustomActionWrapper -> TargetAviary
    _base = show_env.venv.envs[0]  # CustomActionWrapper
    try:
        target_env = _base.env  # TargetAviary (if CustomActionWrapper wraps via .env)
    except AttributeError:
        target_env = _base  # If your wrapper forwards attributes directly

    logger = Logger(
        logging_freq_hz=int(target_env.CTRL_FREQ),
        num_drones=1,
        output_folder=output_folder,
        colab=colab,
    )

    # Run one episode
    obs = show_env.reset()  # shape: (1, obs_dim)
    start = time.time()
    horizon = int((target_env.EPISODE_LEN_SEC + 2) * target_env.CTRL_FREQ)

    for i in range(horizon):
        # model.predict accepts batched obs from VecEnv
        action, _ = model.predict(obs, deterministic=True)  # shape: (1, act_dim)

        # VecEnv step returns 4-tuple
        obs, rewards, dones, infos = show_env.step(action)

        # Optional logging (obs here is normalized & batched)
        if DEFAULT_OBS == ObservationType.KIN:
            obs2 = np.array(obs).squeeze()  # (obs_dim,)
            act2 = np.array(action).squeeze()  # (act_dim,)
            logger.log(
                drone=0,
                timestamp=i / target_env.CTRL_FREQ,
                state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
                control=np.zeros(12),
            )

        if gui:
            try:
                target_env.render()
            except Exception:
                pass

        sync(i, start, target_env.CTRL_TIMESTEP)

        # dones is a vector of length n_envs (here 1)
        if dones[0]:
            break

    # Close and plot
    try:
        target_env.close()
    except Exception:
        pass
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
