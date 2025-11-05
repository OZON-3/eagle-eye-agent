"""
PPO training/eval for TargetAviary (follow a moving/optional static target).

Examples (PowerShell / Windows):
    # Kinematic observations + direct motor RPM control (fastest to get running)
    python learn_target.py --obs kin --act rpm --gui true

    # Vision observations (RGB) + PID waypoint control (uses built-in controller)
    python learn_target.py --obs rgb --act pid --gui true

    # Vision + velocity control with PID
    python learn_target.py --obs rgb --act vel --gui true

Notes
-----
- If obs == rgb, we use CnnPolicy and wrap the env with VecTransposeImage.
- If obs == kin, we use MlpPolicy.
- This script is single-agent (TargetAviary); Multi* variants not used here.
"""

import os
import time
from datetime import datetime
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.envs.TargetAviary import TargetAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_OBS = ObservationType.KIN       # 'kin' or 'rgb'
DEFAULT_ACT = ActionType.RPM            # 'rpm' | 'pid' | 'vel' | 'one_d_rpm' | 'one_d_pid'
DEFAULT_TOTAL_TIMESTEPS = int(1e6)      # tune as you like
DEFAULT_EVAL_FREQ = 5000                # how often to eval/save
DEFAULT_SEED = 0

def parse_obs(s: str) -> ObservationType:
    s = s.strip().lower()
    return ObservationType.RGB if s == "rgb" else ObservationType.KIN

def parse_act(s: str) -> ActionType:
    s = s.strip().lower()
    table = {
        "rpm": ActionType.RPM,
        "pid": ActionType.PID,
        "vel": ActionType.VEL,
        "one_d_rpm": ActionType.ONE_D_RPM,
        "one_d_pid": ActionType.ONE_D_PID,
        "one-d-rpm": ActionType.ONE_D_RPM,
        "one-d-pid": ActionType.ONE_D_PID,
    }
    if s not in table:
        raise ValueError(f"Unknown act '{s}'. Use one of: rpm, pid, vel, one_d_rpm, one_d_pid")
    return table[s]

def make_env(obs: ObservationType,
             act: ActionType,
             gui: bool,
             record: bool,
             seed: int):
    # You can tweak TargetAviary params here (e.g., static vs moving target)
    def _thunk():
        return TargetAviary(
            gui=gui,
            record=record,
            obs=obs,
            act=act,
            # Target/episode knobs you might expose as CLI later:
            episode_len_sec=120,
            target_static=True,
            target_center=(1.5, 0.0, 1.0),  # start a bit away
            target_radius=0.0,
            target_omega_hz=0.0,
            target_height_wave=False,
            yaw_align_weight=0.3,
            smooth_action_weight=0.02,
            success_radius=0.15, show_target=True, camera_mode="follow"
        )
    # SB3 likes vectorized envs; keep n_envs=1 for now to avoid vision memory spikes
    env = make_vec_env(_thunk, n_envs=1, seed=seed)
    # If obs is RGB, transpose to CHW for SB3 CNN policy
    if obs == ObservationType.RGB:
        env = VecTransposeImage(env)  # (N,H,W,C)->(N,C,H,W)
    return env

def select_policy(obs: ObservationType) -> str:
    # SB3 policy strings: "MlpPolicy" or "CnnPolicy"
    return "CnnPolicy" if obs == ObservationType.RGB else "MlpPolicy"

def run(output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VIDEO,
        obs: ObservationType = DEFAULT_OBS,
        act: ActionType = DEFAULT_ACT,
        total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
        eval_freq: int = DEFAULT_EVAL_FREQ,
        seed: int = DEFAULT_SEED):

    # -----------------------------
    # Setup folders
    # -----------------------------
    save_dir = os.path.join(output_folder, 'target_' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # Train & Eval envs
    # -----------------------------
    train_env = make_env(obs=obs, act=act, gui=True, record=False, seed=seed)
    eval_env  = make_env(obs=obs, act=act, gui=False, record=False, seed=seed+1)

    # For info
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    # -----------------------------
    # PPO model
    # -----------------------------
    policy = select_policy(obs)
    model = PPO(
        policy,
        train_env,
        verbose=1,
        # device="cuda",
        tensorboard_log=None  # set to save_dir+"/tb" if you want TB logs
    )

    # -----------------------------
    # Callbacks (save best)
    # -----------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )

    print("[INFO] Starting training...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, log_interval=100)
    model.save(os.path.join(save_dir, 'final_model.zip'))
    print("[INFO] Training complete. Models in:", save_dir)

    # -----------------------------
    # Report eval curve (if any)
    # -----------------------------
    eval_npz = os.path.join(save_dir, 'evaluations.npz')
    if os.path.isfile(eval_npz):
        with np.load(eval_npz) as data:
            for j in range(data['timesteps'].shape[0]):
                print(f"{int(data['timesteps'][j])},{float(data['results'][j][0])}")

    # -----------------------------
    # Load best (if exists), else final
    # -----------------------------
    model_path = os.path.join(save_dir, 'best_model.zip')
    if not os.path.isfile(model_path):
        model_path = os.path.join(save_dir, 'final_model.zip')
    print("[INFO] Loading model:", model_path)
    model = PPO.load(model_path,  device="cuda",
                verbose=1)

    # -----------------------------
    # “Play” the policy with GUI
    # -----------------------------
    test_env = make_env(obs=obs, act=act, gui=gui, record=record_video, seed=seed+123)
    # unwrap to get access to underlying Aviary for logging
    # (VecTransposeImage wraps the VecEnv, so we reach .envs[0] twice if needed)
    base_env = getattr(test_env, 'venv', test_env)
    base_env = getattr(base_env, 'envs', [base_env])[0]

    # Logger assumes kinematic obs; if RGB, we just skip plotting
    logger = Logger(
        logging_freq_hz=int(base_env.CTRL_FREQ),
        num_drones=1,
        output_folder=output_folder,
        colab=False
    )

    # quick quantitative check (no rendering)
    nogui_env = make_env(obs=obs, act=act, gui=False, record=False, seed=seed+321)
    mean_reward, std_reward = evaluate_policy(model, nogui_env, n_eval_episodes=5)
    print(f"\n[INFO] Mean eval reward: {mean_reward:.3f} ± {std_reward:.3f}\n")

    obs_reset = test_env.reset(seed=42)
    if isinstance(obs_reset, tuple):  # gymnasium VecEnv sometimes returns (obs, info)
        obs, _info = obs_reset
    else:
        obs = obs_reset

    start = time.time()
    # run a short showcase episode (TargetAviary sets EPISODE_LEN_SEC internally)
    # we will just run for ~EPISODE_LEN_SEC + 2 seconds
    showcase_steps = 0
    # try to read EPISODE_LEN_SEC from the underlying env
    ep_len_sec = getattr(base_env, 'EPISODE_LEN_SEC', 120)
    steps_to_run = int((ep_len_sec + 2) * base_env.CTRL_FREQ)

    for i in range(steps_to_run):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)

        # render only if GUI True
        if gui:
            base_env.render()

        # Optional: log state when obs is kinematic
        if obs is not None and obs.shape[-1] != 4:  # likely KIN, not (H,W,4)
            try:
                # try to fetch the raw underlying obs & action for the logger
                state_like = base_env._getDroneStateVector(0)
                act_like = np.array(action).reshape(-1)
                logger.log(
                    drone=0,
                    timestamp=i / base_env.CTRL_FREQ,
                    state=np.hstack([state_like[0:3], np.zeros(4), state_like[3:15], act_like[:4] if act_like.size >= 4 else np.zeros(4)]),
                    control=np.zeros(12)
                )
            except Exception:
                pass

        showcase_steps += 1

        # sync to real-time-ish
        try:
            from gym_pybullet_drones.utils.utils import sync
            sync(i, start, base_env.CTRL_TIMESTEP)
        except Exception:
            pass

        # termination check (VecEnv: terminated/truncated are arrays)
        done_now = False
        if isinstance(terminated, (list, tuple, np.ndarray)):
            done_now = bool(terminated[0] or truncated[0])
        else:
            done_now = bool(terminated or truncated)
        if done_now:
            _ = test_env.reset(seed=42)

    test_env.close()

    # plot only if we logged kinematic data
    if obs == ObservationType.KIN:
        logger.plot()

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO for TargetAviary (follow target).")
    parser.add_argument("--gui",                default=DEFAULT_GUI,            type=str2bool, help="Use PyBullet GUI (default: True)")
    parser.add_argument("--record_video",       default=DEFAULT_RECORD_VIDEO,   type=str2bool, help="Record video (default: False)")
    parser.add_argument("--output_folder",      default=DEFAULT_OUTPUT_FOLDER,  type=str,      help='Folder to save logs/models')
    parser.add_argument("--obs",                default="kin",                  type=str,      help='Observation type: "kin" or "rgb"')
    parser.add_argument("--act",                default="rpm",                  type=str,      help='Action type: rpm | pid | vel | one_d_rpm | one_d_pid')
    parser.add_argument("--total_timesteps",    default=DEFAULT_TOTAL_TIMESTEPS,type=int,      help='Training steps')
    parser.add_argument("--eval_freq",          default=DEFAULT_EVAL_FREQ,      type=int,      help='Eval/save frequency (steps)')
    parser.add_argument("--seed",               default=DEFAULT_SEED,           type=int,      help='Random seed')
    args = parser.parse_args()

    run(
        output_folder=args.output_folder,
        gui=args.gui,
        record_video=args.record_video,
        obs=parse_obs(args.obs),
        act=parse_act(args.act),
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        seed=args.seed
    )
