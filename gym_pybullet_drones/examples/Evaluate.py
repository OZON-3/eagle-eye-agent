import os
import numpy as np
import matplotlib.pyplot as plt

def plot_eval_curves(run_dir):
    """
    Expects 'evaluations.npz' in run_dir (created by EvalCallback).
    Plots mean ± std for reward and episode length vs timesteps.
    """
    npz_path = os.path.join(run_dir, "evaluations.npz")
    data = np.load(npz_path)

    ts = data["timesteps"]                      # shape: (K,)
    rewards = data["results"].squeeze()         # shape: (K, N_eval) or (K, 1)
    ep_lengths = data["ep_lengths"].squeeze()   # shape: (K, N_eval)

    rew_mean = rewards.mean(axis=1)
    rew_std = rewards.std(axis=1)
    len_mean = ep_lengths.mean(axis=1)
    len_std = ep_lengths.std(axis=1)

    # --- Reward ---
    plt.figure(figsize=(8,5))
    plt.plot(ts, rew_mean, label="Eval mean reward")
    plt.fill_between(ts, rew_mean-rew_std, rew_mean+rew_std, alpha=0.2, label="±1 std")
    plt.xlabel("Timesteps")
    plt.ylabel("Return")
    plt.title("Evaluation Return vs Timesteps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Episode length ---
    plt.figure(figsize=(8,5))
    plt.plot(ts, len_mean, label="Eval mean episode length")
    plt.fill_between(ts, len_mean-len_std, len_mean+len_std, alpha=0.2, label="±1 std")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode length (steps)")
    plt.title("Evaluation Episode Length vs Timesteps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example:
plot_eval_curves(r"results\save-11.03.2025_10.33.06")
