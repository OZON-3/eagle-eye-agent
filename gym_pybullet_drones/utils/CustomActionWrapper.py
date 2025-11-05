# --- Add these imports near the top (if not present) ---
from gymnasium import ActionWrapper, spaces
import gymnasium as gym
import numpy as np


# --- Insert FlatMixedActionWrapper class (exactly as earlier) ---
class CustomActionWrapper(ActionWrapper):
    """
    Mixed discrete+continuous flat-vector wrapper.

    - binned_channels: list of ints (indices in underlying Box action) to discretize (e.g. [1,3])
    - bins_per_channel: int or list (if int, same bins for each binned channel)
    - Exposes a single flat Box action of length n_disc + n_cont:
        action[:n_disc] -> integers (as floats) for discrete indices (will be rounded/clipped)
        action[n_disc:] -> continuous values for passthrough channels
    """
    def __init__(self, env: gym.Env, binned_channels, bins_per_channel=5, dtype=np.float32):
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Box):
            raise ValueError("Underlying env.action_space must be Box")
        low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
        self.box_low = low
        self.box_high = high
        self.dim = int(self.box_low.shape[0])

        self.binned_channels = [int(x) for x in binned_channels]
        if any((ch < 0 or ch >= self.dim) for ch in self.binned_channels):
            raise ValueError("binned channel index out of range")

        if isinstance(bins_per_channel, int):
            self.bins = [int(bins_per_channel)] * len(self.binned_channels)
        else:
            self.bins = [int(x) for x in bins_per_channel]
            if len(self.bins) != len(self.binned_channels):
                raise ValueError("bins_per_channel length must equal binned_channels length")

        self.cont_channels = [i for i in range(self.dim) if i not in self.binned_channels]

        self._values = []
        for ch_idx, n_bins in zip(self.binned_channels, self.bins):
            lowv = float(self.box_low[ch_idx])
            highv = float(self.box_high[ch_idx])
            if n_bins == 1:
                vals = np.array([0.5 * (lowv + highv)], dtype=np.float32)
            else:
                vals = np.linspace(lowv, highv, num=int(n_bins), dtype=np.float32)
            self._values.append(vals)

        self.n_disc = len(self.binned_channels)
        self.n_cont = len(self.cont_channels)

        disc_low = np.zeros(self.n_disc, dtype=np.float32)
        disc_high = np.array([b - 1 for b in self.bins], dtype=np.float32)
        if self.n_cont > 0:
            cont_low = self.box_low[self.cont_channels].astype(np.float32)
            cont_high = self.box_high[self.cont_channels].astype(np.float32)
            low_vec = np.concatenate([disc_low, cont_low], axis=0).astype(np.float32)
            high_vec = np.concatenate([disc_high, cont_high], axis=0).astype(np.float32)
        else:
            low_vec = disc_low
            high_vec = disc_high

        self.action_space = spaces.Box(low=low_vec, high=high_vec, dtype=dtype)
        self.observation_space = env.observation_space

    def action(self, flat_action):
        a = np.asarray(flat_action).reshape(-1)
        if a.shape[0] != (self.n_disc + self.n_cont):
            raise ValueError(f"Expected action length {self.n_disc + self.n_cont}, got {a.shape[0]}")

        disc = a[:self.n_disc].astype(int)
        cont = a[self.n_disc:].astype(np.float32) if self.n_cont > 0 else np.array([], dtype=np.float32)

        full = np.zeros(self.dim, dtype=np.float32)

        for i, ch in enumerate(self.cont_channels):
            full[ch] = cont[i]

        for i, (idx, vals) in enumerate(zip(disc, self._values)):
            idx_clamped = int(np.clip(int(idx), 0, len(vals) - 1))
            ch = self.binned_channels[i]
            full[ch] = vals[idx_clamped]

        full = np.clip(full, self.box_low, self.box_high)
        # Return as a row for the env (num_drones x action_dim). TargetAviary expects shape (1, 4).
        return full.reshape(1, -1)  # shape -> (1, dim)
