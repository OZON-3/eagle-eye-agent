# gym_pybullet_drones/utils/DiscreteVelWrapper.py
import gymnasium as gym
import numpy as np

class DiscreteVelWrapper(gym.ActionWrapper):
    """
    Expose a MultiDiscrete policy head (true categorical PPO) for 4 DOF velocity control.
    Converts bin indices -> velocity bin centers before stepping the inner env.

    New: optional `centers` argument to use non-uniform bin centers (e.g., tanh-spaced).
    """
    def __init__(self, env,
                 bins=(21, 21, 15, 21),
                 ranges=((-0.6, 0.6),  # vx
                         (-0.6, 0.6),  # vy
                         (-0.4, 0.4),  # vz
                         (-0.6, 0.6)), # yaw_rate
                 centers=None):
        super().__init__(env)

        assert len(bins) == 4 and len(ranges) == 4
        self.bins = tuple(int(b) for b in bins)
        self.ranges = tuple(tuple(r) for r in ranges)

        # Precompute bin centers
        # - If `centers` (tuple of 4 arrays) is provided, use it directly
        # - Else fall back to uniform linspace
        if centers is not None:
            assert len(centers) == 4, "centers must be a 4-tuple of 1D arrays"
            _c = []
            for i, (c, n) in enumerate(zip(centers, self.bins)):
                c = np.asarray(c, dtype=np.float32).reshape(-1)
                assert c.shape[0] == n, f"centers[{i}] must have length {n}"
                _c.append(c)
            self._centers = tuple(_c)
        else:
            _c = []
            for (lo, hi), n in zip(self.ranges, self.bins):
                _c.append(np.linspace(lo, hi, n, dtype=np.float32))
            self._centers = tuple(_c)

        # Expose MultiDiscrete to SB3 (true categorical policy)
        self.action_space = gym.spaces.MultiDiscrete(self.bins)

        # Observation space unchanged
        self.observation_space = env.observation_space

        # If the inner env needs (NUM_DRONES,4), remember it:
        self._num_drones = getattr(env, "NUM_DRONES", 1)

    def action(self, act_idx):
        """
        Map integer bins -> float velocities, return shape (NUM_DRONES, 4)
        """
        a_idx = np.asarray(act_idx, dtype=np.int32).reshape(-1)  # (4,)
        assert a_idx.shape[0] == 4, f"Expected 4 discrete dims, got {a_idx.shape}"

        # clip indices just in case
        for i in range(4):
            a_idx[i] = int(np.clip(a_idx[i], 0, self.bins[i]-1))

        vx = self._centers[0][a_idx[0]]
        vy = self._centers[1][a_idx[1]]
        vz = self._centers[2][a_idx[2]]
        wz = self._centers[3][a_idx[3]]

        a_float = np.array([[vx, vy, vz, wz]], dtype=np.float32)  # (1,4)

        if self._num_drones > 1:
            a_float = np.repeat(a_float, self._num_drones, axis=0)
        return a_float

    def reverse_action(self, a_float):
        # float -> nearest bin (optional)
        a = np.asarray(a_float, dtype=np.float32).reshape(-1)
        out = []
        for i in range(4):
            centers = self._centers[i]
            idx = int(np.abs(centers - a[i]).argmin())
            out.append(int(np.clip(idx, 0, self.bins[i]-1)))
        return np.array(out, dtype=np.int32)
