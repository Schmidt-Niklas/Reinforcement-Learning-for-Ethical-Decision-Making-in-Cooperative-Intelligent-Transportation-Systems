# env_ethics.py

from esrm_env.env import ESRMEnv
from reward_ethics import compute_reward_ethics


class ESRMEnvEthics(ESRMEnv):

    def step(self, action):

        obs, _, done, info = super().step(action)

        metrics = info.get("metrics", {})

        # ✅ KORREKT für dein System
        S = metrics.get("S", 0.0)
        T = metrics.get("T", 0.0)
        F = metrics.get("F", 0.0)

        # ✅ vorerst leer (bis du SUMO-Conflicts integrierst)
        conflicts = metrics.get("conflicts", [])

        # ✅ neuer Reward
        reward, E, ef = compute_reward_ethics(S, T, F, conflicts)

        # ✅ erweitertes Logging
        info["metrics"]["E"] = E
        info["metrics"]["ethical_factor"] = ef
        info["metrics"]["reward_ethics"] = reward

        return obs, reward, done, info