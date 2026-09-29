# reward_ethics.py

from ethics import compute_ethics

W4 = 0.7


class ESRMRewardEthics:

    def __init__(self, base_rewarder):
        self.base = base_rewarder

    def __call__(self, obs, action):

        # ✅ BASE REWARD
        base_reward, components = self.base(
            obs,
            action
        )

      #  S = components["Safety"]
      #  T = components["Throughput"]
      #  F = components["Fairness"]

        # ==================================================
        # ✅ CONFLICT EXTRACTION (verbessert)
        # ==================================================
        conflicts = []

        ttc = obs.get("ttc_min", None)
        dist = obs.get("dist_min", None)
        drac = obs.get("drac_min", None)     
             
        if ttc is not None and dist is not None:
            
            n_peds = obs.get("n_peds", 0)
            n_cars = obs.get("n_cars", 0)
            n_buses = obs.get("n_buses", 0)
            
     
            if n_peds > 0:
                vulnerability = 1.0        # pedestrians
            elif n_buses > 0:
                vulnerability = 0.2        # bus occupants
            else:
                vulnerability = 0.3        # car occupants
       

            conflicts.append({
                "TTC": ttc,
                "Dmin": dist,
                "DRAC": drac if drac is not None else 0.0,
                "conflict_count": 0,
                "vulnerability": vulnerability
            })



        # ==================================================
        # ✅ ETHICS (immer berechnet)
        # ==================================================
        E = compute_ethics(conflicts)

        ethical_factor = max(0.0, 1 + W4 * E)

        # ✅ FINAL REWARD
        reward = base_reward * ethical_factor

        # ==================================================
        # ✅ LOGGING
        # ==================================================
        components.update({
            "E": E,
            "ethical_factor": ethical_factor,
            "U_Base": base_reward,
            "reward_ethics": reward,
        })

        return reward, components