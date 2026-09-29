# reward.py – ESRM Reward (U1: Safety + Throughput + Fairness)

import math
from collections import deque

# --------------------------------------------------
# Normalization constants
# --------------------------------------------------
#REF_FLOW_CAR_PED = 0.028
#REF_FLOW_BUS = 0.0001666

MAX_CARS = 20
MAX_VRUS = 80
MAX_BUSES = 3

# --------------------------------------------------
# Throughput weights
# --------------------------------------------------
ALPHA = {
    "bus": 40 / 42.5,
    "car": 1.5 / 42.5,
    "vru": 1.0 / 42.5,
}

# --------------------------------------------------
# Expected waiting times
# --------------------------------------------------
E_WAIT = {
    "car": 70.0,
    "bus": 70.0,
    "vru": 1.0,     #Literature: Value Pedestrian = 40, but Simulation shows max waiting of VRUs in this specific intersection = 1sec
}

# --------------------------------------------------
# Safety normalization thresholds
# --------------------------------------------------
TTC_REF = 3.0
DIST_REF = 2.0
DRAC_REF = 2.5

# --------------------------------------------------
# Intervention costs
# --------------------------------------------------

ACTION_COST = {
    0: 0.000,
    1: 0.002,
    2: 0.002,
    3: 0.006,
    4: 0.010,
}


class ESRMRewardU1:

    def __init__(self):
        self.T_history = deque(maxlen=15)
        self.step_length = 0.1

    def __call__(self, obs, action):

        # --------------------------------------------------
        # SAFETY
        # --------------------------------------------------
        if obs["collision_occurred"]:
            S = 0.0
            s_ttc = s_dist = s_drac = 0.0
        else:
            ttc = obs["ttc_min"]
            dist = obs["dist_min"]
            drac = obs["drac_min"]

            s_ttc = 1 - max(0.0, (TTC_REF - ttc) / TTC_REF)
            s_dist = 1 - max(0.0, (DIST_REF - dist) / DIST_REF)
            s_drac = 1 - min(1.0, drac / DRAC_REF)

            S = 0.4 * s_ttc + 0.3 * s_dist + 0.3 * s_drac

        # --------------------------------------------------
        # THROUGHPUT
        # --------------------------------------------------

        # flow approximation
        t_car = obs["n_cars"] / MAX_CARS
        t_vru = obs["n_peds"] / MAX_VRUS
        t_bus = obs["n_buses"] / MAX_BUSES

        num = 0.0
        den = 0.0

        if obs["n_buses"] > 0:
            num += ALPHA["bus"] * t_bus
            den += ALPHA["bus"]

        if obs["n_cars"] > 0:
            num += ALPHA["car"] * t_car
            den += ALPHA["car"]

        if obs["n_peds"] > 0:
            num += ALPHA["vru"] * t_vru
            den += ALPHA["vru"]

        # absolute throughput
        T_raw = num / den if den > 1e-6 else 0.0

        # soft clipping
        T_raw = T_raw / (1.0 + T_raw)

        # smoothing
        self.T_history.append(T_raw)
        T = sum(self.T_history) / len(self.T_history) if self.T_history else 0.0

        # --------------------------------------------------
        # FAIRNESS
        # --------------------------------------------------
        w_car = obs["mean_wait_car"] or E_WAIT["car"]
        w_bus = obs["mean_wait_bus"] or E_WAIT["bus"]
        w_vru = obs["mean_wait_vru"] or E_WAIT["vru"]

        x_car = E_WAIT["car"] / w_car
        x_bus = E_WAIT["bus"] / w_bus
        x_vru = E_WAIT["vru"] / w_vru

        num_f = (x_car + x_bus + x_vru) ** 2
        den_f = 3 * (x_car**2 + x_bus**2 + x_vru**2)
        F = max(0.0, min(1.0, num_f / den_f))

        # --------------------------------------------------
        # FINAL REWARD
        # --------------------------------------------------
        reward = 0.5 * S + 0.3 * T + 0.2 * F
        
        
        reward -= ACTION_COST[action]

        # --------------------------------------------------
        # COMPONENTS (für Logging)
        # --------------------------------------------------
        components = {
            # Main metrics
            "Safety": S,
            "Throughput": T,
            "Fairness": F,

            # Safety decomposition
            "s_ttc": s_ttc,
            "s_dist": s_dist,
            "s_drac": s_drac,

            # ✅ NEW: Throughput decomposition
            "t_car": t_car,
            "t_bus": t_bus,
            "t_vru": t_vru,

            # ✅ NEW: Fairness decomposition
            "x_car": x_car,
            "x_bus": x_bus,
            "x_vru": x_vru,
            
            "action_cost": ACTION_COST[action],
        }

        return reward, components