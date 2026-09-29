# stf_callback.py

from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import numpy as np

class STFLoggerCallback(BaseCallback):

    def __init__(self, total_timesteps, print_every_steps, print_every_percent, verbose=0):
        super().__init__(verbose)

        self.total_timesteps = total_timesteps
        self.print_every_steps = print_every_steps
        self.print_every_percent = print_every_percent

        self._last_print_step = 0
        self._last_print_percent = 0.0

        # --------------------------------------------------
        # HISTORY BUFFERS
        # --------------------------------------------------
        self.window_size = 1000
        self.S_hist = deque(maxlen=self.window_size)
        self.T_hist = deque(maxlen=self.window_size)
        self.F_hist = deque(maxlen=self.window_size)
        self.U_hist = deque(maxlen=self.window_size)

        # Interaction-only buffers
        #self.S_active_hist = deque(maxlen=self.window_size)
        #self.T_active_hist = deque(maxlen=self.window_size)
        #self.F_active_hist = deque(maxlen=self.window_size)
        #self.U_active_hist = deque(maxlen=self.window_size)

        # Interaction rate tracking
        self.interaction_hist = deque(maxlen=self.window_size)
        
        self.imp_S_hist = deque(maxlen=self.window_size)
        self.imp_T_hist = deque(maxlen=self.window_size)
        self.imp_F_hist = deque(maxlen=self.window_size)
        self.imp_E_hist = deque(maxlen=self.window_size)

    def _on_step(self):

        step = self.num_timesteps
        if step > self.total_timesteps:
            return True

        info = self.locals.get("infos", [{}])[0]
        metrics = info.get("metrics", {})

        interaction_active = bool(metrics.get("interaction_active", False))

        S = metrics.get("Safety", 0.0)
        T = metrics.get("Throughput", 0.0)
        F = metrics.get("Fairness", 0.0)
        U = float(self.locals.get("rewards", [0.0])[0])

        # --------------------------------------------------
        # INTERACTION FLAG
        # --------------------------------------------------
        self.logger.record("interaction/active", int(interaction_active))

        # --------------------------------------------------
        # RAW VALUES (wichtig für Verständnis!)
        # --------------------------------------------------
        self.logger.record("raw/S", S)
        self.logger.record("raw/T", T)
        self.logger.record("raw/F", F)

        # --------------------------------------------------
        # ✅ IMPORTANCE (U1 baseline)
        # --------------------------------------------------
        # ✅ gleiche weights wie im Reward (U1!)
        w1, w2, w3 = 0.5, 0.3, 0.2

        S_c = w1 * S
        T_c = w2 * T
        F_c = w3 * F
        E_c = 0.0

        norm = max(abs(S_c) + abs(T_c) + abs(F_c), 1e-6)

        imp_S = S_c / norm
        imp_T = T_c / norm
        imp_F = F_c / norm
        imp_E = 0.0

        self.logger.record("importance/S", imp_S)
        self.logger.record("importance/T", imp_T)
        self.logger.record("importance/F", imp_F)
        self.logger.record("importance/E", imp_E)
        
        # --------------------------------------------------
        # ✅ ETHICS PLACEHOLDER (für U1 vs Duarte Vergleich)
        # --------------------------------------------------
        self.logger.record("ethics/E", 0.0)
        self.logger.record("ethics/impact", 0.0)
         
        # --------------------------------------------------
        # SAFETY DECOMPOSITION
        # --------------------------------------------------
        if "s_ttc" in metrics:
            self.logger.record("safety/s_ttc", metrics["s_ttc"])
        if "s_dist" in metrics:
            self.logger.record("safety/s_dist", metrics["s_dist"])
        if "s_drac" in metrics:
            self.logger.record("safety/s_drac", metrics["s_drac"])

        # --------------------------------------------------
        # THROUGHPUT DECOMPOSITION
        # --------------------------------------------------
        if "t_car" in metrics:
            self.logger.record("throughput/t_car", metrics["t_car"])
        if "t_bus" in metrics:
            self.logger.record("throughput/t_bus", metrics["t_bus"])
        if "t_vru" in metrics:
            self.logger.record("throughput/t_vru", metrics["t_vru"])

        # --------------------------------------------------
        # FAIRNESS DECOMPOSITION
        # --------------------------------------------------
        if "x_car" in metrics:
            self.logger.record("fairness/x_car", metrics["x_car"])
        if "x_bus" in metrics:
            self.logger.record("fairness/x_bus", metrics["x_bus"])
        if "x_vru" in metrics:
            self.logger.record("fairness/x_vru", metrics["x_vru"])

        # --------------------------------------------------
        # COLLECT GLOBAL HISTORY
        # --------------------------------------------------
        self.S_hist.append(S)
        self.T_hist.append(T)
        self.F_hist.append(F)
        self.U_hist.append(U)

        self.interaction_hist.append(int(interaction_active))

        # --------------------------------------------------
        # INTERACTION-ONLY DATA
        # --------------------------------------------------
       # if interaction_active:
       #     self.S_active_hist.append(S)
       #     self.T_active_hist.append(T)
       #     self.F_active_hist.append(F)
       #     self.U_active_hist.append(U)

           # self.logger.record("reward_active/S", 0.5 * S)
           # self.logger.record("reward_active/T", 0.3 * T)
           # self.logger.record("reward_active/F", 0.2 * F)

        # --------------------------------------------------
        # HELPER: safe correlation
        # --------------------------------------------------
       # def safe_corr(a, b):
       #     if len(a) < 50 or len(b) < 50:
       #         return 0.0
       #     if np.std(a) < 1e-6 or np.std(b) < 1e-6:
       #         return 0.0
       #     return float(np.corrcoef(a, b)[0, 1])

        # --------------------------------------------------
        # INTERACTION-ONLY CORRELATIONS (zentral!)
        # --------------------------------------------------
       # if len(self.U_active_hist) > 50:
       #    self.logger.record("corr_active/S_U", safe_corr(self.S_active_hist, self.U_active_hist))
       #   self.logger.record("corr_active/T_U", safe_corr(self.T_active_hist, self.U_active_hist))
       #  self.logger.record("corr_active/F_U", safe_corr(self.F_active_hist, self.U_active_hist))

        # --------------------------------------------------
        # INTERACTION RATE
        # --------------------------------------------------
        if len(self.interaction_hist) > 50:
            interaction_rate = sum(self.interaction_hist) / len(self.interaction_hist)
            self.logger.record("interaction/rate", interaction_rate)

        # --------------------------------------------------
        # ✅ IMPORTANCE MEAN
        # --------------------------------------------------
        self.imp_S_hist.append(imp_S)
        self.imp_T_hist.append(imp_T)
        self.imp_F_hist.append(imp_F)
        self.imp_E_hist.append(imp_E)

        if len(self.imp_S_hist) > 50:
            self.logger.record("importance_mean/S", float(np.mean(self.imp_S_hist)))
            self.logger.record("importance_mean/T", float(np.mean(self.imp_T_hist)))
            self.logger.record("importance_mean/F", float(np.mean(self.imp_F_hist)))
            self.logger.record("importance_mean/E", float(np.mean(self.imp_E_hist)))

        # --------------------------------------------------
        # REWARD VARIANCE (Stabilität)
        # --------------------------------------------------
        if len(self.U_hist) > 50:
            self.logger.record("stats/reward_variance", float(np.var(self.U_hist)))

        return True