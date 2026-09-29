# stfe_callback.py

from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import numpy as np


class STFELoggerCallback(BaseCallback):

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
        self.E_active_hist = deque(maxlen=self.window_size)

        self.interaction_hist = deque(maxlen=self.window_size)
        
        # ✅ NEW: IMPORTANCE HISTORY
        self.importance_S_hist = deque(maxlen=self.window_size)
        self.importance_T_hist = deque(maxlen=self.window_size)
        self.importance_F_hist = deque(maxlen=self.window_size)
        self.importance_E_hist = deque(maxlen=self.window_size)

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

        E = metrics.get("E", 0.0)
        ef = metrics.get("ethical_factor", 1.0)
        
        # --------------------------------------------------
        # LOGGING RAW
        # --------------------------------------------------
        
        self.logger.record("interaction/active", int(interaction_active))

        self.logger.record("raw/S", S)
        self.logger.record("raw/T", T)
        self.logger.record("raw/F", F)

        self.logger.record("ethics/E", E)
      #  self.logger.record("ethics/factor", ef)

        # --------------------------------------------------
        # ✅ SAFETY DECOMPOSITION (U1 zurückbringen)
        # --------------------------------------------------
        if "s_ttc" in metrics:
            self.logger.record("safety/s_ttc", metrics["s_ttc"])
        if "s_dist" in metrics:
            self.logger.record("safety/s_dist", metrics["s_dist"])
        if "s_drac" in metrics:
            self.logger.record("safety/s_drac", metrics["s_drac"])

        # --------------------------------------------------
        # ✅ THROUGHPUT DECOMPOSITION
        # --------------------------------------------------
        if "t_car" in metrics:
            self.logger.record("throughput/t_car", metrics["t_car"])
        if "t_bus" in metrics:
            self.logger.record("throughput/t_bus", metrics["t_bus"])
        if "t_vru" in metrics:
            self.logger.record("throughput/t_vru", metrics["t_vru"])

        # --------------------------------------------------
        # ✅ FAIRNESS DECOMPOSITION
        # --------------------------------------------------
        if "x_car" in metrics:
            self.logger.record("fairness/x_car", metrics["x_car"])
        if "x_bus" in metrics:
            self.logger.record("fairness/x_bus", metrics["x_bus"])
        if "x_vru" in metrics:
            self.logger.record("fairness/x_vru", metrics["x_vru"])


        # --------------------------------------------------
        # HISTORY
        # --------------------------------------------------
        
        self.interaction_hist.append(int(interaction_active))

        if interaction_active:
            self.E_active_hist.append(E)
        # --------------------------------------------------
        # INTERACTION RATE
        # --------------------------------------------------
        if len(self.interaction_hist) > 50:
            interaction_rate = sum(self.interaction_hist) / len(self.interaction_hist)
            self.logger.record("interaction/rate", interaction_rate)

        # --------------------------------------------------
        # REWARD VARIANCE
        # --------------------------------------------------
       # if len(self.U_hist) > 50:
        #    self.logger.record("stats/reward_variance", float(np.var(self.U_hist)))

        # --------------------------------------------------
        # ETHICS (ACTIVE)
        # --------------------------------------------------      
        if len(self.E_active_hist) > 50:
            E_active_arr = np.array(self.E_active_hist)
            self.logger.record("ethics_active/violation_lt_-0.5", np.mean(E_active_arr < -0.5))
            self.logger.record("ethics_active/violation_lt_-1.0", np.mean(E_active_arr < -1.0))
            self.logger.record("ethics_active/E_mean", float(np.mean(E_active_arr)))

      
       
        # --------------------------------------------------
        # ✅ RELATIVE CONTRIBUTION (FINAL FIXED VERSION)
        # --------------------------------------------------
        base = S + T + F

        if base > 1e-6:

            S_c = S * ef
            T_c = T * ef
            F_c = F * ef
            E_c = (ef - 1.0) * base

            self.logger.record("ethics/impact", E_c)

            total = abs(S_c) + abs(T_c) + abs(F_c) + abs(E_c)

            if total > 1e-6:

                S_i = S_c / total
                T_i = T_c / total
                F_i = F_c / total
                E_i = E_c / total

                # direkte Werte
                self.logger.record("importance/S", S_i)
                self.logger.record("importance/T", T_i)
                self.logger.record("importance/F", F_i)
                self.logger.record("importance/E", E_i)

                # HIST speichern
                self.importance_S_hist.append(S_i)
                self.importance_T_hist.append(T_i)
                self.importance_F_hist.append(F_i)
                self.importance_E_hist.append(E_i)

        # geglättete Werte
        if len(self.importance_S_hist) > 50:
            self.logger.record("importance_mean/S", np.mean(self.importance_S_hist))
            self.logger.record("importance_mean/T", np.mean(self.importance_T_hist))
            self.logger.record("importance_mean/F", np.mean(self.importance_F_hist))
            self.logger.record("importance_mean/E", np.mean(self.importance_E_hist))
            
        # --------------------------------------------------
        # ✅ TRAINING METRICS
        # --------------------------------------------------
        if "rewards" in self.locals:
            self.logger.record("train/reward", float(self.locals["rewards"][0]))

        if "values" in self.locals and "returns" in self.locals:
            values = self.locals["values"]
            returns = self.locals["returns"]

            if len(values) > 0 and len(returns) > 0:
                var_y = np.var(returns)
                ev = np.nan if var_y == 0 else 1 - np.var(returns - values) / var_y
                self.logger.record("train/explained_variance", ev)


        return True