# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:09:27 2026

@author: 49151
"""

# env_u2b.py – Interaction = ACTIVE; vorher: GLOBAL ETHICS (U2b)

import time
import subprocess
import random
import numpy as np
import traci
import sumolib
from gym import Env, spaces

from esrm_env.observation import ESRMStateExtractorV2
from esrm_env.controller import ESRMController
from esrm_env.reward import ESRMRewardU1


class ESRMEnv(Env):

    def __init__(
        self,
                sumocfg_path: str,
        junction_polygon,
        episode_seconds: float = 600.0,
        step_length: float = 0.1,
        vru_inner_edges=None,
    ):
        super().__init__()

        self._seed = None
        
        self.sumocfg_path = sumocfg_path
        self.junction_polygon = junction_polygon
        self.step_length = step_length
        self.horizon = int(episode_seconds / step_length)

        self.state_extractor = ESRMStateExtractorV2(
            polygon=junction_polygon,
            step_length=step_length,
        )

        if vru_inner_edges is not None:
            self.state_extractor.vru_inner_edges = set(vru_inner_edges)
        else:
            self.state_extractor.vru_inner_edges = set()

        self.controller = ESRMController(junction_polygon)
        self.rewarder = None

        self.current_action = 0
        self.action_time_left = 0.0
        self.action_durations = {
            0: 0.0,
            1: 4.0,
            2: 4.0,
            3: 3.0,
            4: 1.5,
        }

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)

        self.t = 0
        self.traci_conn = None

    def _start_sumo(self):

        # ✅ 1. alte TraCI Verbindung schließen
        try:
            traci.switch("default")
            traci.close()
        except Exception:
            pass

        # ✅ 2. interne TraCI registry resetten
        import traci.connection as tc
        try:
            tc._connections.clear()
        except Exception:
            pass

        # ✅ 3. alte SUMO Prozesse killen
        try:
            subprocess.call(
                "taskkill /f /im sumo.exe",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
             )
        except Exception:
            pass

        time.sleep(0.5)

        # ✅ 4. SUMO starten
        sumo_binary = sumolib.checkBinary("sumo")

        sumo_cmd = [
            sumo_binary,
            "-c", self.sumocfg_path,
            "--step-length", str(self.step_length),
            "--seed", str(self._seed if self._seed is not None else 0),
            "--quit-on-end", "true",
            "--no-warnings",
        ]

        last_error = None

        for _ in range(20):
            base_port = 50000
            offset = self._seed if isinstance(self._seed, int) else 0
            port = base_port + offset + int(time.time() * 1000) % 1000

            try:
                traci.start(sumo_cmd, port=port, numRetries=1)
                self.traci_conn = traci.getConnection()
                self.traci_conn.simulationStep()
                return
            except Exception as e:
                last_error = e
                try:
                    traci.close()
                except:
                    pass
                time.sleep(0.5)

        raise RuntimeError("Failed to start SUMO/TraCI") from last_error
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # ✅ Seed speichern
        self._seed = seed

        # ✅ Alle Random Sources setzen
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self._start_sumo()

        # ✅ erst nachdem SUMO läuft
        self.state_extractor.reset()

        self.t = 0
        self.current_action = 0
        self.action_time_left = 0.0

        _ = self.state_extractor.compute()
        return np.array([0.0], dtype=np.float32)

    def step(self, action: int):

        if self.action_time_left <= 0.0:
            self.current_action = int(action)
            self.action_time_left = self.action_durations[self.current_action]

        self.controller.apply(self.current_action)
        self.action_time_left -= self.step_length

        self.traci_conn.simulationStep()
        self.t += 1

        obs_dict = self.state_extractor.compute()

        # --------------------------------------------------
        # Interaction (für Gating + Logging)
        # --------------------------------------------------
        ttc = obs_dict.get("ttc_min", None)
        dist = obs_dict.get("dist_min", None)
        drac = obs_dict.get("drac_min", None)

        # ✅ NEW: Presence Checks
        n_cars = obs_dict.get("n_cars", 0)
        n_buses = obs_dict.get("n_buses", 0)

        vehicles_present = (n_cars + n_buses) > 0
        pedestrians_present = obs_dict.get("n_peds", 0) > 0

        interaction_active = (
            vehicles_present and pedestrians_present and (
                (ttc is not None and ttc < 6.0) or
                (dist is not None and dist < 8.0) or
                (drac is not None and drac > 1.0)
            )
        )
        # --------------------------------------------------
        assert self.rewarder is not None, "Rewarder not set!"
        # --------------------------------------------------
              
        reward, reward_components = self.rewarder(obs_dict)

        # --------------------------------------------------
        # ✅ INTERACTION GATING (NEU!)
        # --------------------------------------------------
        if interaction_active:
            reward_final = reward
        else:
            reward_final = 0.0

        done = self.t >= self.horizon


        info = {
            "metrics": {
                **obs_dict,
                **reward_components,
                "interaction_active": int(interaction_active),
            }
        }

        obs = np.array([float(interaction_active)], dtype=np.float32)

        return obs, reward_final, done, info


    def close(self):
        try:
            traci.close()
        except Exception:
            pass

        # ✅ kleine Pause für Socket-Freigabe
        time.sleep(0.2)


