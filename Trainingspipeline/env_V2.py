# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 17:27:03 2026

@author: 49151
"""

# env_V2.py – ESRM Environment (interaction-gated reward, review-friendly)

import time
import random
import numpy as np
import traci
import sumolib
from gym import Env, spaces

from esrm_env.observation import ESRMStateExtractorV2
from esrm_env.controller import ESRMController
from esrm_env.reward import ESRMRewardU1
from esrm_env.ethics_tracker import ESRMEthicsTracker


class ESRMEnv(Env):
    
    
    def __init__(
        self,
        sumocfg_path: str,
        junction_polygon,
        episode_seconds: float = 600.0,
        step_length: float = 0.1,
        vru_inner_edges=None,
        seed: int = 1,
        rewarder=None,
    ):
        super().__init__()
    
        self.sumocfg_path = sumocfg_path
        self.junction_polygon = junction_polygon
        self.step_length = step_length
        self.horizon = int(
            episode_seconds / step_length
        )
    
        # --------------------------------------------------
        # Reproducibility
        # --------------------------------------------------
    
        self.base_seed = int(seed)
        self.current_seed = int(seed)
        self.episode_index = 0

        # --------------------------------------------------
        # State extraction
        # --------------------------------------------------
        self.state_extractor = ESRMStateExtractorV2(
            polygon=junction_polygon,
            step_length=step_length,
        )

        if vru_inner_edges is not None:
            self.state_extractor.vru_inner_edges = set(vru_inner_edges)
        else:
            self.state_extractor.vru_inner_edges = set()

        
        # --------------------------------------------------
        # Control & reward modules
        # --------------------------------------------------
        
        self.controller = ESRMController(
            junction_polygon
        )
        
        if rewarder is None:
        
            self.rewarder = ESRMRewardU1()
        
        else:
        
            self.rewarder = rewarder

        # --------------------------------------------------
        # Action timing
        # --------------------------------------------------
        self.current_action = 0
        self.action_time_left = 0.0
        self.action_durations = {
            0: 0.0,
            1: 4.0,
            2: 4.0,
            3: 3.0,
            4: 1.5,
        }

        # --------------------------------------------------
        # Gym spaces
        # --------------------------------------------------
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(11,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)

        self.t = 0
        self.traci_conn = None
        
        self.critical_vehicle_id = None
        self.critical_pedestrian_id = None
        
        # --------------------------------------------------
        # Pair-based interaction tracking
        # --------------------------------------------------
        
        self.release_steps = 10
        
        self.active_conflicts = {}
        
        # Ensures that each vehicle-VRU pair is counted
        # only once per episode.
        self.completed_pairs = set()
        
        self.ethics_tracker = ESRMEthicsTracker(
            output_events_file="ethics_events.json",
            output_summary_file="ethics_summary.json",
        )
        

    # --------------------------------------------------
    # SUMO start
    # --------------------------------------------------
    def _start_sumo(self):

        # ✅ alte Verbindung sauber schließen
        try:
            if traci.isLoaded():
                traci.close()  # ✅ OHNE False!
                time.sleep(2.0)
        except Exception as e:
            print("Warning while closing TraCI:", e)

        self.traci_conn = None

        sumo_binary = sumolib.checkBinary(
            "sumo"
        )
        
        sumo_cmd = [
            sumo_binary,
            "-c",
            self.sumocfg_path,
            "--step-length",
            str(self.step_length),
            "--seed",
            str(self.current_seed),
            "--quit-on-end",
            "true",
            "--no-warnings",
        ]
        
        last_error = None
        port_rng = random.SystemRandom()
        
        for attempt in range(10):
        
            port = port_rng.randint(
                50000,
                60000
            )
        
            try:
                
                print(
                    f"STARTING SUMO | "
                    f"episode={self.episode_index} | "
                    f"seed={self.current_seed} | "
                    f"attempt={attempt+1}"
                )
        
                traci.start(
                    sumo_cmd,
                    port=port,
                    numRetries=1
                )
                
                print(
                    "SUMO STARTED"
                )
        
                self.traci_conn = (
                    traci.getConnection()
                )
        
                self.traci_conn.simulationStep()
        
                return
        
            except Exception as e:
        
                last_error = e
        
                print(
                    "Retry SUMO start "
                    f"(attempt {attempt + 1})"
                )
        
                try:
        
                    if traci.isLoaded():
                        traci.close()
                        time.sleep(2.0)
        
                except Exception:
                    pass
        
                time.sleep(0.5)

        raise RuntimeError("Failed to start SUMO/TraCI") from last_error

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------
    def reset(
        self,
        seed=None,
        options=None
    ):
    
        # --------------------------------------------------
        # Resolve episode seed
        # --------------------------------------------------
    
        if seed is not None:
    
            self.base_seed = int(seed)
            self.episode_index = 0
    
        self.current_seed = (
            self.base_seed * 100_000 + self.episode_index
        )
    
        self.episode_index += 1
    
        super().reset(
            seed=self.current_seed
        )
    
        # Seed Gym spaces explicitly.
        self.action_space.seed(
            self.current_seed
        )
    
        self.observation_space.seed(
            self.current_seed
        )
    
        print(
            "RESET ENV | "
            f"episode={self.episode_index} | "
            f"sumo_seed={self.current_seed}"
        )
    
        # --------------------------------------------------
        # Start SUMO
        # --------------------------------------------------
               
        self._start_sumo()
    
        self.state_extractor.reset()
    
        # --------------------------------------------------
        # Reset episode state
        # --------------------------------------------------
    
        self.t = 0
    
        self.current_action = 0
        self.action_time_left = 0.0
    
        self.active_conflicts.clear()
        self.completed_pairs.clear()
    
        self.ethics_tracker.reset()
    
        self.critical_vehicle_id = None
        self.critical_pedestrian_id = None
    
        # --------------------------------------------------
        # Initial observation
        # --------------------------------------------------
    
        obs_dict = (
            self.state_extractor.compute()
        )
        
        self.critical_vehicle_id = (
            obs_dict.get(
                "critical_vehicle_id"
            )
        )
        
        self.critical_pedestrian_id = (
            obs_dict.get(
                "critical_pedestrian_id"
            )
        )
        
        interaction_active = (
            self._is_interaction_active(
                obs_dict
            )
        )
        
        initial_observation = (
            self._build_observation(
                obs_dict=obs_dict,
                interaction_active=interaction_active
            )
        )
        
        return initial_observation
    
    
    def _build_observation(
        self,
        obs_dict,
        interaction_active
    ):
    
        def safe_float(
            value,
            default
        ):
    
            if value is None:
                return float(default)
    
            try:
    
                numeric_value = float(value)
    
            except (
                TypeError,
                ValueError
            ):
    
                return float(default)
    
            if not np.isfinite(
                numeric_value
            ):
    
                return float(default)
    
            return numeric_value
    
        return np.array(
            [
                # Interaction
                float(interaction_active),
    
                # Safety
                safe_float(
                    obs_dict.get("ttc_min"),
                    10.0
                ),
                safe_float(
                    obs_dict.get("dist_min"),
                    10.0
                ),
                safe_float(
                    obs_dict.get("drac_min"),
                    0.0
                ),
    
                # Demand / throughput
                safe_float(
                    obs_dict.get("n_cars"),
                    0.0
                ),
                safe_float(
                    obs_dict.get("n_buses"),
                    0.0
                ),
                safe_float(
                    obs_dict.get("n_peds"),
                    0.0
                ),
    
                # Fairness
                safe_float(
                    obs_dict.get("mean_wait_car"),
                    0.0
                ),
                safe_float(
                    obs_dict.get("mean_wait_bus"),
                    0.0
                ),
                safe_float(
                    obs_dict.get("mean_wait_vru"),
                    0.0
                ),
    
                # Conflicts
                safe_float(
                    obs_dict.get("num_conflicts"),
                    0.0
                ),
            ],
            dtype=np.float32
        )
    
    def _is_interaction_active(
            self,
            obs_dict
        ):
        
            n_cars = obs_dict.get(
                "n_cars",
                0
            )
        
            n_buses = obs_dict.get(
                "n_buses",
                0
            )
        
            n_peds = obs_dict.get(
                "n_peds",
                0
            )
        
            vehicles_present = (
                n_cars + n_buses
            ) > 0
        
            pedestrians_present = (
                n_peds > 0
            )
        
            ttc = obs_dict.get(
                "ttc_min"
            )
        
            dist = obs_dict.get(
                "dist_min"
            )
        
            try:
        
                ttc_value = float(ttc)
                ttc_valid = np.isfinite(
                    ttc_value
                )
        
            except (
                TypeError,
                ValueError
            ):
        
                ttc_value = None
                ttc_valid = False
        
            try:
        
                dist_value = float(dist)
                dist_valid = np.isfinite(
                    dist_value
                )
        
            except (
                TypeError,
                ValueError
            ):
        
                dist_value = None
                dist_valid = False
        
            return bool(
                vehicles_present
                and pedestrians_present
                and dist_valid
                and (
                    (
                        ttc_valid
                        and ttc_value < 6.0
                        and dist_value < 30.0
                    )
                    or dist_value < 8.0
                )
            )
    
    # --------------------------------------------------
    # Step
    # --------------------------------------------------
    def step(self, action: int):

        # -------------------------------
        # Action scheduling
        # -------------------------------
        if self.action_time_left <= 0.0:
            self.current_action = int(action)
            self.action_time_left = self.action_durations[self.current_action]

        self.controller.apply(
            self.current_action,
            self.critical_vehicle_id,
            self.critical_pedestrian_id
        )
        self.action_time_left -= self.step_length

        # -------------------------------
        # Advance SUMO
        # -------------------------------
        self.traci_conn.simulationStep()
        self.t += 1

        # -------------------------------
        # State extraction
        # -------------------------------
        obs_dict = self.state_extractor.compute()
        
        self.critical_vehicle_id = obs_dict.get(
            "critical_vehicle_id"
        )
        
        self.critical_pedestrian_id = obs_dict.get(
            "critical_pedestrian_id"
        )
        
        n_cars = obs_dict.get("n_cars", 0)
        n_buses = obs_dict.get("n_buses", 0)
        n_peds = obs_dict.get("n_peds", 0)

        vehicles_present = (n_cars + n_buses) > 0
        pedestrians_present = n_peds > 0

        # --------------------------------------------------
        # ✅ Interaction definition (SSM-based)
        # --------------------------------------------------
        ttc = obs_dict.get("ttc_min", None)
        dist = obs_dict.get("dist_min", None)
        drac = obs_dict.get("drac_min", None)

        ttc = obs_dict.get(
            "ttc_min"
        )
        
        dist = obs_dict.get(
            "dist_min"
        )
        
        drac = obs_dict.get(
            "drac_min"
        )
        
        interaction_active = (
            self._is_interaction_active(
                obs_dict
            )
        )

        # --------------------------------------------------
        # Pair-based ethics event tracking
        # --------------------------------------------------
        
        try:
        
            simulation_time = (
                self.traci_conn.simulation.getTime()
            )
        
        except Exception:
        
            simulation_time = (
                self.t * self.step_length
            )
        
        
        current_pair = None
        
        if (
            interaction_active
            and self.critical_vehicle_id is not None
            and self.critical_pedestrian_id is not None
        ):
        
            current_pair = (
                self.critical_vehicle_id,
                self.critical_pedestrian_id
            )
        
        
        # --------------------------------------------------
        # Update the currently detected pair
        # --------------------------------------------------
        
        if (
            current_pair is not None
            and current_pair not in self.completed_pairs
        ):
        
            veh_id, ped_id = current_pair
        
            if current_pair not in self.active_conflicts:
        
                self.active_conflicts[current_pair] = {
                    "missing_steps": 0,
                    "last_seen_step": self.t,
                    "last_action": self.current_action,
                }
        
            else:
        
                self.active_conflicts[
                    current_pair
                ]["missing_steps"] = 0
        
                self.active_conflicts[
                    current_pair
                ]["last_seen_step"] = self.t
        
                self.active_conflicts[
                    current_pair
                ]["last_action"] = (
                    self.current_action
                )
        
            self.ethics_tracker.update(
                simulation_time=simulation_time,
                step=self.t,
                vehicle_id=veh_id,
                vru_id=ped_id,
                action=self.current_action,
                ttc=ttc,
                distance=dist,
                drac=drac,
            )
        
            print(
                "STORE INTERACTION:",
                veh_id,
                ped_id
            )
        
        
        # --------------------------------------------------
        # Advance hysteresis for all other active pairs
        # --------------------------------------------------
        
        pairs_to_finalize = []
        
        for pair, pair_state in list(
            self.active_conflicts.items()
        ):
        
            # The current pair was already updated above.
            if pair == current_pair:
                continue
        
            veh_id, ped_id = pair
        
            pair_state["missing_steps"] += 1
        
            # Continue recording the pair during hysteresis.
            self.ethics_tracker.update(
                simulation_time=simulation_time,
                step=self.t,
                vehicle_id=veh_id,
                vru_id=ped_id,
                action=pair_state["last_action"],
                ttc=None,
                distance=None,
                drac=None,
            )
        
            if (
                pair_state["missing_steps"]
                >= self.release_steps
            ):
        
                pairs_to_finalize.append(
                    pair
                )
        
        
        # --------------------------------------------------
        # Finalize expired pairs individually
        # --------------------------------------------------
        
        for pair in pairs_to_finalize:

            veh_id, ped_id = pair
        
            print(
                "FINALIZE INTERACTION | "
                f"VEH={veh_id} | "
                f"VRU={ped_id}"
            )
        
            self.ethics_tracker.finalize_pair(
                vehicle_id=veh_id,
                vru_id=ped_id,
            )
        
            self.completed_pairs.add(
                pair
            )
        
            self.active_conflicts.pop(
                pair,
                None
            )
        
            vehicle_still_active = any(
                active_veh_id == veh_id
                for active_veh_id, _ in (
                    self.active_conflicts.keys()
                )
            )
        
            if not vehicle_still_active:
        
                try:
        
                    if veh_id in (
                        self.traci_conn.vehicle.getIDList()
                    ):
        
                        self.traci_conn.vehicle.setSpeed(
                            veh_id,
                            -1
                        )
        
                        print(
                            f"RELEASE VEHICLE | "
                            f"VEH={veh_id}"
                        )
        
                except Exception:
                    pass              


        # --------------------------------------------------
        # ✅ ALWAYS compute reward (→ ensures E is always computed)
        # --------------------------------------------------
        reward_raw, reward_components = self.rewarder(
            obs_dict,
            self.current_action
        )

        if interaction_active:
            reward = reward_raw
        else:
            reward = 0.0

        done = self.t >= self.horizon

        if done:

            # --------------------------------------------------
            # Finalize all still-active pairs
            # --------------------------------------------------
        
            for veh_id, ped_id in list(
                self.active_conflicts.keys()
            ):
        
                self.ethics_tracker.finalize_pair(
                    vehicle_id=veh_id,
                    vru_id=ped_id,
                )
        
                self.completed_pairs.add(
                    (veh_id, ped_id)
                )
        
                self.active_conflicts.pop(
                    (veh_id, ped_id),
                    None
                )
        
            ethics_summary = (
                self.ethics_tracker.export()
            )
        
        else:
                
            ethics_summary = None

        # -------------------------------
        # Logging
        # -------------------------------
        info = {
            "metrics": {
                **obs_dict,
                **reward_components,
                "interaction_active": int(interaction_active),
            },
            "ethics_summary": ethics_summary,
        }
        

        obs = self._build_observation(
            obs_dict=obs_dict,
            interaction_active=interaction_active
        )

        return obs, reward, done, info

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    def close(self):

        try:

            for veh_id, ped_id in list(
                self.active_conflicts.keys()
            ):
        
                self.ethics_tracker.finalize_pair(
                    vehicle_id=veh_id,
                    vru_id=ped_id,
                )
        
                self.completed_pairs.add(
                    (veh_id, ped_id)
                )
        
                self.active_conflicts.pop(
                    (veh_id, ped_id),
                    None
                )
        
            if (
                len(self.ethics_tracker.active_events) > 0
                or len(
                    self.ethics_tracker.completed_events
                ) > 0
            ):
        
                self.ethics_tracker.export()
        
        except Exception as e:
        
            print(
                "Warning while exporting ethics data:",
                e
            )
    
        try:
    
            if traci.isLoaded():
                traci.close()
    
        except Exception:
            pass