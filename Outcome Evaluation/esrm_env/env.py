# env.py – ESRM Environment (interaction-gated reward, review-friendly)

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
    ):
        super().__init__()

        self.sumocfg_path = sumocfg_path
        self.junction_polygon = junction_polygon
        self.step_length = step_length
        self.horizon = int(episode_seconds / step_length)

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
        self.controller = ESRMController(junction_polygon)
        self.rewarder = ESRMRewardU1()

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
            high=10.0,
            shape=(11,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)

        self.t = 0
        self.traci_conn = None
        
        self.critical_vehicle_id = None
        self.critical_pedestrian_id = None
        
        self.active_conflicts = {}
        self.release_counter = 0
        
        self.ethics_tracker = ESRMEthicsTracker(
            output_events_file="ethics_events.json",
            output_summary_file="ethics_summary.json",
        )

    # --------------------------------------------------
    # SUMO start
    # --------------------------------------------------
    def _start_sumo(self):
        import traci
        import time
        import random
        import sumolib

        # ✅ alte Verbindung sauber schließen
        try:
            if traci.isLoaded():
                traci.close()  # ✅ OHNE False!
                time.sleep(0.5)
        except Exception as e:
            print("Warning while closing TraCI:", e)

        self.traci_conn = None

        sumo_binary = sumolib.checkBinary("sumo")
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumocfg_path,
            "--step-length", str(self.step_length),
            "--quit-on-end", "true",
            "--no-warnings",
        ]

        last_error = None

        for attempt in range(10):
            port = random.randint(50000, 60000)

            try:
                traci.start(sumo_cmd, port=port, numRetries=1)
                self.traci_conn = traci.getConnection()
                self.traci_conn.simulationStep()
                return

            except Exception as e:
                last_error = e
                print(f"Retry SUMO start (attempt {attempt})")

                # ✅ WICHTIG: nochmal sauber schließen
                try:
                    if traci.isLoaded():
                        traci.close()
                        time.sleep(0.5)
                except:
                    pass

                time.sleep(0.5)

        raise RuntimeError("Failed to start SUMO/TraCI") from last_error

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._start_sumo()
        self.state_extractor.reset()

        self.t = 0
        self.current_action = 0
        self.action_time_left = 0.0
        self.active_conflicts.clear()
        self.release_counter = 0
        self.ethics_tracker.reset()
        
        self.critical_vehicle_id = None
        self.critical_pedestrian_id = None

        _ = self.state_extractor.compute()

        return np.zeros(
            11,
            dtype=np.float32
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

        interaction_active = (
            vehicles_present
            and pedestrians_present
            and (
                (ttc < 6.0 and dist < 30.0)
                or dist < 8.0
            )
        )
        
        if interaction_active:
            self.release_counter = 0
        else:
            self.release_counter += 1
        
        release_condition = (
            self.release_counter >= 10
        )   
        
        if (
            interaction_active
            and self.critical_vehicle_id is not None
            and self.critical_pedestrian_id is not None
        ):
        
            self.active_conflicts[
                (
                    self.critical_vehicle_id,
                    self.critical_pedestrian_id
                )
            ] = {
                "action": self.current_action
            }
        
            try:
        
                simulation_time = (
                    self.traci_conn.simulation.getTime()
                )
        
            except Exception:
        
                simulation_time = (
                    self.t * self.step_length
                )
        
            self.ethics_tracker.update(
                simulation_time=simulation_time,
                step=self.t,
                vehicle_id=self.critical_vehicle_id,
                vru_id=self.critical_pedestrian_id,
                action=self.current_action,
                ttc=ttc,
                distance=dist,
                drac=drac,
            )
        
            print(
                "STORE CONFLICT:",
                self.critical_vehicle_id,
                self.critical_pedestrian_id
            )
        
        else:

            # --------------------------------------------------
            # Continue tracking during the release hysteresis
            # --------------------------------------------------
            if len(self.active_conflicts) > 0:
        
                try:
                    simulation_time = (
                        self.traci_conn.simulation.getTime()
                    )
                except Exception:
                    simulation_time = (
                        self.t * self.step_length
                    )
        
                for veh_id, ped_id in list(
                    self.active_conflicts.keys()
                ):
        
                    self.ethics_tracker.update(
                        simulation_time=simulation_time,
                        step=self.t,
                        vehicle_id=veh_id,
                        vru_id=ped_id,
                        action=self.current_action,
                        ttc=None,
                        distance=None,
                        drac=None,
                    )
        
            # --------------------------------------------------
            # Final release after hysteresis
            # --------------------------------------------------
            if (
                release_condition
                and len(self.active_conflicts) > 0
            ):
        
                print(
                    "RELEASE CHECK:",
                    len(self.active_conflicts)
                )
        
                released_vehicles = set()
        
                for veh_id, ped_id in list(
                    self.active_conflicts.keys()
                ):
        
                    if veh_id not in released_vehicles:
        
                        try:
        
                            if veh_id in traci.vehicle.getIDList():
        
                                print(
                                    f"RELEASE | VEH={veh_id}"
                                )
        
                                traci.vehicle.setSpeed(
                                    veh_id,
                                    -1
                                )
        
                            released_vehicles.add(
                                veh_id
                            )
        
                        except Exception:
                            pass
        
                    self.ethics_tracker.finalize_pair(
                        vehicle_id=veh_id,
                        vru_id=ped_id,
                    )
        
                    self.active_conflicts.pop(
                        (veh_id, ped_id),
                        None
                    )               


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
        

        obs = np.array([

            # --------------------------------------------------
            # Stage 1: Interaction
            # --------------------------------------------------
            float(interaction_active),

            # --------------------------------------------------
            # Stage 2: Safety
            # --------------------------------------------------
            obs_dict["ttc_min"],
            obs_dict["dist_min"],
            obs_dict["drac_min"],

            # --------------------------------------------------
            # Stage 2: Demand / Throughput
            # --------------------------------------------------
            obs_dict["n_cars"],
            obs_dict["n_buses"],
            obs_dict["n_peds"],

            # --------------------------------------------------
            # Stage 2: Fairness
            # --------------------------------------------------
            obs_dict["mean_wait_car"],
            obs_dict["mean_wait_bus"],
            obs_dict["mean_wait_vru"],

            # --------------------------------------------------
            # Stage 2: Conflict
            # --------------------------------------------------
            obs_dict["num_conflicts"],

        ], dtype=np.float32)

        return obs, reward, done, info

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    def close(self):

        try:
    
            if (
                len(self.ethics_tracker.active_events) > 0
                or len(self.ethics_tracker.completed_events) > 0
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