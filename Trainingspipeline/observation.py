# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:05:20 2026

@author: 49151
"""

# observation.py – ESRM State Extraction (presence-robust, 0.1s compatible)
# -*- coding: utf-8 -*-
"""
observation.py – ESRM State Extraction (FINAL, conflict-aware)

Conflict definition:
A conflict occurs if
1) at least one vehicle AND one VRU are present in the junction area, and
2) at least one vehicle exhibits yielding behaviour, operationalized via:
   - low absolute speed, OR
   - negative acceleration (braking), OR
   - positive acceleration (reactive acceleration)

All logic is evaluated per simulation step (online).
"""

import traci
import numpy as np
import math


def point_in_polygon(point, polygon):
    """Ray-casting point-in-polygon test (vehicles)."""
    x, y = point
    inside = False
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if ((y1 > y) != (y2 > y)) and \
           (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
            inside = not inside
    return inside


class ESRMStateExtractorV2:
    """
    Step-wise traffic state extraction for ESRM.

    Vehicles:
      - inside polygon OR internal junction lanes (laneID starts with ':')

    VRUs:
      - on junction-related edges OR internal crossing edges

    Conflict:
      - simultaneous vehicle–VRU presence AND
      - yielding behaviour from at least one vehicle
    """

    def __init__(self, polygon, step_length):
        self.poly = polygon
        self.step_length = step_length

        # Waiting time tracking
        self.wait_car = {}
        self.wait_bus = {}
        self.wait_vru = {}

        # Previous inside-state (exit detection)
        self.prev_inside = {}

        # Previous speed per vehicle (for acceleration)
        self.prev_speed = {}

        # Must be injected by env.py
        self.vru_inner_edges = set()

        # Yielding thresholds (option A + B)
        self.YIELD_SPEED = 1.5        # m/s  (slow crawling)
        self.BRAKE_ACC = -1.0         # m/s² (mild braking)
        self.ACCEL_ACC = 1.0          # m/s² (reactive acceleration)

    def reset(self):
        self.wait_car.clear()
        self.wait_bus.clear()
        self.wait_vru.clear()
        self.prev_inside.clear()
        self.prev_speed.clear()

    def compute(self):
        vehicle_ids = traci.vehicle.getIDList()
        person_ids = traci.person.getIDList()

        # Presence counters
        n_cars = n_buses = n_peds = 0

        # Throughput counters
        car_exit = bus_exit = vru_exit = 0

        veh_states = []   # (vid, x, y, v, a)
        vru_states = []   # (pid, x, y, v)

        # ==================================================
        # VEHICLES
        # ==================================================
        for vid in vehicle_ids:
            pos = traci.vehicle.getPosition(vid)
            speed = traci.vehicle.getSpeed(vid)
            vclass = traci.vehicle.getVehicleClass(vid)
            lane_id = traci.vehicle.getLaneID(vid)

            inside = (
                point_in_polygon(pos, self.poly)
                or lane_id.startswith(":")
            )

            # Acceleration (finite difference)
            prev_v = self.prev_speed.get(vid, speed)
            acc = (speed - prev_v) / max(self.step_length, 1e-6)
            self.prev_speed[vid] = speed

            if inside:
                if vclass == "bus":
                    n_buses += 1
                elif vclass == "passenger":
                    n_cars += 1
                veh_states.append(
                    (
                        vid,
                        pos[0],
                        pos[1],
                        speed,
                        acc
                    )
                )

            # Waiting time
            if speed < 1.0:
                if vclass == "bus":
                    self.wait_bus[vid] = self.wait_bus.get(vid, 0.0) + self.step_length
                elif vclass == "passenger":
                    self.wait_car[vid] = self.wait_car.get(vid, 0.0) + self.step_length

            # Exit detection
            prev = self.prev_inside.get(vid, False)
            if prev and not inside:
                if vclass == "bus":
                    bus_exit += 1
                elif vclass == "passenger":
                    car_exit += 1

            self.prev_inside[vid] = inside

        # ==================================================
        # PEDESTRIANS (VRUs)
        # ==================================================
        for pid in person_ids:
            pos = traci.person.getPosition(pid)
            speed = traci.person.getSpeed(pid)
            road_id = traci.person.getRoadID(pid)

            inside = (
                road_id in self.vru_inner_edges
                or road_id.startswith(":")
            )

            if inside:
                n_peds += 1
                vru_states.append(
                    (
                        pid,
                        pos[0],
                        pos[1],
                        speed
                    )
                )

            if speed < 0.3:
                self.wait_vru[pid] = self.wait_vru.get(pid, 0.0) + self.step_length

            prev = self.prev_inside.get(pid, False)
            if prev and not inside:
                vru_exit += 1

            self.prev_inside[pid] = inside

        # ==================================================
        # SAFETY SURROGATES
        # ==================================================
        critical_vehicle_id = None
        critical_pedestrian_id = None
        
        max_risk_score = -1.0
        
        ttc_min = 10.0
        dist_min = 50.0
        drac_max = 0.0

        if veh_states and vru_states:

            for vid, xv, yv, vv, _ in veh_states:
        
                for pid, xp, yp, vp in vru_states:
        
                    d = math.hypot(
                        xv - xp,
                        yv - yp
                    )
        
                    dv = abs(vv - vp)
        
                    ttc = d / max(dv, 1e-6)

                    drac = (vv * vv) / max(2 * d, 1e-6)
                    
                    is_interaction_pair = (
                        (ttc < 6.0 and d < 30.0)
                        or d < 8.0
                    )
                    
                    if is_interaction_pair:

                        ttc_min = min(ttc_min, ttc)
                    
                        dist_min = min(dist_min, d)
                    
                        drac_max = max(drac_max, drac)
                    
                        risk_score = (
                            max(0.0, 6.0 - ttc) / 6.0
                            +
                            max(0.0, 8.0 - d) / 8.0
                            +
                            max(0.0, drac - 1.0) / 4.0
                        )
                    
                        if risk_score > max_risk_score:
    
                            max_risk_score = risk_score
                        
                            critical_vehicle_id = vid
                            critical_pedestrian_id = pid
                        
                            if risk_score > 1.0:
                        
                                print(
                                    "CRITICAL PAIR | "
                                    f"veh={critical_vehicle_id} | "
                                    f"ped={critical_pedestrian_id} | "
                                    f"TTC={ttc:.2f} | "
                                    f"DIST={d:.2f} | "
                                    f"DRAC={drac:.2f} | "
                                    f"RISK={risk_score:.2f}"
                                )
                                
        # ==================================================
        # CONFLICT DETECTION (Option A + B)
        # ==================================================
        conflict = False
        if veh_states and vru_states:
            for _, _, _, v, a in veh_states:
                if (
                    v < self.YIELD_SPEED or         # crawling / waiting
                    a < self.BRAKE_ACC or           # braking
                    a > self.ACCEL_ACC              # reactive acceleration
                ):
                    conflict = True
                    break

        num_conflicts = int(conflict)

        # ==================================================
        # RETURN OBSERVATION
        # ==================================================
        return {
            # Presence
            "n_cars": n_cars,
            "n_buses": n_buses,
            "n_peds": n_peds,

            # Throughput
            "car_exit": car_exit,
            "bus_exit": bus_exit,
            "vru_exit": vru_exit,

            # Waiting times
            "mean_wait_car": np.mean(list(self.wait_car.values())) if self.wait_car else 0.0,
            "mean_wait_bus": np.mean(list(self.wait_bus.values())) if self.wait_bus else 0.0,
            "mean_wait_vru": np.mean(list(self.wait_vru.values())) if self.wait_vru else 0.0,

            # Safety metrics
            "ttc_min": ttc_min,
            "dist_min": dist_min,
            "drac_min": drac_max,
            
            # Conflict metric (as defined in thesis)
            "num_conflicts": num_conflicts,
            
            "critical_vehicle_id": critical_vehicle_id,
            "critical_pedestrian_id": critical_pedestrian_id,

            # Collisions
            "num_collisions": traci.simulation.getCollidingVehiclesNumber(),
            "collision_occurred": traci.simulation.getCollidingVehiclesNumber() > 0,
        }