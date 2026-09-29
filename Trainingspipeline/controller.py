# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:06:10 2026

@author: 49151
"""
# controller.py

import traci
from esrm_env.observation import point_in_polygon


class ESRMController:
    """
    Executes discrete, soft traffic interventions.
    Actions are applied only to traffic participants
    within the junction-relevant area.
    Temporal constraints are handled by the environment.
    """

    def __init__(self, junction_polygon):
        self.poly = junction_polygon

    def apply(
        self,
        action: int,
        critical_vehicle_id=None,
        critical_pedestrian_id=None
    ):
    
        # A0: No-op
        if action == 0:
            return
    
        # A1: VRU priority (mild)
        elif action == 1:
    
            if critical_vehicle_id is not None:
    
                try:
    
                    if critical_vehicle_id in traci.vehicle.getIDList():
    
                        print(
                            f"INTERVENE | "
                            f"ACTION={action} | "
                            f"VEH={critical_vehicle_id}"
                        )
    
                        traci.vehicle.setSpeed(
                            critical_vehicle_id,
                            5.0
                        )
    
                except traci.TraCIException:
                    pass
    
        # A2: Vehicle priority
        elif action == 2:
    
            if critical_pedestrian_id is not None:
    
                try:
    
                    if critical_pedestrian_id in traci.person.getIDList():
    
                        print(
                            f"INTERVENE | "
                            f"ACTION={action} | "
                            f"PED={critical_pedestrian_id}"
                        )
    
                        traci.person.setSpeed(
                            critical_pedestrian_id,
                            0.8
                        )
    
                except traci.TraCIException:
                    pass
    
        # A3: Cooperative moderation
        elif action == 3:
    
            if critical_vehicle_id is not None:
    
                try:
    
                    if critical_vehicle_id in traci.vehicle.getIDList():
    
                        print(
                            f"INTERVENE | "
                            f"ACTION={action} | "
                            f"VEH={critical_vehicle_id}"
                        )
    
                        traci.vehicle.setSpeed(
                            critical_vehicle_id,
                            2.5
                        )
    
                except traci.TraCIException:
                    pass
    
            if critical_pedestrian_id is not None:
    
                try:
    
                    if critical_pedestrian_id in traci.person.getIDList():
    
                        print(
                            f"INTERVENE | "
                            f"ACTION={action} | "
                            f"PED={critical_pedestrian_id}"
                        )
    
                        traci.person.setSpeed(
                            critical_pedestrian_id,
                            0.8
                        )
    
                except traci.TraCIException:
                    pass
    
        # A4: Emergency intervention
        elif action == 4:
    
            if critical_vehicle_id is not None:
    
                try:
    
                    if critical_vehicle_id in traci.vehicle.getIDList():
    
                        print(
                            f"INTERVENE | "
                            f"ACTION={action} | "
                            f"VEH={critical_vehicle_id}"
                        )
    
                        traci.vehicle.setSpeed(
                            critical_vehicle_id,
                            0.0
                        )
    
                except traci.TraCIException:
                    pass