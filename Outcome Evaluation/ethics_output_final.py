# -*- coding: utf-8 -*-
"""
ESRM ETHICS MODULE (FINAL OUTCOME VERSION)
------------------------------------------

Outcome Ethics Metrics:

1. Ethical Alignment
   EA = (N_proactive + N_reactive) / N_conflicts

2. VRU Priority Score
   P_VRU = N_proactive / N_conflicts

3. Conflict Classification
   - Proactive Yielding
   - Reactive Yielding
   - Non-Yielding

All conflicts:
- localized to the ESRM intersection polygon
- restricted to car–VRU interactions
- reconstructed from SSM + FCD outputs
"""

import xml.etree.ElementTree as ET
from shapely.geometry import Point, Polygon
import config


# ============================================================================
# LOAD VRU SPEEDS
# ============================================================================

def load_fcd_participants():

    vru_speeds = {}
    vehicle_ids = set()
    sample_vehicles = []

    tree = ET.parse(config.FCD_PEDESTRIANS_FILE)
    root = tree.getroot()

    for ts in root.findall("timestep"):

        t = float(ts.get("time"))

        # --------------------------------------------------
        # Persons
        # --------------------------------------------------
        for person in ts.findall("person"):

            person_id = person.get("id")

            if person_id is None:
                continue

            speed = float(
                person.get("speed", 0.0)
            )

            vru_speeds.setdefault(
                person_id,
                {}
            )[t] = speed

        # --------------------------------------------------
        # Vehicles
        # --------------------------------------------------
        for vehicle in ts.findall("vehicle"):

            vehicle_id = vehicle.get("id")

            if vehicle_id is None:
                continue

            vehicle_ids.add(vehicle_id)

            if len(sample_vehicles) < 10:

                sample_vehicles.append({
                    "id": vehicle_id,
                    "type": vehicle.get("type"),
                    "x": vehicle.get("x"),
                    "y": vehicle.get("y"),
                    "speed": vehicle.get("speed"),
                    "edge": vehicle.get("edge"),
                })

    return (
        vru_speeds,
        vehicle_ids,
        sample_vehicles
    )

# ============================================================================
# VRU SLOWDOWN
# ============================================================================

def vru_slowdown(
    vru_id,
    time_values,
    speeds,
    thr=0.4
):

    if vru_id not in speeds:
        return None

    person_speeds = speeds[vru_id]

    observed = False

    for t in time_values:

        if t not in person_speeds:
            continue

        observed = True

        if person_speeds[t] < thr:
            return True

    if not observed:
        return None

    return False


# ============================================================================
# ETHICS COMPUTATION
# ============================================================================

def compute_ethics():

    poly = Polygon(config.JUNCTION_POLYGON)

    (
        vru_speeds,
        fcd_vehicle_ids,
        sample_fcd_vehicles
    ) = load_fcd_participants()
    
    vru_ids = set(
        vru_speeds.keys()
    )

    tree = ET.parse(config.SSM_FILE)
    root = tree.getroot()

    N_conf = 0
    N_rea = 0
    N_pro = 0
    
    # Temporary diagnostics for participant identification
    N_local_ssm = 0
    N_car_vru = 0
    N_no_vru = 0
    N_both_vru = 0
    
    N_all_ssm = 0
    N_all_ssm_with_vru = 0
    
    sample_exact_vru_matches = []
    sample_ssm_participants = []
    
    car_vru_diagnostics = []

    def sf(v, default=999):

        try:
            return float(v)
        except:
            return default

    for c in root.findall("conflict"):
        
        N_all_ssm += 1

        raw_ego_id = c.get("ego")
        raw_foe_id = c.get("foe")
        
        if raw_ego_id is not None and raw_foe_id is not None:
        
            if len(sample_ssm_participants) < 20:
                sample_ssm_participants.append(
                    (raw_ego_id, raw_foe_id)
                )
        
            raw_ego_is_vru = raw_ego_id in vru_ids
            raw_foe_is_vru = raw_foe_id in vru_ids
        
            if raw_ego_is_vru != raw_foe_is_vru:

                N_all_ssm_with_vru += 1
            
                if len(sample_exact_vru_matches) < 10:
                    sample_exact_vru_matches.append(
                        (raw_ego_id, raw_foe_id)
                    )
            
                min_ttc_node = c.find("minTTC")
            
                min_ttc_position = None
                min_ttc_inside_polygon = False
                min_ttc_value = None
            
                if min_ttc_node is not None:
            
                    min_ttc_value = min_ttc_node.get("value")
            
                    position_text = min_ttc_node.get(
                        "position",
                        ""
                    )
            
                    if "," in position_text:
            
                        try:
            
                            px, py = map(
                                float,
                                position_text.split(",")
                            )
            
                            min_ttc_position = (
                                px,
                                py
                            )
            
                            min_ttc_inside_polygon = poly.covers(
                                Point(px, py)
                            )
            
                        except ValueError:
                            pass
            
                car_vru_diagnostics.append({
                    "ego": raw_ego_id,
                    "foe": raw_foe_id,
                    "begin": c.get("begin"),
                    "end": c.get("end"),
                    "min_ttc": min_ttc_value,
                    "position": min_ttc_position,
                    "inside_polygon": min_ttc_inside_polygon,
                })

        # --------------------------------------------------------------
        # LOCALIZATION
        # --------------------------------------------------------------

        TTCnode = c.find("minTTC")

        if TTCnode is None:
            continue

        pos = TTCnode.get("position", "")

        if "," not in pos:
            continue

        x, y = map(float, pos.split(","))

        if not poly.contains(Point(x, y)):
            continue
        
        N_local_ssm += 1

        # --------------------------------------------------------------
        # IDENTIFY CONFLICT PARTICIPANTS
        # --------------------------------------------------------------
        
        ego_id = c.get("ego")
        foe_id = c.get("foe")
        
        if ego_id is None or foe_id is None:
            continue
        
        ego_is_vru = ego_id in vru_ids
        foe_is_vru = foe_id in vru_ids
        
        # Neither participant is listed in the persons FCD.
        if not ego_is_vru and not foe_is_vru:
            N_no_vru += 1
            continue
        
        # Both participants are listed in the persons FCD.
        if ego_is_vru and foe_is_vru:
            N_both_vru += 1
            continue
        
        # Exactly one participant is a VRU.
        if ego_is_vru:
            vru_id = ego_id
            vehicle_id = foe_id
        else:
            vru_id = foe_id
            vehicle_id = ego_id
        
        N_car_vru += 1

        # --------------------------------------------------------------
        # VALID CAR-VRU CONFLICT
        # --------------------------------------------------------------

        N_conf += 1

        TTC = sf(TTCnode.get("value"), 999)

        PETnode = c.find("PET")

        PET = (
            sf(PETnode.get("value"), 999)
            if PETnode is not None
            and PETnode.get("value") != "NA"
            else 999
        )

        DRACnode = c.find("maxDRAC")

        DRAC = (
            sf(DRACnode.get("value"), 0)
            if DRACnode is not None
            else 0
        )

        # --------------------------------------------------------------
        # TIMESPAN
        # --------------------------------------------------------------

        time_values = []

        tsn = c.find("timeSpan")

        if tsn is not None:

            try:
                time_values = [
                    float(t)
                    for t in tsn.get("values", "").split()
                ]
            except:
                time_values = []

        # --------------------------------------------------------------
        # REACTIVE YIELDING
        # --------------------------------------------------------------

        if (TTC < 3 or PET < 2) and DRAC > 2.5:

            N_rea += 1
            continue

        # --------------------------------------------------------------
        # PROACTIVE YIELDING
        # --------------------------------------------------------------

        if TTC >= 3 and PET >= 2 and DRAC < 2.5:

            slowdown = vru_slowdown(
                vru_id,
                time_values,
                vru_speeds
            )
        
            if slowdown is False:
                N_pro += 1

    # ------------------------------------------------------------------
    # FINAL METRICS
    # ------------------------------------------------------------------

    N_non = max(
        0,
        N_conf - N_rea - N_pro
    )

    EA = (
        (N_rea + N_pro) / N_conf
        if N_conf > 0
        else 1.0
    )

    P_VRU = (
        N_pro / N_conf
        if N_conf > 0
        else 1.0
    )

    print("\n=== ETHICS PARTICIPANT DIAGNOSTICS ===")
    print(f"Local SSM conflicts:       {N_local_ssm}")
    print(f"Car-VRU conflicts:         {N_car_vru}")
    print(f"Conflicts without a VRU:   {N_no_vru}")
    print(f"VRU-VRU conflicts:         {N_both_vru}")
    print(f"Persons in FCD index:      {len(vru_ids)}")
    print(
        f"Vehicles in FCD index:     "
        f"{len(fcd_vehicle_ids)}"
    )
    
    print("\nSample FCD vehicles:")
    
    for vehicle in sample_fcd_vehicles:
    
        print(
            f"  id={vehicle['id']} | "
            f"type={vehicle['type']} | "
            f"x={vehicle['x']} | "
            f"y={vehicle['y']} | "
            f"speed={vehicle['speed']} | "
            f"edge={vehicle['edge']}"
        )
    print(f"All SSM conflicts:         {N_all_ssm}")
    print(f"All SSM Car-VRU matches:   {N_all_ssm_with_vru}")
    
    print("\nSample SSM participant pairs:")
    for ego_id, foe_id in sample_ssm_participants:
        print(f"  ego={ego_id} | foe={foe_id}")
    
    print("\nSample exact Car-VRU matches:")
    for ego_id, foe_id in sample_exact_vru_matches:
        print(f"  ego={ego_id} | foe={foe_id}")

    print("\nCar-VRU spatial diagnostics:")

    for item in car_vru_diagnostics:
    
        print(
            f"  begin={item['begin']} | "
            f"ego={item['ego']} | "
            f"foe={item['foe']} | "
            f"minTTC={item['min_ttc']} | "
            f"position={item['position']} | "
            f"inside_polygon={item['inside_polygon']}"
        )
    
    return {
        "N_conflicts": N_conf,
        "N_reactive": N_rea,
        "N_proactive": N_pro,
        "N_nonyield": N_non,
        "EthicalAlignment": EA,
        "VRUPriorityScore": P_VRU
    }


# ============================================================================
# COMPACT OUTPUT
# ============================================================================

def print_ethics_compact(E):

    print("\n=== ETHICS ===")

    print(f"Car–VRU conflicts:  {E['N_conflicts']}")
    print(f"Reactive yielding:  {E['N_reactive']}")
    print(f"Proactive yielding: {E['N_proactive']}")
    print(f"Non-yielding:       {E['N_nonyield']}")

    print(f"\nEthical Alignment = {E['EthicalAlignment']:.4f}")
    print(f"VRU Priority Score = {E['VRUPriorityScore']:.4f}\n")


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":

    E = compute_ethics()

    print_ethics_compact(E)