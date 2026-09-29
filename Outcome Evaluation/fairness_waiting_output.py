# -*- coding: utf-8 -*-
"""
ESRM FAIRNESS MODULE (FINAL OUTCOME VERSION)
--------------------------------------------

Outcome Fairness Metrics:

1. F₂ Fairness (Duarte et al.)
   - F_car
   - F_bus
   - F_vru
   - F_global
   - F_min

2. Jain Group Fairness
   - aligns outcome evaluation with RL training reward

Uses:
- tripinfo.xml
- personinfo.xml
- vehroute.xml
- FCD pedestrians
"""

import xml.etree.ElementTree as ET
import numpy as np
import json
import config


# ============================================================================
# REFERENCE WAITING TIMES (RL TRAINING FORMULATION)
# ============================================================================

E_PED = 1.0
E_CAR = 70.0
E_BUS = 70.0


# ============================================================================
# LOAD INBOUND EDGES
# ============================================================================

def load_inbound_edges():

    try:
        with open("inbound_edges.json", "r") as f:
            return set(json.load(f))
    except:
        return set()


# ============================================================================
# VEHICLE + BUS WAITING TIMES
# ============================================================================

def load_vehicle_and_bus_waiting_times_local():

    inbound = load_inbound_edges()

    tree = ET.parse(config.VEHROUTE_FILE)
    root = tree.getroot()

    valid_ids = set()

    for veh in root.findall("vehicle"):

        rid = veh.get("id")

        rnode = veh.find("route")

        if rnode is None:
            continue

        edges = rnode.get("edges", "").split()

        if any(e in inbound for e in edges):
            valid_ids.add(rid)

    tree = ET.parse(config.TRIPINFO_FILE)
    root = tree.getroot()

    car_wt = []
    bus_wt = []

    for trip in root.findall("tripinfo"):

        vid = trip.get("id")

        if vid not in valid_ids:
            continue

        wt = float(trip.get("waitingTime", 0))

        vtype = trip.get("vType", "").lower()

        if "bus" in vtype:
            bus_wt.append(wt)
        else:
            car_wt.append(wt)

    return car_wt, bus_wt


# ============================================================================
# VRU WAITING TIMES
# ============================================================================

def load_vru_waiting_times_local():

    inbound = load_inbound_edges()

    tree = ET.parse(config.FCD_PEDESTRIANS_FILE)
    root = tree.getroot()

    vru_ids = set()

    for ts in root.findall("timestep"):

        for p in ts.findall("person"):

            edge = p.get("edge")

            if edge in inbound:
                vru_ids.add(p.get("id"))

    tree = ET.parse(config.PERSONINFO_FILE)
    root = tree.getroot()

    vru_wt = []

    for p in root.findall("personinfo"):

        pid = p.get("id")

        if pid not in vru_ids:
            continue

        for walk in p.findall("walk"):

            wt = float(walk.get("waitingTime", 0))
            vru_wt.append(wt)

    return vru_wt


# ============================================================================
# F₂ FAIRNESS
# ============================================================================

def compute_F2(arr):

    if len(arr) == 0:
        return 1.0

    A = np.array(arr)

    mu = A.mean()

    if mu == 0:
        return 1.0

    n = len(A)

    diff = np.abs(
        A.reshape(-1, 1) -
        A.reshape(1, -1)
    ).sum()

    return max(
        0,
        min(
            1,
            1 - diff / (2 * n * n * mu)
        )
    )


# ============================================================================
# JAIN GROUP FAIRNESS
# ============================================================================

def compute_jain_group_fairness(
    car_wt,
    bus_wt,
    vru_wt
):

    mean_car = np.mean(car_wt) if len(car_wt) > 0 else E_CAR
    mean_bus = np.mean(bus_wt) if len(bus_wt) > 0 else E_BUS
    mean_vru = np.mean(vru_wt) if len(vru_wt) > 0 else E_PED

    mean_car = max(mean_car, 0.001)
    mean_bus = max(mean_bus, 0.001)
    mean_vru = max(mean_vru, 0.001)

    x_car = E_CAR / mean_car
    x_bus = E_BUS / mean_bus
    x_vru = E_PED / mean_vru

    numerator = (x_car + x_bus + x_vru) ** 2

    denominator = 3 * (
        x_car ** 2 +
        x_bus ** 2 +
        x_vru ** 2
    )

    if denominator == 0:
        return 1.0

    return numerator / denominator


# ============================================================================
# STATS
# ============================================================================

def stats(arr):

    if len(arr) == 0:
        return dict(
            count=0,
            mean=0,
            min=0,
            max=0,
            median=0,
            std=0
        )

    A = np.array(arr)

    return dict(
        count=len(A),
        mean=float(A.mean()),
        min=float(A.min()),
        max=float(A.max()),
        median=float(np.median(A)),
        std=float(A.std())
    )


# ============================================================================
# MAIN FAIRNESS METRIC
# ============================================================================

def compute_fairness():

    car_wt, bus_wt = load_vehicle_and_bus_waiting_times_local()
    vru_wt = load_vru_waiting_times_local()

    F_car = compute_F2(car_wt)
    F_bus = compute_F2(bus_wt)
    F_vru = compute_F2(vru_wt)

    F_global = (
        F_car +
        F_bus +
        F_vru
    ) / 3

    F_min = min(
        F_car,
        F_bus,
        F_vru
    )

    F_jain = compute_jain_group_fairness(
        car_wt,
        bus_wt,
        vru_wt
    )

    return {
        "F_car": F_car,
        "F_bus": F_bus,
        "F_vru": F_vru,
        "F_global": F_global,
        "F_min": F_min,
        "F_jain_groups": F_jain,

        "car_stats": stats(car_wt),
        "bus_stats": stats(bus_wt),
        "vru_stats": stats(vru_wt),

        "car_waiting": sum(x > 0 for x in car_wt),
        "bus_waiting": sum(x > 0 for x in bus_wt),
        "vru_waiting": sum(x > 0 for x in vru_wt),
    }


# ============================================================================
# RAW EXPORT
# ============================================================================

def load_fairness_raw():

    car_wt, bus_wt = load_vehicle_and_bus_waiting_times_local()
    vru_wt = load_vru_waiting_times_local()

    return dict(
        car_waiting_times=car_wt,
        bus_waiting_times=bus_wt,
        vru_waiting_times=vru_wt
    )


# ============================================================================
# CONSOLE OUTPUT
# ============================================================================

def print_fairness_compact(F):

    print("\n=== FAIRNESS ===")

    cs = F["car_stats"]
    bs = F["bus_stats"]
    vs = F["vru_stats"]

    print(f"Cars:  n={cs['count']}, mean={cs['mean']:.2f}, std={cs['std']:.2f}")
    print(f"Buses: n={bs['count']}, mean={bs['mean']:.2f}, std={bs['std']:.2f}")
    print(f"VRUs:  n={vs['count']}, mean={vs['mean']:.2f}, std={vs['std']:.2f}")

    print(f"\nF_car         = {F['F_car']:.4f}")
    print(f"F_bus         = {F['F_bus']:.4f}")
    print(f"F_vru         = {F['F_vru']:.4f}")
    print(f"F_global      = {F['F_global']:.4f}")
    print(f"F_min         = {F['F_min']:.4f}")
    print(f"F_jain_groups = {F['F_jain_groups']:.4f}\n")