# -*- coding: utf-8 -*-
"""
ESRM THROUGHPUT MODULE (FINAL OUTCOME VERSION)
----------------------------------------------

Implements the final throughput evaluation metrics:

1. Throughput
   - TP_car
   - TP_bus
   - TP_ped
   - TP_total

2. Average Travel Time (ATT)
   - ATT_car
   - ATT_bus
   - ATT_ped
   - ATT_global

Delay Reduction (DR) is intentionally not calculated here,
because it requires comparison between Baseline, U1 and U2 runs.
"""

import xml.etree.ElementTree as ET
import numpy as np
import config


# ============================================================================
# HELPERS
# ============================================================================

def _safe_float(value, default=0.0):
    try:
        return float(value)
    except:
        return default


# ============================================================================
# LOAD VEHICLE TRAVEL TIMES
# ============================================================================

def load_vehicle_travel_times():

    tree = ET.parse(config.TRIPINFO_FILE)
    root = tree.getroot()

    car_tt = []
    bus_tt = []

    for trip in root.findall("tripinfo"):

        travel_time = _safe_float(trip.get("duration"))

        vtype = trip.get("vType", "").lower()

        if "bus" in vtype:
            bus_tt.append(travel_time)
        else:
            car_tt.append(travel_time)

    return car_tt, bus_tt


# ============================================================================
# LOAD PEDESTRIAN TRAVEL TIMES
# ============================================================================

def load_pedestrian_travel_times():

    tree = ET.parse(config.PERSONINFO_FILE)
    root = tree.getroot()

    ped_tt = []

    for person in root.findall("personinfo"):

        duration = _safe_float(
            person.get("duration")
        )

        if duration > 0:
            ped_tt.append(duration)

    return ped_tt


# ============================================================================
# THROUGHPUT
# ============================================================================

def compute_throughput():

    car_tt, bus_tt = load_vehicle_travel_times()
    ped_tt = load_pedestrian_travel_times()

    # ----------------------------------------------------------------------
    # COUNTS
    # ----------------------------------------------------------------------

    N_car = len(car_tt)
    N_bus = len(bus_tt)
    N_ped = len(ped_tt)

    N_total = N_car + N_bus + N_ped

    # ----------------------------------------------------------------------
    # SIMULATION DURATION
    # ----------------------------------------------------------------------

    sim_duration = 1.0

    try:
        sim_duration = float(config.SIM_END) - float(config.SIM_BEGIN)
    except:
        pass

    if sim_duration <= 0:
        sim_duration = 1.0

    sim_minutes = sim_duration / 60.0

    # ----------------------------------------------------------------------
    # THROUGHPUT [participants/min]
    # ----------------------------------------------------------------------

    TP_car = N_car / sim_minutes
    TP_bus = N_bus / sim_minutes
    TP_ped = N_ped / sim_minutes

    TP_total = N_total / sim_minutes

    # ----------------------------------------------------------------------
    # ATT
    # ----------------------------------------------------------------------

    ATT_car = float(np.mean(car_tt)) if car_tt else 0.0
    ATT_bus = float(np.mean(bus_tt)) if bus_tt else 0.0
    ATT_ped = float(np.mean(ped_tt)) if ped_tt else 0.0

    all_tt = car_tt + bus_tt + ped_tt
    ATT_global = float(np.mean(all_tt)) if all_tt else 0.0

    return {
        "TP_car": TP_car,
        "TP_bus": TP_bus,
        "TP_ped": TP_ped,
        "TP_total": TP_total,
        "N_car": N_car,
        "N_bus": N_bus,
        "N_ped": N_ped,
        "N_total": N_total,
        "ATT_car": ATT_car,
        "ATT_bus": ATT_bus,
        "ATT_ped": ATT_ped,
        "ATT_global": ATT_global
    }


# ============================================================================
# RAW EXPORT
# ============================================================================

def load_throughput_raw():

    data = compute_throughput()

    return {
        "N_car": data["N_car"],
        "N_bus": data["N_bus"],
        "N_ped": data["N_ped"],
        "N_total": data["N_total"]
    }


# ============================================================================
# COMPACT PRINT
# ============================================================================

def print_throughput_compact(T):

    print("\n=== THROUGHPUT & EFFICIENCY ===")

    print("\n[THROUGHPUT]")
    print(f"Cars/min:         {T['TP_car']:.2f}")
    print(f"Buses/min:        {T['TP_bus']:.2f}")
    print(f"Pedestrians/min:  {T['TP_ped']:.2f}")
    print(f"Total/min:        {T['TP_total']:.2f}")

    print("\n[AVERAGE TRAVEL TIME]")
    print(f"Cars:             {T['ATT_car']:.2f} s")
    print(f"Buses:            {T['ATT_bus']:.2f} s")
    print(f"Pedestrians:      {T['ATT_ped']:.2f} s")
    print(f"Global:           {T['ATT_global']:.2f} s\n")


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":

    T = compute_throughput()
    print_throughput_compact(T)