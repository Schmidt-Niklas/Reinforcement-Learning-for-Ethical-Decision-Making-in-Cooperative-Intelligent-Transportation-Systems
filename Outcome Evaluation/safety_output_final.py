# -*- coding: utf-8 -*-
"""
ESRM SAFETY MODULE (FINAL OUTCOME VERSION)
------------------------------------------

Safety Metrics:

- Collision Count
- TTC Critical Rate
- PET Critical Rate
- Distance Critical Rate
- DRAC Critical Rate
- TIT (Time Integrated TTC)

No aggregated Safety Score is reported.

Following:
Duarte et al. Evaluation Framework
"""

import xml.etree.ElementTree as ET
from shapely.geometry import Point, Polygon
import math
import config


POLY = Polygon(config.JUNCTION_POLYGON)


# ============================================================================
# HELPERS
# ============================================================================

def safe_float(v, default=None):
    try:
        return float(v)
    except:
        return default


# ============================================================================
# COLLISIONS
# ============================================================================

def load_collisions():

    try:
        root = ET.parse(config.COLLISION_FILE).getroot()
    except:
        return 0

    count = 0

    for col in root.findall("collision"):

        for attr in (
            "colliderFront",
            "colliderBack",
            "victimFront",
            "victimBack"
        ):

            p = col.get(attr)

            if p and "," in p:

                x, y = map(float, p.split(","))

                if POLY.contains(Point(x, y)):
                    count += 1
                    break

    return count


# ============================================================================
# MINIMUM DISTANCE
# ============================================================================

def compute_min_dist(conf):

    ego = conf.find("egoPosition")
    foe = conf.find("foePosition")

    if ego is None or foe is None:
        return None

    ego_vals = ego.get("values", "").split()
    foe_vals = foe.get("values", "").split()

    N = min(len(ego_vals), len(foe_vals))

    if N == 0:
        return None

    min_d = None

    for i in range(N):

        if "," not in ego_vals[i]:
            continue

        if "," not in foe_vals[i]:
            continue

        x1, y1 = map(float, ego_vals[i].split(","))
        x2, y2 = map(float, foe_vals[i].split(","))

        d = math.hypot(x1 - x2, y1 - y2)

        if min_d is None or d < min_d:
            min_d = d

    return min_d


# ============================================================================
# TIT
# ============================================================================

def compute_TIT(conf):

    TTCnode = conf.find("TTCSpan")
    TIMEnode = conf.find("timeSpan")

    if TTCnode is None:
        return 0.0

    if TIMEnode is None:
        return 0.0

    ttc_raw = TTCnode.get("values", "").split()
    time_raw = TIMEnode.get("values", "").split()

    if len(ttc_raw) < 2:
        return 0.0

    if len(time_raw) < 2:
        return 0.0

    N = min(
        len(ttc_raw),
        len(time_raw)
    )

    TIT = 0.0

    for i in range(N - 1):

        try:

            if ttc_raw[i] == "NA":
                continue

            ttc = float(ttc_raw[i])

            dt = (
                float(time_raw[i + 1])
                - float(time_raw[i])
            )

        except:
            continue

        if ttc < 3.0:

            TIT += (
                (3.0 - ttc)
                * dt
            )

    return TIT


# ============================================================================
# LOCAL CONFLICTS
# ============================================================================

def load_local_conflicts():

    root = ET.parse(config.SSM_FILE).getroot()

    conflicts = []

    types = []

    for c in root.findall("conflict"):

        m = c.find("minTTC")

        if m is None:
            continue

        pos = m.get("position", "")

        if "," not in pos:
            continue

        x, y = map(float, pos.split(","))

        if not POLY.contains(Point(x, y)):
            continue

        ego = c.get("ego", "").lower()
        foe = c.get("foe", "").lower()

        def cls(x):

            if "bus" in x:
                return "bus"

            if x.startswith("h"):
                return "vru"

            return "car"

        try:
            pair = "-".join(
                sorted(
                    [cls(ego), cls(foe)]
                )
            )
        except:
            pair = "unknown"

        conflicts.append(c)
        types.append(pair)

    return conflicts, types


# ============================================================================
# MAIN
# ============================================================================

def compute_safety():

    confs, types = load_local_conflicts()

    TTCcrit = 0
    PETcrit = 0
    DISTcrit = 0
    DRACcrit = 0

    TIT = 0.0

    for c in confs:

        TTC = safe_float(
            c.find("minTTC").get("value", "999"),
            999
        )

        if TTC < 3:
            TTCcrit += 1

        PETnode = c.find("PET")

        PET = (
            safe_float(PETnode.get("value"))
            if PETnode is not None
            else None
        )

        if PET is not None and PET < 2:
            PETcrit += 1

        DRACnode = c.find("maxDRAC")

        DRAC = (
            safe_float(DRACnode.get("value"))
            if DRACnode is not None
            else None
        )

        if DRAC is not None and DRAC > 2.5:
            DRACcrit += 1

        dist = compute_min_dist(c)

        if dist is not None and dist < 2:
            DISTcrit += 1

        TIT += compute_TIT(c)

    return {
        "N_conflicts": len(confs),
        "Collisions": load_collisions(),
        "TTCcrit": TTCcrit,
        "PETcrit": PETcrit,
        "DISTcrit": DISTcrit,
        "DRACcrit": DRACcrit,
        "TIT": TIT
    }


# ============================================================================
# RAW EXPORT
# ============================================================================

def load_safety_raw():

    confs, types = load_local_conflicts()

    raw = []

    for c, ct in zip(confs, types):

        raw.append({
            "type": ct,
            "TIT": compute_TIT(c),
            "minDist": compute_min_dist(c)
        })

    return raw


# ============================================================================
# PRINT
# ============================================================================

def print_safety_compact(S):

    print("\n=== SAFETY ===")

    print(f"Collisions:            {S['Collisions']}")
    print(f"Conflicts:             {S['N_conflicts']}")
    print(f"TTC-critical:          {S['TTCcrit']}")
    print(f"PET-critical:          {S['PETcrit']}")
    print(f"DIST-critical:         {S['DISTcrit']}")
    print(f"DRAC-critical:         {S['DRACcrit']}")
    print(f"TIT:                   {S['TIT']:.4f}\n")


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":

    S = compute_safety()

    print_safety_compact(S)