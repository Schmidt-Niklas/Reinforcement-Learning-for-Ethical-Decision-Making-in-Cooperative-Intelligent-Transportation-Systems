# -*- coding: utf-8 -*-
"""
CONFIGURATION FOR ESRM PIPELINE
--------------------------------
This is the ONLY file you modify when switching SUMO experiment folders.
All other modules import file paths from here.
"""

import os

# ============================================================================
# PATH TO SUMO OUTPUT FOLDER
# Modify this once per experiment.
# ============================================================================
SUMO_FOLDER = r"C:\Users\49151\Desktop\Master\07_Masterthesis\04_Outcome_Evaluation\Midday\U2"

ETHICS_EVENTS_FILE = os.path.join(
    SUMO_FOLDER,
    "ethics_events.json"
)

ETHICS_SUMMARY_FILE = os.path.join(
    SUMO_FOLDER,
    "ethics_summary.json"
)

SIM_BEGIN = 43200
SIM_END = 50400

# ============================================================================
# INPUT FILES (use exact names present in your SUMO output folder)
# ============================================================================
NET_FILE = os.path.join(SUMO_FOLDER, "new_ingolstadt.net.xml")
SSM_FILE = os.path.join(SUMO_FOLDER, "ssm_output.xml")
TRIPINFO_FILE = os.path.join(SUMO_FOLDER, "InTAS.simulation.tripinfo.xml")
PERSONINFO_FILE = os.path.join(SUMO_FOLDER, "InTAS.simulation.personinfo.xml")
FCD_PEDESTRIANS_FILE = os.path.join(SUMO_FOLDER, "InTAS.simulation.fcd.persons.xml")
VEHROUTE_FILE = os.path.join(SUMO_FOLDER, "InTAS.simulation.vehroute.xml")
COLLISION_FILE = os.path.join(
    SUMO_FOLDER,
    "collisions.xml"
)

# ============================================================================
# INTERSECTION GEOMETRY (polygon used for Safety and Ethics localization)
# ============================================================================
JUNCTION_POLYGON = [
    (215275.72, 450927.73), (215284.45, 450922.08), (215284.04, 450921.03),
    (215284.07, 450920.57), (215284.26, 450920.14), (215284.60, 450919.75),
    (215285.10, 450919.40), (215279.90, 450910.39), (215277.88, 450910.67),
    (215276.72, 450910.22), (215275.47, 450909.38), (215274.13, 450908.16),
    (215272.69, 450906.54), (215264.65, 450913.15), (215265.18, 450914.14),
    (215265.21, 450914.61), (215265.08, 450915.06), (215264.78, 450915.49),
    (215264.33, 450915.90), (215270.63, 450924.18), (215272.03, 450923.94),
    (215272.84, 450924.35), (215273.73, 450925.12), (215274.69, 450926.25)
]