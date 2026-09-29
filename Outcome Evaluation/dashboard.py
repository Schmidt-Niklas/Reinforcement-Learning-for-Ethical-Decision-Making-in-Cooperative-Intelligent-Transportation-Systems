# -*- coding: utf-8 -*-
"""
ESRM DASHBOARD
--------------

Aggregates:
- Safety
- Traffic Performance
- Fairness
- Ethics

Exports:
- esrm_dashboard.json

The dashboard is written directly into the currently selected
scenario folder defined by config.SUMO_FOLDER.
"""

import json
import os

import config

from throughput_output import (
    compute_throughput,
    load_throughput_raw,
    print_throughput_compact,
)

from safety_output_final import (
    compute_safety,
    load_safety_raw,
    print_safety_compact,
)

from fairness_waiting_output import (
    compute_fairness,
    load_fairness_raw,
    print_fairness_compact,
)

from ethics_output_final_V2 import (
    compute_ethics,
    print_ethics_compact,
)


# ============================================================================
# HELPERS
# ============================================================================

def round_or_none(
    value,
    digits=4,
):
    """
    Rounds numeric values while preserving None.
    """

    if value is None:
        return None

    return round(
        float(value),
        digits,
    )


# ============================================================================
# BUILD DASHBOARD
# ============================================================================

def build_esrm_dashboard(
    export_json=True,
):

    # ----------------------------------------------------------------------
    # SAFETY
    # ----------------------------------------------------------------------

    S = compute_safety()

    # ----------------------------------------------------------------------
    # TRAFFIC PERFORMANCE
    # ----------------------------------------------------------------------

    T = compute_throughput()

    # ----------------------------------------------------------------------
    # FAIRNESS
    # ----------------------------------------------------------------------

    F = compute_fairness()

    # ----------------------------------------------------------------------
    # ETHICS
    # ----------------------------------------------------------------------

    E = compute_ethics()

    # ----------------------------------------------------------------------
    # TERMINAL OUTPUT
    # ----------------------------------------------------------------------

    print(
        "\n==================== ESRM DASHBOARD ====================\n"
    )

    print("SAFETY")
    print_safety_compact(S)

    print("TRAFFIC PERFORMANCE")
    print_throughput_compact(T)

    print("FAIRNESS")
    print_fairness_compact(F)

    print("ETHICS")
    print_ethics_compact(E)

    print(
        "\n========================================================\n"
    )

    # ----------------------------------------------------------------------
    # JSON EXPORT STRUCTURE
    # ----------------------------------------------------------------------

    dashboard = {

        # ==================================================================
        # SAFETY
        # ==================================================================

        "SAFETY": {

            "Collisions": S["Collisions"],
            "Conflicts": S["N_conflicts"],

            "TTC_critical": S["TTCcrit"],
            "PET_critical": S["PETcrit"],
            "DIST_critical": S["DISTcrit"],
            "DRAC_critical": S["DRACcrit"],

            "TIT": round_or_none(
                S["TIT"]
            ),
        },

        # ==================================================================
        # TRAFFIC PERFORMANCE
        # ==================================================================

        "TRAFFIC_PERFORMANCE": {

            # --------------------------------------------------------------
            # Throughput at the investigated intersection
            # --------------------------------------------------------------

            "Throughput": {

                "Cars_per_min": round_or_none(
                    T["TP_car"]
                ),

                "Buses_per_min": round_or_none(
                    T["TP_bus"]
                ),

                "Pedestrians_per_min": round_or_none(
                    T["TP_ped"]
                ),

                "Total_per_min": round_or_none(
                    T["TP_total"]
                ),
            },

            # --------------------------------------------------------------
            # Absolute participant counts
            # --------------------------------------------------------------

            "Counts": {

                "Cars": T["N_car"],
                "Buses": T["N_bus"],
                "Pedestrians": T["N_ped"],
                "Total": T["N_total"],
            },

            # --------------------------------------------------------------
            # Current ATT values
            #
            # These values currently refer to complete network trips.
            # They must later be replaced by intersection-local travel
            # times or delays.
            # --------------------------------------------------------------

            "AverageTravelTime_Preliminary": {

                "Scope": "complete_simulation_network",

                "Cars_seconds": round_or_none(
                    T["ATT_car"]
                ),

                "Buses_seconds": round_or_none(
                    T["ATT_bus"]
                ),

                "Pedestrians_seconds": round_or_none(
                    T["ATT_ped"]
                ),

                "Global_seconds": round_or_none(
                    T["ATT_global"]
                ),
            },
        },

        # ==================================================================
        # FAIRNESS
        # ==================================================================

        "FAIRNESS": {

            "F_car": round_or_none(
                F["F_car"]
            ),

            "F_bus": round_or_none(
                F["F_bus"]
            ),

            "F_vru": round_or_none(
                F["F_vru"]
            ),

            "F_global": round_or_none(
                F["F_global"]
            ),

            "F_min": round_or_none(
                F["F_min"]
            ),

            "F_jain_groups": round_or_none(
                F["F_jain_groups"]
            ),

            "car_stats": F["car_stats"],
            "bus_stats": F["bus_stats"],
            "vru_stats": F["vru_stats"],
        },

        # ==================================================================
        # ETHICS
        # ==================================================================

        "ETHICS": {

            # --------------------------------------------------------------
            # Interaction overview
            # --------------------------------------------------------------

            "InteractionOverview": {

                "TotalInteractions": (
                    E["N_interactions"]
                ),

                "ClassifiedInteractions": (
                    E["N_classified"]
                ),

                "UnclassifiedInteractions": (
                    E["N_unclassified"]
                ),

                "ClassificationCoverage": (
                    round_or_none(
                        E["ClassificationCoverage"]
                    )
                ),
            },

            # --------------------------------------------------------------
            # Interaction criticality
            # --------------------------------------------------------------

            "InteractionCriticality": {

                "NonCriticalInteractions": (
                    E["N_noncritical"]
                ),

                "CriticalInteractions": (
                    E["N_critical"]
                ),

                "UnknownCriticality": (
                    E["N_unknown_criticality"]
                ),
            },

            # --------------------------------------------------------------
            # Yielding behaviour
            # --------------------------------------------------------------

            "YieldingBehaviour": {

                "ProactiveYielding": (
                    E["N_proactive"]
                ),

                "ReactiveYielding": (
                    E["N_reactive"]
                ),

                "NonYielding": (
                    E["N_nonyield"]
                ),
            },

            # --------------------------------------------------------------
            # Burden allocation
            # --------------------------------------------------------------

            "BurdenAllocation": {

                "VehicleOnlyMitigation": (
                    E["N_vehicle_only"]
                ),

                "SharedMitigation": (
                    E["N_shared"]
                ),

                "VRUOnlyMitigation": (
                    E["N_vru_only"]
                ),

                "MitigatedInteractions": (
                    E["N_mitigated"]
                ),

                "SharedWeight": round_or_none(
                    E["SharedWeight"]
                ),
            },

            # --------------------------------------------------------------
            # Aggregated ethics metrics
            # --------------------------------------------------------------

            "AggregatedMetrics": {

                "EthicalAlignment": (
                    round_or_none(
                        E["EthicalAlignment"]
                    )
                ),

                "EthicalAlignmentClassified": (
                    round_or_none(
                        E[
                            "EthicalAlignmentClassified"
                        ]
                    )
                ),

                "VRUPriorityScoreV2": (
                    round_or_none(
                        E["VRUPriorityScoreV2"]
                    )
                ),
            },
        },

        # ==================================================================
        # RAW VALUES
        # ==================================================================

        "RAW": {

            "SAFETY": load_safety_raw(),

            "TRAFFIC_PERFORMANCE": (
                load_throughput_raw()
            ),

            "FAIRNESS": load_fairness_raw(),

            "ETHICS": E,
        },
    }

    # ----------------------------------------------------------------------
    # WRITE JSON
    # ----------------------------------------------------------------------

    if export_json:

        output_path = os.path.join(
            config.SUMO_FOLDER,
            "esrm_dashboard.json",
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                dashboard,
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(
            "[EXPORT] ESRM dashboard written to:\n"
            f"{output_path}\n"
        )

    return dashboard


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    build_esrm_dashboard()