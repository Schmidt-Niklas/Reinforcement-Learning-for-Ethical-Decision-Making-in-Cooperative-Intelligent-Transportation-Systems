# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:26:34 2026

@author: 49151
"""

# -*- coding: utf-8 -*-
"""
ESRM ETHICS MODULE
------------------

Determines the final outcome ethics metrics from the existing
pair-based ESRMEthicsTracker outputs.

Inputs:
- ethics_events.json
- ethics_summary.json

No simulation is executed.
No interaction events are reconstructed from SSM or FCD.
"""

import json
import os

import config


def _safe_ratio(
    numerator,
    denominator
):
    if denominator <= 0:
        return None

    return numerator / denominator


def load_ethics_events():

    path = config.ETHICS_EVENTS_FILE

    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Ethics events file not found:\n"
            f"{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        events = json.load(file)

    if not isinstance(events, list):
        raise TypeError(
            "ethics_events.json must contain "
            "a list of interaction events."
        )

    return events


def load_tracker_summary():

    path = config.ETHICS_SUMMARY_FILE

    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Ethics summary file not found:\n"
            f"{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        summary = json.load(file)

    if not isinstance(summary, dict):
        raise TypeError(
            "ethics_summary.json must contain "
            "a JSON object."
        )

    return summary


def compute_ethics():

    events = load_ethics_events()
    tracker_summary = load_tracker_summary()

    # ==========================================================
    # INTERACTION COUNTS
    # ==========================================================

    n_interactions = len(events)

    n_proactive = sum(
        event.get("classification") == "proactive"
        for event in events
    )

    n_reactive = sum(
        event.get("classification") == "reactive"
        for event in events
    )

    n_nonyield = sum(
        event.get("classification") == "non_yielding"
        for event in events
    )
    

    n_unclassified = sum(
        event.get("classification") == "unclassified"
        for event in events
    )

    n_classified = (
        n_proactive
        + n_reactive
        + n_nonyield
    )
    
    # ==========================================================
    # CONSISTENCY WITH ORIGINAL TRACKER SUMMARY
    # ==========================================================
    
    comparison_keys = {
        "N_conflicts": n_interactions,
        "N_classified": n_classified,
        "N_proactive": n_proactive,
        "N_reactive": n_reactive,
        "N_nonyield": n_nonyield,
        "N_unclassified": n_unclassified,
    }
    
    for key, recomputed_value in comparison_keys.items():
    
        if key not in tracker_summary:
            raise KeyError(
                "Missing key in original "
                "ethics_summary.json: "
                f"{key}"
            )
    
        stored_value = tracker_summary[key]
    
        if stored_value != recomputed_value:
            raise ValueError(
                "Mismatch between ethics_events.json "
                "and ethics_summary.json for "
                f"{key}: stored={stored_value}, "
                f"recomputed={recomputed_value}"
            )
    
    # ==========================================================
    # BURDEN ALLOCATION
    # ==========================================================
    
    n_vehicle_only = sum(
        event.get("vehicle_braked") is True
        and event.get("vru_slowed") is False
        for event in events
    )
    
    n_shared = sum(
        event.get("vehicle_braked") is True
        and event.get("vru_slowed") is True
        for event in events
    )
    
    n_vru_only = sum(
        event.get("vehicle_braked") is False
        and event.get("vru_slowed") is True
        for event in events
    )
    
    n_no_mitigation = sum(
        event.get("vehicle_braked") is False
        and event.get("vru_slowed") is False
        for event in events
    )
    
    n_mitigated = (
        n_vehicle_only
        + n_shared
        + n_vru_only
    )
    
    # ==========================================================
    # INTERACTION CRITICALITY
    # ==========================================================

    n_critical = sum(
        event.get("critical_state_reached") is True
        for event in events
    )

    n_noncritical = sum(
        event.get("critical_state_reached") is False
        for event in events
    )

    n_unknown_criticality = (
        n_interactions
        - n_critical
        - n_noncritical
    )

    # ==========================================================
    # COVERAGE AND ALIGNMENT
    # ==========================================================

    classification_coverage = _safe_ratio(
        n_classified,
        n_interactions
    )

    ethical_alignment = _safe_ratio(
        n_proactive + n_reactive,
        n_interactions
    )

    ethical_alignment_classified = _safe_ratio(
        n_proactive + n_reactive,
        n_classified
    )

    # ==========================================================
    # VRU PRIORITY SCORE VERSION 2
    # ==========================================================
    #
    # Evaluates the allocation of mitigation burden in
    # vehicle-VRU interactions with an observable response.
    #
    # Vehicle-only mitigation receives weight 1.0.
    # Shared mitigation receives weight 0.5.
    # VRU-only mitigation receives weight 0.0.
    #
    # Interactions without observable mitigation are excluded
    # from the denominator.
    # ===========================================

    shared_weight = 0.5

    vru_priority_score_v2 = _safe_ratio(
        n_vehicle_only
        + shared_weight * n_shared,
        n_mitigated
    )

    result = {
        # Interaction overview
        "N_interactions": n_interactions,
    
        # Backward-compatible alias
        "N_conflicts": n_interactions,
    
        "N_classified": n_classified,
        "N_unclassified": n_unclassified,
    
        # Interaction criticality
        "N_noncritical": n_noncritical,
        "N_critical": n_critical,
        "N_unknown_criticality": n_unknown_criticality,
    
        # Yielding behaviour
        "N_proactive": n_proactive,
        "N_reactive": n_reactive,
        "N_nonyield": n_nonyield,
    
        # Burden allocation
        "N_vehicle_only": n_vehicle_only,
        "N_shared": n_shared,
        "N_vru_only": n_vru_only,
        "N_mitigated": n_mitigated,
        "SharedWeight": shared_weight,
        "N_no_mitigation": n_no_mitigation,
    
        # Aggregated metrics
        "ClassificationCoverage": classification_coverage,
        "EthicalAlignment": ethical_alignment,
        "EthicalAlignmentClassified": (
            ethical_alignment_classified
        ),
        "VRUPriorityScoreV2": (
            vru_priority_score_v2
        ),
    
        # Reproducibility
        "TrackerSummaryOriginal": tracker_summary,
    }
    
    validate_ethics_result(
        result
    )

    return result


def validate_ethics_result(E):

    if (
        E["N_interactions"]
        != E["N_classified"]
        + E["N_unclassified"]
    ):
        raise ValueError(
            "Ethics inconsistency: "
            "N_interactions must equal "
            "N_classified + N_unclassified."
        )

    if (
        E["N_classified"]
        != E["N_proactive"]
        + E["N_reactive"]
        + E["N_nonyield"]
    ):
        raise ValueError(
            "Ethics inconsistency: "
            "N_classified must equal "
            "N_proactive + N_reactive "
            "+ N_nonyield."
        )

    if (
        E["N_interactions"]
        != E["N_vehicle_only"]
        + E["N_shared"]
        + E["N_vru_only"]
        + E["N_no_mitigation"]
    ):
        raise ValueError(
            "Ethics inconsistency: "
            "N_interactions must equal "
            "N_vehicle_only + N_shared + "
            "N_vru_only + N_no_mitigation."
        )

    
    vru_priority_score_v2 = E[
        "VRUPriorityScoreV2"
    ]
    
    if (
        vru_priority_score_v2 is not None
        and not (
            0.0
            <= vru_priority_score_v2
            <= 1.0
        )
    ):
        raise ValueError(
            "Ethics inconsistency: "
            "VRUPriorityScoreV2 must be "
            "between 0 and 1."
        )
    
    if (
        E["N_interactions"]
        != E["N_vehicle_only"]
        + E["N_shared"]
        + E["N_vru_only"]
        + E["N_no_mitigation"]
    ):
        raise ValueError(
            "Ethics inconsistency: "
            "burden-allocation categories must "
            "sum to N_interactions."
        )


def _format_metric(value):

    if value is None:
        return "N/A"

    return f"{value:.4f}"


def print_ethics_compact(E):

    coverage = _format_metric(
        E["ClassificationCoverage"]
    )

    ea_all = _format_metric(
        E["EthicalAlignment"]
    )

    ea_classified = _format_metric(
        E["EthicalAlignmentClassified"]
    )

    vru_v2 = _format_metric(
        E["VRUPriorityScoreV2"]
    )

    print("\n=== ETHICS ===")

    print("\n[INTERACTION OVERVIEW]")

    print(
        f"Total interactions:        "
        f"{E['N_interactions']}"
    )

    print(
        f"Classified interactions:   "
        f"{E['N_classified']}"
    )

    print(
        f"Unclassified interactions: "
        f"{E['N_unclassified']}"
    )

    print(
        f"Classification coverage:   "
        f"{coverage}"
    )

    print("\n[INTERACTION CRITICALITY]")

    print(
        f"Non-critical interactions: "
        f"{E['N_noncritical']}"
    )

    print(
        f"Critical interactions:     "
        f"{E['N_critical']}"
    )

    if E["N_unknown_criticality"] > 0:
        print(
            f"Unknown criticality:       "
            f"{E['N_unknown_criticality']}"
        )

    print("\n[YIELDING BEHAVIOUR]")

    print(
        f"Proactive yielding:        "
        f"{E['N_proactive']}"
    )

    print(
        f"Reactive yielding:         "
        f"{E['N_reactive']}"
    )

    print(
        f"Non-yielding:              "
        f"{E['N_nonyield']}"
    )

    print("\n[BURDEN ALLOCATION]")

    print(
        f"Vehicle-only mitigation:   "
        f"{E['N_vehicle_only']}"
    )

    print(
        f"Shared mitigation:         "
        f"{E['N_shared']}"
    )

    print(
        f"VRU-only mitigation:       "
        f"{E['N_vru_only']}"
    )

    print(
        f"No observable mitigation:  "
        f"{E['N_no_mitigation']}"
    )

    print(
        f"Interactions with response:"
        f"{E['N_mitigated']}"
    )

    print("\n[AGGREGATED METRICS]")

    print(
        f"Ethical Alignment:         "
        f"{ea_all}"
    )

    print(
        f"Ethical Alignment "
        f"Classified: {ea_classified}"
    )

    print(
        f"VRU Priority Score V2:     "
        f"{vru_v2}"
    )

    print()
    