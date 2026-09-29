# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 13:25:32 2026

@author: 49151
"""

#analyze_ethics_events_v2.py

# -*- coding: utf-8 -*-
"""
analyze_ethics_events_v2.py

Offline re-evaluation of existing ethics_events.json files.

The script does not modify the original tracker outputs.

For every event, it determines:

1. Timing of mitigation:
   - proactive
   - reactive
   - no_mitigation

2. analyze_ethics_events_v2.py

Offline re-evaluation of existing ethics_events.json files.

The original tracker outputs are not modified.

Each tracked vehicle-VRU interaction is classified into one of
the following mutually exclusive outcome categories:

1. Benign passage
   No critical state and no measurable mitigation was required.

2. Proactive vehicle-only mitigation
3. Proactive shared mitigation
4. Proactive VRU-only mitigation

5. Reactive vehicle-only mitigation
6. Reactive shared mitigation
7. Reactive VRU-only mitigation

8. Unresolved critical interaction
   A critical state was reached without a measurable response.

Generated files per scenario:
- ethics_events_v2.json
- ethics_summary_v2.json

A combined comparison is written to:
- ethics_comparison_v2.json

Scenarios:
- Baseline
- U1
- U2


Outputs per scenario:
- ethics_events_v2.json
- ethics_summary_v2.json

Scenarios:
- Baseline
- U1
- U2
"""

import json
import os
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

RESULT_ROOT = (
    r"C:\Users\49151\Desktop\Master\07_Masterthesis"
    r"\04_Outcome_Evaluation\Midday"
)

SCENARIOS = [
    "Baseline",
    "U1",
    "U2",
]

INPUT_FILENAME = "ethics_events.json"
OUTPUT_EVENTS_FILENAME = "ethics_events_v2.json"
OUTPUT_SUMMARY_FILENAME = "ethics_summary_v2.json"

# Weight assigned to shared mitigation in the optional
# weighted Ethical Alignment metric.
SHARED_WEIGHT = 0.5


# ============================================================
# EVENT CLASSIFICATION
# ============================================================

def validate_input_event(
    event,
    event_index,
    scenario
):

    required_fields = {
        "vehicle_id",
        "vru_id",
        "critical_state_reached",
        "vehicle_braked",
        "vru_slowed",
    }

    missing_fields = (
        required_fields
        - set(event.keys())
    )

    if missing_fields:

        raise KeyError(
            f"Scenario {scenario}, event "
            f"{event_index}: missing fields "
            f"{sorted(missing_fields)}"
        )

    boolean_fields = [
        "critical_state_reached",
        "vehicle_braked",
        "vru_slowed",
    ]

    for field in boolean_fields:

        if not isinstance(
            event[field],
            bool
        ):

            raise TypeError(
                f"Scenario {scenario}, event "
                f"{event_index}: field {field} "
                "must be boolean, found "
                f"{type(event[field]).__name__}."
            )

    if event["vehicle_id"] is None:

        raise ValueError(
            f"Scenario {scenario}, event "
            f"{event_index}: vehicle_id is None."
        )

    if event["vru_id"] is None:

        raise ValueError(
            f"Scenario {scenario}, event "
            f"{event_index}: vru_id is None."
        )

    return True


def classify_burden(event):

    vehicle_braked = bool(
        event.get("vehicle_braked", False)
    )

    vru_slowed = bool(
        event.get("vru_slowed", False)
    )

    if vehicle_braked and not vru_slowed:
        return "vehicle_only"

    if vehicle_braked and vru_slowed:
        return "shared"

    if not vehicle_braked and vru_slowed:
        return "vru_only"

    return "none"


def classify_event_v2(event):

    vehicle_braked = bool(
        event.get("vehicle_braked", False)
    )

    vru_slowed = bool(
        event.get("vru_slowed", False)
    )

    critical = bool(
        event.get(
            "critical_state_reached",
            False
        )
    )

    burden = classify_burden(event)

    mitigation_detected = (
        vehicle_braked
        or vru_slowed
    )

    # ----------------------------------------------------
    # Conflict-free passage
    # ----------------------------------------------------

    if not critical and not mitigation_detected:

        outcome = "benign_passage"
        timing = "not_required"

    # ----------------------------------------------------
    # Critical situation without measurable response
    # ----------------------------------------------------

    elif critical and not mitigation_detected:

        outcome = "unresolved_critical"
        timing = "no_response"

    # ----------------------------------------------------
    # Measurable mitigation before criticality
    # ----------------------------------------------------

    elif not critical and mitigation_detected:

        timing = "proactive"
        outcome = f"proactive_{burden}"

    # ----------------------------------------------------
    # Measurable mitigation in critical situation
    # ----------------------------------------------------

    else:

        timing = "reactive"
        outcome = f"reactive_{burden}"

    enriched_event = dict(event)

    enriched_event.update({
        "timing_classification_v2": timing,
        "burden_classification_v2": burden,
        "outcome_classification_v2": outcome,
    })

    return enriched_event


# ============================================================
# SUMMARY CALCULATION
# ============================================================

def calculate_summary(events):

    outcome_counts = Counter(
        event["outcome_classification_v2"]
        for event in events
    )

    n_events = len(events)

    # ----------------------------------------------------
    # Main outcome categories
    # ----------------------------------------------------

    n_benign = outcome_counts[
        "benign_passage"
    ]

    n_unresolved_critical = outcome_counts[
        "unresolved_critical"
    ]

    # ----------------------------------------------------
    # Proactive categories
    # ----------------------------------------------------

    n_proactive_vehicle_only = outcome_counts[
        "proactive_vehicle_only"
    ]

    n_proactive_shared = outcome_counts[
        "proactive_shared"
    ]

    n_proactive_vru_only = outcome_counts[
        "proactive_vru_only"
    ]

    # ----------------------------------------------------
    # Reactive categories
    # ----------------------------------------------------

    n_reactive_vehicle_only = outcome_counts[
        "reactive_vehicle_only"
    ]

    n_reactive_shared = outcome_counts[
        "reactive_shared"
    ]

    n_reactive_vru_only = outcome_counts[
        "reactive_vru_only"
    ]

    # ----------------------------------------------------
    # Burden totals
    # ----------------------------------------------------

    n_vehicle_only = (
        n_proactive_vehicle_only
        + n_reactive_vehicle_only
    )

    n_shared = (
        n_proactive_shared
        + n_reactive_shared
    )

    n_vru_only = (
        n_proactive_vru_only
        + n_reactive_vru_only
    )

    n_mitigated = (
        n_vehicle_only
        + n_shared
        + n_vru_only
    )

    n_critical = (
        n_reactive_vehicle_only
        + n_reactive_shared
        + n_reactive_vru_only
        + n_unresolved_critical
    )

    n_opportunities = (
        n_mitigated
        + n_unresolved_critical
    )

    # ----------------------------------------------------
    # All-event metrics
    # ----------------------------------------------------

    if n_events > 0:

        benign_passage_rate = (
            n_benign / n_events
        )

        unresolved_critical_rate_all = (
            n_unresolved_critical / n_events
        )

        mitigation_opportunity_rate = (
            n_opportunities / n_events
        )

        system_level_ethical_outcome = (
            n_benign
            + n_vehicle_only
            + SHARED_WEIGHT * n_shared
        ) / n_events

    else:

        benign_passage_rate = None
        unresolved_critical_rate_all = None
        mitigation_opportunity_rate = None
        system_level_ethical_outcome = None

    # ----------------------------------------------------
    # Mitigated-event metrics
    # ----------------------------------------------------

    if n_mitigated > 0:

        vehicle_contribution_rate = (
            n_vehicle_only + n_shared
        ) / n_mitigated

        strict_vru_priority_rate = (
            n_vehicle_only / n_mitigated
        )

        shared_mitigation_rate = (
            n_shared / n_mitigated
        )

        vru_only_burden_rate = (
            n_vru_only / n_mitigated
        )

        proactive_system_mitigation_rate = (
            n_proactive_vehicle_only
            + n_proactive_shared
        ) / n_mitigated

    else:

        vehicle_contribution_rate = None
        strict_vru_priority_rate = None
        shared_mitigation_rate = None
        vru_only_burden_rate = None
        proactive_system_mitigation_rate = None

    # ----------------------------------------------------
    # Critical-event metrics
    # ----------------------------------------------------

    if n_critical > 0:

        unresolved_critical_rate = (
            n_unresolved_critical / n_critical
        )

    else:

        unresolved_critical_rate = None

    # ----------------------------------------------------
    # Opportunity-conditioned alignment
    # ----------------------------------------------------

    if n_opportunities > 0:

        ethical_alignment_opportunity = (
            n_vehicle_only
            + SHARED_WEIGHT * n_shared
        ) / n_opportunities

    else:

        ethical_alignment_opportunity = None

    return {
        "N_tracked_events": n_events,

        "N_benign_passage": n_benign,
        "N_mitigation_opportunities": (
            n_opportunities
        ),
        "N_mitigated": n_mitigated,
        "N_critical": n_critical,
        "N_unresolved_critical": (
            n_unresolved_critical
        ),

        "N_vehicle_only": n_vehicle_only,
        "N_shared": n_shared,
        "N_vru_only": n_vru_only,

        "N_proactive_vehicle_only": (
            n_proactive_vehicle_only
        ),
        "N_proactive_shared": (
            n_proactive_shared
        ),
        "N_proactive_vru_only": (
            n_proactive_vru_only
        ),

        "N_reactive_vehicle_only": (
            n_reactive_vehicle_only
        ),
        "N_reactive_shared": (
            n_reactive_shared
        ),
        "N_reactive_vru_only": (
            n_reactive_vru_only
        ),

        "BenignPassageRate": (
            benign_passage_rate
        ),

        "MitigationOpportunityRate": (
            mitigation_opportunity_rate
        ),

        "VehicleContributionRate": (
            vehicle_contribution_rate
        ),

        "StrictVRUPriorityRate": (
            strict_vru_priority_rate
        ),

        "SharedMitigationRate": (
            shared_mitigation_rate
        ),

        "VRUOnlyBurdenRate": (
            vru_only_burden_rate
        ),

        "ProactiveSystemMitigationRate": (
            proactive_system_mitigation_rate
        ),

        "UnresolvedCriticalRateAll": (
            unresolved_critical_rate_all
        ),

        "UnresolvedCriticalRateCritical": (
            unresolved_critical_rate
        ),

        "EthicalAlignmentOpportunity": (
            ethical_alignment_opportunity
        ),

        "SystemLevelEthicalOutcomeRate": (
            system_level_ethical_outcome
        ),

        "SharedWeight": SHARED_WEIGHT,
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_summary(summary):

    required_keys = {
        "N_tracked_events",
        "N_benign_passage",
        "N_mitigation_opportunities",
        "N_mitigated",
        "N_critical",
        "N_unresolved_critical",
        "N_vehicle_only",
        "N_shared",
        "N_vru_only",
        "N_proactive_vehicle_only",
        "N_proactive_shared",
        "N_proactive_vru_only",
        "N_reactive_vehicle_only",
        "N_reactive_shared",
        "N_reactive_vru_only",
        "BenignPassageRate",
        "MitigationOpportunityRate",
        "VehicleContributionRate",
        "StrictVRUPriorityRate",
        "SharedMitigationRate",
        "VRUOnlyBurdenRate",
        "ProactiveSystemMitigationRate",
        "UnresolvedCriticalRateAll",
        "UnresolvedCriticalRateCritical",
        "EthicalAlignmentOpportunity",
        "SystemLevelEthicalOutcomeRate",
        "SharedWeight",
    }

    missing_keys = (
        required_keys
        - set(summary.keys())
    )

    if missing_keys:

        raise KeyError(
            "Missing keys in ethics_summary_v2: "
            f"{sorted(missing_keys)}"
        )

    n_events = summary[
        "N_tracked_events"
    ]

    n_benign = summary[
        "N_benign_passage"
    ]

    n_unresolved = summary[
        "N_unresolved_critical"
    ]

    n_vehicle_only = summary[
        "N_vehicle_only"
    ]

    n_shared = summary[
        "N_shared"
    ]

    n_vru_only = summary[
        "N_vru_only"
    ]

    n_mitigated = summary[
        "N_mitigated"
    ]

    n_critical = summary[
        "N_critical"
    ]

    n_opportunities = summary[
        "N_mitigation_opportunities"
    ]

    # --------------------------------------------------------
    # Basic value validation
    # --------------------------------------------------------

    count_keys = [
        "N_tracked_events",
        "N_benign_passage",
        "N_mitigation_opportunities",
        "N_mitigated",
        "N_critical",
        "N_unresolved_critical",
        "N_vehicle_only",
        "N_shared",
        "N_vru_only",
        "N_proactive_vehicle_only",
        "N_proactive_shared",
        "N_proactive_vru_only",
        "N_reactive_vehicle_only",
        "N_reactive_shared",
        "N_reactive_vru_only",
    ]

    for key in count_keys:

        value = summary[key]

        if (
            not isinstance(value, int)
            or value < 0
        ):

            raise ValueError(
                f"Invalid count for {key}: "
                f"{value}"
            )

    # --------------------------------------------------------
    # Mitigated events
    # --------------------------------------------------------

    expected_mitigated = (
        n_vehicle_only
        + n_shared
        + n_vru_only
    )

    if n_mitigated != expected_mitigated:

        raise ValueError(
            "Invalid N_mitigated. "
            "Expected "
            f"{expected_mitigated}, "
            f"found {n_mitigated}."
        )

    # --------------------------------------------------------
    # Complete event partition
    # --------------------------------------------------------

    expected_total = (
        n_benign
        + n_unresolved
        + n_vehicle_only
        + n_shared
        + n_vru_only
    )

    if n_events != expected_total:

        raise ValueError(
            "Invalid event partition. "
            "N_tracked_events must equal "
            "N_benign_passage + "
            "N_unresolved_critical + "
            "N_vehicle_only + "
            "N_shared + "
            "N_vru_only. "
            f"Expected {expected_total}, "
            f"found {n_events}."
        )

    # --------------------------------------------------------
    # Timing partition
    # --------------------------------------------------------

    n_proactive = (
        summary[
            "N_proactive_vehicle_only"
        ]
        + summary[
            "N_proactive_shared"
        ]
        + summary[
            "N_proactive_vru_only"
        ]
    )

    n_reactive = (
        summary[
            "N_reactive_vehicle_only"
        ]
        + summary[
            "N_reactive_shared"
        ]
        + summary[
            "N_reactive_vru_only"
        ]
    )

    if (
        n_proactive
        + n_reactive
        != n_mitigated
    ):

        raise ValueError(
            "Invalid timing partition. "
            "Proactive and reactive events "
            "must sum to N_mitigated. "
            f"Expected {n_mitigated}, "
            f"found {n_proactive + n_reactive}."
        )

    # --------------------------------------------------------
    # Burden allocation partition
    # --------------------------------------------------------

    expected_vehicle_only = (
        summary[
            "N_proactive_vehicle_only"
        ]
        + summary[
            "N_reactive_vehicle_only"
        ]
    )

    expected_shared = (
        summary[
            "N_proactive_shared"
        ]
        + summary[
            "N_reactive_shared"
        ]
    )

    expected_vru_only = (
        summary[
            "N_proactive_vru_only"
        ]
        + summary[
            "N_reactive_vru_only"
        ]
    )

    if n_vehicle_only != expected_vehicle_only:

        raise ValueError(
            "Invalid N_vehicle_only. "
            f"Expected {expected_vehicle_only}, "
            f"found {n_vehicle_only}."
        )

    if n_shared != expected_shared:

        raise ValueError(
            "Invalid N_shared. "
            f"Expected {expected_shared}, "
            f"found {n_shared}."
        )

    if n_vru_only != expected_vru_only:

        raise ValueError(
            "Invalid N_vru_only. "
            f"Expected {expected_vru_only}, "
            f"found {n_vru_only}."
        )

    # --------------------------------------------------------
    # Critical events
    # --------------------------------------------------------

    expected_critical = (
        n_reactive
        + n_unresolved
    )

    if n_critical != expected_critical:

        raise ValueError(
            "Invalid N_critical. "
            "N_critical must equal "
            "reactive events plus "
            "unresolved critical events. "
            f"Expected {expected_critical}, "
            f"found {n_critical}."
        )

    # --------------------------------------------------------
    # Mitigation opportunities
    # --------------------------------------------------------

    expected_opportunities = (
        n_mitigated
        + n_unresolved
    )

    if n_opportunities != expected_opportunities:

        raise ValueError(
            "Invalid N_mitigation_opportunities. "
            "Expected "
            f"{expected_opportunities}, "
            f"found {n_opportunities}."
        )

    # --------------------------------------------------------
    # Metric helper
    # --------------------------------------------------------

    def check_metric(
        key,
        expected
    ):

        actual = summary[key]

        if expected is None:

            if actual is not None:

                raise ValueError(
                    f"Invalid {key}: expected None, "
                    f"found {actual}."
                )

            return

        if actual is None:

            raise ValueError(
                f"Invalid {key}: expected "
                f"{expected}, found None."
            )

        if abs(actual - expected) > 1e-12:

            raise ValueError(
                f"Invalid {key}: expected "
                f"{expected}, found {actual}."
            )

    # --------------------------------------------------------
    # Metrics based on all events
    # --------------------------------------------------------

    if n_events > 0:

        check_metric(
            "BenignPassageRate",
            n_benign / n_events
        )

        check_metric(
            "MitigationOpportunityRate",
            n_opportunities / n_events
        )

        check_metric(
            "UnresolvedCriticalRateAll",
            n_unresolved / n_events
        )

        expected_system_outcome = (
            n_benign
            + n_vehicle_only
            + summary["SharedWeight"]
            * n_shared
        ) / n_events

        check_metric(
            "SystemLevelEthicalOutcomeRate",
            expected_system_outcome
        )

    else:

        check_metric(
            "BenignPassageRate",
            None
        )

        check_metric(
            "MitigationOpportunityRate",
            None
        )

        check_metric(
            "UnresolvedCriticalRateAll",
            None
        )

        check_metric(
            "SystemLevelEthicalOutcomeRate",
            None
        )

    # --------------------------------------------------------
    # Metrics based on mitigated events
    # --------------------------------------------------------

    if n_mitigated > 0:

        check_metric(
            "VehicleContributionRate",
            (
                n_vehicle_only
                + n_shared
            ) / n_mitigated
        )

        check_metric(
            "StrictVRUPriorityRate",
            n_vehicle_only / n_mitigated
        )

        check_metric(
            "SharedMitigationRate",
            n_shared / n_mitigated
        )

        check_metric(
            "VRUOnlyBurdenRate",
            n_vru_only / n_mitigated
        )

        check_metric(
            "ProactiveSystemMitigationRate",
            (
                summary[
                    "N_proactive_vehicle_only"
                ]
                + summary[
                    "N_proactive_shared"
                ]
            ) / n_mitigated
        )

    else:

        check_metric(
            "VehicleContributionRate",
            None
        )

        check_metric(
            "StrictVRUPriorityRate",
            None
        )

        check_metric(
            "SharedMitigationRate",
            None
        )

        check_metric(
            "VRUOnlyBurdenRate",
            None
        )

        check_metric(
            "ProactiveSystemMitigationRate",
            None
        )

    # --------------------------------------------------------
    # Critical-event metric
    # --------------------------------------------------------

    if n_critical > 0:

        check_metric(
            "UnresolvedCriticalRateCritical",
            n_unresolved / n_critical
        )

    else:

        check_metric(
            "UnresolvedCriticalRateCritical",
            None
        )

    # --------------------------------------------------------
    # Opportunity-conditioned alignment
    # --------------------------------------------------------

    if n_opportunities > 0:

        expected_alignment = (
            n_vehicle_only
            + summary["SharedWeight"]
            * n_shared
        ) / n_opportunities

        check_metric(
            "EthicalAlignmentOpportunity",
            expected_alignment
        )

    else:

        check_metric(
            "EthicalAlignmentOpportunity",
            None
        )

    print(
        "Ethics V2 summary validation successful."
    )

    return True


# ============================================================
# PRINTING
# ============================================================

def format_metric(value):

    if value is None:
        return "N/A"

    return f"{value:.4f}"


def print_summary(
    scenario,
    summary
):

    print("\n================================================")
    print(f"ETHICS V2: {scenario}")
    print("================================================")

    print(
        "Tracked interaction events:  ",
        summary["N_tracked_events"],
        sep=""
    )

    print(
        "Benign passages:             ",
        summary["N_benign_passage"],
        sep=""
    )

    print(
        "Mitigation opportunities:    ",
        summary["N_mitigation_opportunities"],
        sep=""
    )

    print(
        "Mitigated events:            ",
        summary["N_mitigated"],
        sep=""
    )

    print(
        "Critical events:             ",
        summary["N_critical"],
        sep=""
    )

    print(
        "Unresolved critical events:  ",
        summary["N_unresolved_critical"],
        sep=""
    )

    print("\n[BURDEN ALLOCATION]")

    print(
        "Vehicle-only mitigation:     ",
        summary["N_vehicle_only"],
        sep=""
    )

    print(
        "Shared mitigation:           ",
        summary["N_shared"],
        sep=""
    )

    print(
        "VRU-only mitigation:         ",
        summary["N_vru_only"],
        sep=""
    )

    print("\n[PROACTIVE MITIGATION]")

    print(
        "Proactive vehicle-only:      ",
        summary["N_proactive_vehicle_only"],
        sep=""
    )

    print(
        "Proactive shared:            ",
        summary["N_proactive_shared"],
        sep=""
    )

    print(
        "Proactive VRU-only:          ",
        summary["N_proactive_vru_only"],
        sep=""
    )

    print("\n[REACTIVE MITIGATION]")

    print(
        "Reactive vehicle-only:       ",
        summary["N_reactive_vehicle_only"],
        sep=""
    )

    print(
        "Reactive shared:             ",
        summary["N_reactive_shared"],
        sep=""
    )

    print(
        "Reactive VRU-only:           ",
        summary["N_reactive_vru_only"],
        sep=""
    )

    print("\n[PRIMARY RATES]")

    print(
        "Benign Passage Rate:         ",
        format_metric(
            summary["BenignPassageRate"]
        ),
        sep=""
    )

    print(
        "Mitigation Opportunity Rate: ",
        format_metric(
            summary["MitigationOpportunityRate"]
        ),
        sep=""
    )

    print(
        "Vehicle Contribution Rate:   ",
        format_metric(
            summary["VehicleContributionRate"]
        ),
        sep=""
    )

    print(
        "Strict VRU Priority Rate:    ",
        format_metric(
            summary["StrictVRUPriorityRate"]
        ),
        sep=""
    )

    print(
        "Shared Mitigation Rate:      ",
        format_metric(
            summary["SharedMitigationRate"]
        ),
        sep=""
    )

    print(
        "VRU-only Burden Rate:        ",
        format_metric(
            summary["VRUOnlyBurdenRate"]
        ),
        sep=""
    )

    print(
        "Proactive System Rate:       ",
        format_metric(
            summary[
                "ProactiveSystemMitigationRate"
            ]
        ),
        sep=""
    )

    print(
        "Unresolved Critical, all:    ",
        format_metric(
            summary[
                "UnresolvedCriticalRateAll"
            ]
        ),
        sep=""
    )

    print(
        "Unresolved Critical, crit.:  ",
        format_metric(
            summary[
                "UnresolvedCriticalRateCritical"
            ]
        ),
        sep=""
    )

    print("\n[AGGREGATED SECONDARY METRICS]")

    print(
        "Ethical Alignment, opp.:     ",
        format_metric(
            summary[
                "EthicalAlignmentOpportunity"
            ]
        ),
        sep=""
    )

    print(
        "System Ethical Outcome:      ",
        format_metric(
            summary[
                "SystemLevelEthicalOutcomeRate"
            ]
        ),
        sep=""
    )

    print(
        "Shared mitigation weight:    ",
        f"{summary['SharedWeight']:.2f}",
        sep=""
    )

# ============================================================
# SCENARIO PROCESSING
# ============================================================

def analyze_scenario(scenario):

    scenario_folder = os.path.join(
        RESULT_ROOT,
        scenario
    )

    input_path = os.path.join(
        scenario_folder,
        INPUT_FILENAME
    )

    output_events_path = os.path.join(
        scenario_folder,
        OUTPUT_EVENTS_FILENAME
    )

    output_summary_path = os.path.join(
        scenario_folder,
        OUTPUT_SUMMARY_FILENAME
    )

    if not os.path.isfile(input_path):

        raise FileNotFoundError(
            f"Missing ethics event file:\n"
            f"{input_path}"
        )

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        original_events = json.load(file)

        enriched_events = []

        for event_index, event in enumerate(
            original_events
        ):
    
            validate_input_event(
                event=event,
                event_index=event_index,
                scenario=scenario
            )
    
            enriched_events.append(
                classify_event_v2(event)
            )

    summary = calculate_summary(
        enriched_events
    )

    validate_summary(
        summary
    )

    with open(
        output_events_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            enriched_events,
            file,
            indent=4,
            ensure_ascii=False
        )

    with open(
        output_summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    print_summary(
        scenario,
        summary
    )

    print(
        f"\nEvents written to:\n"
        f"{output_events_path}"
    )

    print(
        f"\nSummary written to:\n"
        f"{output_summary_path}"
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    all_summaries = {}

    for scenario in SCENARIOS:

        all_summaries[scenario] = (
            analyze_scenario(
                scenario
            )
        )

    comparison_path = os.path.join(
        RESULT_ROOT,
        "ethics_comparison_v2.json"
    )

    with open(
        comparison_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_summaries,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n================================================")
    print("ETHICS V2 ANALYSIS COMPLETE")
    print("================================================")

    print(
        f"Combined comparison written to:\n"
        f"{comparison_path}"
    )


if __name__ == "__main__":

    main()