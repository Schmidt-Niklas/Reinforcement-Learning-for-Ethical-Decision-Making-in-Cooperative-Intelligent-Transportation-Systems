# -*- coding: utf-8 -*-
"""
ESRM Evaluation Pipeline
------------------------

Executes the complete ESRM evaluation workflow:

1) Throughput
2) Safety
3) Fairness
4) Ethics
5) Dashboard generation

Console Output:
- Dashboard summary only
- No duplicate metric reporting

Output:
- ESRM dashboard
- esrm_dashboard.json

"""

import traceback

import throughput_output
import safety_output_final
import fairness_waiting_output
import ethics_output_final_V2
import dashboard


def run_pipeline():

    print(
        "\n==================== ESRM PIPELINE START ====================\n"
    )

    try:

        T = throughput_output.compute_throughput()

    except Exception:

        print("❌ ERROR in Throughput:")
        traceback.print_exc()
        return

    try:

        S = safety_output_final.compute_safety()

    except Exception:

        print("❌ ERROR in Safety:")
        traceback.print_exc()
        return

    try:

        F = fairness_waiting_output.compute_fairness()

    except Exception:

        print("❌ ERROR in Fairness:")
        traceback.print_exc()
        return

    try:

        E = ethics_output_final_V2.compute_ethics()

    except Exception:

        print("❌ ERROR in Ethics:")
        traceback.print_exc()
        return

    try:

        dashboard.build_esrm_dashboard(
            export_json=True
        )

    except Exception:

        print("❌ ERROR in Dashboard:")
        traceback.print_exc()
        return

    print(
        "\n==================== ESRM PIPELINE COMPLETE ✓ ====================\n"
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    run_pipeline()