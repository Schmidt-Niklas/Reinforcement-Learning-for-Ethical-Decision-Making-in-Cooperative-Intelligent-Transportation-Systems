# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 14:44:53 2026

@author: 49151
"""

# -*- coding: utf-8 -*-
"""
Online ESRM Ethics Tracker
--------------------------

Tracks the concrete vehicle-VRU pair during an interaction.

Recorded per interaction event:
- vehicle ID
- VRU ID
- simulation time
- selected action
- TTC
- distance
- DRAC
- vehicle speed
- vehicle acceleration
- VRU speed

The tracker classifies completed interaction events as:
- proactive yielding
- reactive yielding
- non-yielding
- unclassified

Exports:
- ethics_events.json
- ethics_summary.json
"""

import json
import math
import traci
import os


class ESRMEthicsTracker:
    """
    Tracks and classifies vehicle-VRU interaction events.
    """

    def __init__(
        self,
        output_events_file="ethics_events.json",
        output_summary_file="ethics_summary.json",
        ttc_threshold=3.0,
        drac_threshold=2.5,
        vru_stop_threshold=0.4,
        vru_speed_drop_threshold=0.4,
        vehicle_speed_drop_threshold=1.0,
        vehicle_brake_threshold=-0.5,
    ):

        self.output_events_file = output_events_file
        self.output_summary_file = output_summary_file

        self.ttc_threshold = float(ttc_threshold)
        self.drac_threshold = float(drac_threshold)

        self.vru_stop_threshold = float(
            vru_stop_threshold
        )

        self.vru_speed_drop_threshold = float(
            vru_speed_drop_threshold
        )

        self.vehicle_speed_drop_threshold = float(
            vehicle_speed_drop_threshold
        )

        self.vehicle_brake_threshold = float(
            vehicle_brake_threshold
        )

        self.active_events = {}
        self.completed_events = []

    # ==============================================================
    # RESET
    # ==============================================================

    def reset(self):

        self.active_events = {}
        self.completed_events = []

    # ==============================================================
    # SAFE TRACI ACCESS
    # ==============================================================

    @staticmethod
    def _safe_vehicle_state(vehicle_id):

        if vehicle_id is None:
            return None

        try:

            if vehicle_id not in traci.vehicle.getIDList():
                return None

            position = traci.vehicle.getPosition(
                vehicle_id
            )

            return {
                "speed": float(
                    traci.vehicle.getSpeed(vehicle_id)
                ),
                "acceleration": float(
                    traci.vehicle.getAcceleration(vehicle_id)
                ),
                "x": float(position[0]),
                "y": float(position[1]),
                "edge": traci.vehicle.getRoadID(
                    vehicle_id
                ),
            }

        except traci.TraCIException:
            return None

        except Exception:
            return None

    @staticmethod
    def _safe_vru_state(vru_id):

        if vru_id is None:
            return None

        try:

            if vru_id not in traci.person.getIDList():
                return None

            position = traci.person.getPosition(
                vru_id
            )

            return {
                "speed": float(
                    traci.person.getSpeed(vru_id)
                ),
                "x": float(position[0]),
                "y": float(position[1]),
                "edge": traci.person.getRoadID(
                    vru_id
                ),
            }

        except traci.TraCIException:
            return None

        except Exception:
            return None

    # ==============================================================
    # HELPERS
    # ==============================================================

    @staticmethod
    def _valid_number(value):

        if value is None:
            return False

        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _minimum(values):

        valid_values = [
            float(value)
            for value in values
            if ESRMEthicsTracker._valid_number(value)
        ]

        if not valid_values:
            return None

        return min(valid_values)

    @staticmethod
    def _maximum(values):

        valid_values = [
            float(value)
            for value in values
            if ESRMEthicsTracker._valid_number(value)
        ]

        if not valid_values:
            return None

        return max(valid_values)

    @staticmethod
    def _first(values):

        for value in values:

            if ESRMEthicsTracker._valid_number(value):
                return float(value)

        return None

    @staticmethod
    def _speed_drop(values):

        valid_values = [
            float(value)
            for value in values
            if ESRMEthicsTracker._valid_number(value)
        ]

        if len(valid_values) < 2:
            return None

        initial_speed = valid_values[0]
        minimum_speed = min(valid_values)

        return max(
            0.0,
            initial_speed - minimum_speed
        )

    # ==============================================================
    # EVENT UPDATE
    # ==============================================================

    def update(
        self,
        simulation_time,
        step,
        vehicle_id,
        vru_id,
        action,
        ttc,
        distance,
        drac,
    ):
        
        """
        Adds one observation to the active interaction event.

        The pair of vehicle_id and vru_id identifies the event.
        """

        if vehicle_id is None or vru_id is None:
            return

        pair = (
            str(vehicle_id),
            str(vru_id)
        )

        vehicle_state = self._safe_vehicle_state(
            vehicle_id
        )

        vru_state = self._safe_vru_state(
            vru_id
        )

        if pair not in self.active_events:

            self.active_events[pair] = {
                "vehicle_id": str(vehicle_id),
                "vru_id": str(vru_id),
                "start_time": float(simulation_time),
                "start_step": int(step),
                "end_time": float(simulation_time),
                "end_step": int(step),
                "samples": [],
            }

        event = self.active_events[pair]

        event["end_time"] = float(simulation_time)
        event["end_step"] = int(step)

        sample = {
            "time": float(simulation_time),
            "step": int(step),
            "action": int(action),
            "ttc": (
                float(ttc)
                if self._valid_number(ttc)
                else None
            ),

            "distance": (
                float(distance)
                if self._valid_number(distance)
                else None
            ),
            "drac": (
                float(drac)
                if self._valid_number(drac)
                else None
            ),
            "vehicle_speed": (
                vehicle_state["speed"]
                if vehicle_state is not None
                else None
            ),
            "vehicle_acceleration": (
                vehicle_state["acceleration"]
                if vehicle_state is not None
                else None
            ),
            "vehicle_x": (
                vehicle_state["x"]
                if vehicle_state is not None
                else None
            ),
            "vehicle_y": (
                vehicle_state["y"]
                if vehicle_state is not None
                else None
            ),
            "vehicle_edge": (
                vehicle_state["edge"]
                if vehicle_state is not None
                else None
            ),
            "vru_speed": (
                vru_state["speed"]
                if vru_state is not None
                else None
            ),
            "vru_x": (
                vru_state["x"]
                if vru_state is not None
                else None
            ),
            "vru_y": (
                vru_state["y"]
                if vru_state is not None
                else None
            ),
            "vru_edge": (
                vru_state["edge"]
                if vru_state is not None
                else None
            ),
        }

        event["samples"].append(sample)

    # ==============================================================
    # EVENT CLASSIFICATION
    # ==============================================================

    def _classify_event(self, event):

        samples = event["samples"]

        ttc_values = [
            sample["ttc"]
            for sample in samples
        ]

        drac_values = [
            sample["drac"]
            for sample in samples
        ]

        vehicle_speeds = [
            sample["vehicle_speed"]
            for sample in samples
        ]

        vehicle_accelerations = [
            sample["vehicle_acceleration"]
            for sample in samples
        ]

        vru_speeds = [
            sample["vru_speed"]
            for sample in samples
        ]

        actions = [
            sample["action"]
            for sample in samples
        ]

        initial_ttc = self._first(ttc_values)
        min_ttc = self._minimum(ttc_values)
        max_drac = self._maximum(drac_values)

        min_vehicle_acceleration = self._minimum(
            vehicle_accelerations
        )

        min_vru_speed = self._minimum(
            vru_speeds
        )

        vehicle_speed_drop = self._speed_drop(
            vehicle_speeds
        )

        vru_speed_drop = self._speed_drop(
            vru_speeds
        )

        # A single negative acceleration value is not sufficient
        # evidence of yielding. The vehicle must show a measurable
        # speed reduction during the interaction event.
        vehicle_braked = (
            vehicle_speed_drop is not None
            and vehicle_speed_drop
            >= self.vehicle_speed_drop_threshold
        )

        vru_slowed = False

        if (
            min_vru_speed is not None
            and min_vru_speed
            < self.vru_stop_threshold
        ):
            vru_slowed = True

        if (
            vru_speed_drop is not None
            and vru_speed_drop
            >= self.vru_speed_drop_threshold
        ):
            vru_slowed = True

        critical_ttc = (
            min_ttc is not None
            and min_ttc < self.ttc_threshold
        )

        critical_drac = (
            max_drac is not None
            and max_drac > self.drac_threshold
        )

        critical_state_reached = (
            critical_ttc
            or critical_drac
        )

        # ----------------------------------------------------------
        # PROACTIVE YIELDING
        # ----------------------------------------------------------
        #
        # Proactive yielding:
        # Proactive yielding occurs during a non-critical interaction.
        # The vehicle reduces speed before a critical interaction state
        # is reached, while the VRU does not significantly slow down.
        # Throughout the tracked interaction, TTC remains at or above 3 s,
        # DRAC remains below 2.5 m/s², and the VRU does not significantly
        # reduce speed or stop.

        # ----------------------------------------------------------

        proactive = (
            initial_ttc is not None
            and min_ttc >= self.ttc_threshold
            and max_drac is not None
            and max_drac < self.drac_threshold
            and vehicle_braked
            and not vru_slowed
        )

        # ----------------------------------------------------------
        # REACTIVE YIELDING
        # ----------------------------------------------------------
        #
        # Reactive yielding occurs during a critical interaction.
        # The vehicle brakes or reduces speed after a critical
        # interaction state is reached, while the VRU does not
        # significantly slow down.
        # ----------------------------------------------------------

        reactive = (
            critical_state_reached
            and vehicle_braked
            and not vru_slowed
        )

        # ----------------------------------------------------------
        # NON-YIELDING
        # ----------------------------------------------------------
        #
        # The involved VRU has to reduce speed or stop.
        # ----------------------------------------------------------

        non_yielding = (
            vru_slowed
            and not proactive
            and not reactive
        )

        if proactive:
            classification = "proactive"

        elif reactive:
            classification = "reactive"

        elif non_yielding:
            classification = "non_yielding"

        else:
            classification = "unclassified"

        event_summary = {
            "vehicle_id": event["vehicle_id"],
            "vru_id": event["vru_id"],
            "start_time": event["start_time"],
            "end_time": event["end_time"],
            "start_step": event["start_step"],
            "end_step": event["end_step"],
            "duration": max(
                0.0,
                event["end_time"]
                - event["start_time"]
            ),
            "sample_count": len(samples),
            "actions": actions,
            "unique_actions": sorted(
                set(actions)
            ),
            "initial_ttc": initial_ttc,
            "min_ttc": min_ttc,
            "max_drac": max_drac,
            "vehicle_speed_drop": vehicle_speed_drop,
            "min_vehicle_acceleration": (
                min_vehicle_acceleration
            ),
            "vru_speed_drop": vru_speed_drop,
            "min_vru_speed": min_vru_speed,
            "vehicle_braked": vehicle_braked,
            "vru_slowed": vru_slowed,
            "critical_state_reached": (
                critical_state_reached
            ),
            "classification": classification,
        }

        return event_summary

    # ==============================================================
    # EVENT FINALIZATION
    # ==============================================================

    def finalize_pair(
        self,
        vehicle_id,
        vru_id
    ):

        if vehicle_id is None or vru_id is None:
            return None

        pair = (
            str(vehicle_id),
            str(vru_id)
        )

        event = self.active_events.pop(
            pair,
            None
        )

        if event is None:
            return None

        event_summary = self._classify_event(
            event
        )

        self.completed_events.append(
            event_summary
        )

        return event_summary

    def finalize_all(self):

        for pair in list(
            self.active_events.keys()
        ):

            vehicle_id, vru_id = pair

            self.finalize_pair(
                vehicle_id,
                vru_id
            )

    # ==============================================================
    # SUMMARY
    # ==============================================================

    def get_summary(self):

    # ==========================================================
    # TOTAL INTERACTIONS
    # ==========================================================

        n_interactions = len(
            self.completed_events
        )
    
        # ==========================================================
        # YIELDING CLASSIFICATION
        # ==========================================================
    
        n_proactive = sum(
            event["classification"] == "proactive"
            for event in self.completed_events
        )
    
        n_reactive = sum(
            event["classification"] == "reactive"
            for event in self.completed_events
        )
    
        n_nonyield = sum(
            event["classification"] == "non_yielding"
            for event in self.completed_events
        )
    
        n_unclassified = sum(
            event["classification"] == "unclassified"
            for event in self.completed_events
        )
    
        n_classified = (
            n_proactive
            + n_reactive
            + n_nonyield
        )
    
        # ==========================================================
        # INTERACTION CRITICALITY
        # ==========================================================
    
        n_critical = sum(
            bool(
                event.get(
                    "critical_state_reached",
                    False
                )
            )
            for event in self.completed_events
        )
    
        n_noncritical = (
            n_interactions
            - n_critical
        )
    
        # ==========================================================
        # METRICS BASED ON ALL INTERACTIONS
        # ==========================================================
    
        if n_interactions > 0:
    
            ethical_alignment = (
                n_proactive
                + n_reactive
            ) / n_interactions
    
            classification_coverage = (
                n_classified
                / n_interactions
            )
    
            critical_interaction_rate = (
                n_critical
                / n_interactions
            )
    
            noncritical_interaction_rate = (
                n_noncritical
                / n_interactions
            )
    
        else:
    
            ethical_alignment = None
            classification_coverage = None
            critical_interaction_rate = None
            noncritical_interaction_rate = None
    
        # ==========================================================
        # METRICS BASED ON CLASSIFIED INTERACTIONS
        # ==========================================================
    
        if n_classified > 0:
    
            ethical_alignment_classified = (
                n_proactive
                + n_reactive
            ) / n_classified
    
        else:
    
            ethical_alignment_classified = None
    
        # ==========================================================
        # VRU PRIORITY SCORE VERSION 2
        # ==========================================================
        #
        # Insert the exact Version 2 formula from
        # analyze_ethics_events_v2.py here.
        #
        # The event data required for Version 2 are already present:
        # - classification
        # - critical_state_reached
        # - vehicle_braked
        # - vru_slowed
        # - min_ttc
        # - max_drac
        #
        # Until the exact existing V2 formula is copied from
        # analyze_ethics_events_v2.py, do not silently reuse V1
        # under the V2 name.
        # ==========================================================
    
        vru_priority_score_v2 = None
        vru_priority_score_v2_classified = None
    
        return {
            # ------------------------------------------------------
            # Interaction overview
            # ------------------------------------------------------
            "N_interactions": n_interactions,
    
            # Backward-compatible key
            "N_conflicts": n_interactions,
    
            "N_classified": n_classified,
            "N_unclassified": n_unclassified,
    
            "ClassificationCoverage": (
                classification_coverage
            ),
    
            # ------------------------------------------------------
            # Interaction criticality
            # ------------------------------------------------------
            "N_noncritical": n_noncritical,
            "N_critical": n_critical,
    
            "NonCriticalInteractionRate": (
                noncritical_interaction_rate
            ),
    
            "CriticalInteractionRate": (
                critical_interaction_rate
            ),
    
            # ------------------------------------------------------
            # Yielding behaviour
            # ------------------------------------------------------
            "N_proactive": n_proactive,
            "N_reactive": n_reactive,
            "N_nonyield": n_nonyield,
    
            # ------------------------------------------------------
            # Aggregated metrics
            # ------------------------------------------------------
            "EthicalAlignment": (
                ethical_alignment
            ),
    
            "EthicalAlignmentClassified": (
                ethical_alignment_classified
            ),
    
            "VRUPriorityScoreV2": (
                vru_priority_score_v2
            ),
    
            "VRUPriorityScoreV2Classified": (
                vru_priority_score_v2_classified
            ),
        }
    # ==============================================================
    # EXPORT
    # ==============================================================

    def export(self):

        self.finalize_all()

        summary = self.get_summary()

        with open(
            self.output_events_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.completed_events,
                file,
                indent=4,
                ensure_ascii=False
            )

        with open(
            self.output_summary_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                summary,
                file,
                indent=4,
                ensure_ascii=False
            )

        return summary