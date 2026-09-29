# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:46:44 2026

@author: 49151
"""

# -*- coding: utf-8 -*-
"""
run_outcome_evaluation.py

Master Thesis Outcome Evaluation

Runs:

1. Baseline
2. U1
3. U2

For each scenario:

- Run full SUMO simulation
- Save XML outputs
- Run ESRM evaluation pipeline
- Save esrm_dashboard.json

Author:
Niklas Schmidt
"""

import os
import numpy as np
import shutil
from pathlib import Path
import traci
import sumolib
import time
from stable_baselines3 import PPO
from esrm_env.env_V2 import ESRMEnv
#from esrm_env.env_u2b import ESRMEnv as ESRMEnvU2
from esrm_env.reward import ESRMRewardU1
import sys
import torch as th
import json

PROJECT_PATH = (
    r"C:\Users\49151\Desktop\Master\07_Masterthesis"
    r"\00_InTas_New - RL\Pyscripts\esrm_env"
)

if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)
    
from esrm_env.reward_ethics import ESRMRewardEthics


# ============================================================
# CONFIG
# ============================================================

SUMOCFG = (
    r"C:\Users\49151\Desktop\Master\07_Masterthesis"
    r"\00_InTAS_New - RL\02_Midday\InTAS_2.0.RL.sumocfg"
)

#U1_MODEL = (
#    r"C:\Users\49151\Desktop\Master\07_Masterthesis"
#    r"\00_InTas_New - RL\Pyscripts\ppo_esrm_v3_seed_3.zip"
#)

U1_MODEL = (
    r"C:\Users\49151\Desktop\Master\07_Masterthesis"
    r"\00_InTas_New - RL\Pyscripts\models\U1\seed_3\ppo_esrm_u1_500k_seed_3.zip"
)
    
#U2_MODEL = (
#    r"C:\Users\49151\Desktop\Master\07_Masterthesis"
#    r"\00_InTAS_New - RL\Pyscripts\ppo_esrm_ethics_seed_3.zip"
#)

#U2_MODEL = (
#    r"C:\Users\49151\Desktop\Master\07_Masterthesis"
#    r"\00_InTas_New - RL\Pyscripts\models\U2\seed_2\ppo_esrm_u2_500k_seed_2.zip"
#)

U2_MODEL = (
    r"C:\Users\49151\Desktop\Master\07_Masterthesis\00_InTas_New - RL\Pyscripts\models\U2\seed_2\ppo_esrm_u2_ethics_500k_seed_2.zip"
)

SUMO_OUTPUT_FOLDER = (
    r"C:\Users\49151\Desktop\Master\07_Masterthesis"
    r"\00_InTAS_New - RL\02_Midday"
)

RESULT_ROOT = (
    r"C:\Users\49151\Desktop\Master\07_Masterthesis"
    r"\04_Outcome_Evaluation\Midday"
)

XML_FILES = [
    "ssm_output.xml",
    "collisions.xml",
    "InTAS.simulation.tripinfo.xml",
    "InTAS.simulation.personinfo.xml",
    "InTAS.simulation.vehroute.xml",
    "InTAS.simulation.fcd.persons.xml",
    "InTAS.simulation.statistic.xml",
]

ETHICS_FILES = [
    "ethics_events.json",
    "ethics_summary.json",
]
    

JUNCTION_POLYGON = [
    (215275.72, 450927.73), (215284.45, 450922.08),
    (215284.04, 450921.03), (215284.07, 450920.57),
    (215284.26, 450920.14), (215284.60, 450919.75),
    (215285.10, 450919.40), (215279.90, 450910.39),
    (215277.88, 450910.67), (215276.72, 450910.22),
    (215275.47, 450909.38), (215274.13, 450908.16),
    (215272.69, 450906.54), (215264.65, 450913.15),
    (215265.18, 450914.14), (215265.21, 450914.61),
    (215265.08, 450915.06), (215264.78, 450915.49),
    (215264.33, 450915.90), (215270.63, 450924.18),
    (215272.03, 450923.94), (215272.84, 450924.35),
    (215273.73, 450925.12), (215274.69, 450926.25),
]

VRU_INNER_ALL = {
    "-201953160#1",
    "-201953160#4",
    "-208010459#1",
    "-40799279",
    "201953160#0",
    "201953160#3",
    "208010459#1",
    "40799279",
}

# ============================================================
# FOLDERS
# ============================================================

def create_folder_structure():

    for scenario in [
        "Baseline",
        "U1",
        "U2"
    ]:

        os.makedirs(
            os.path.join(
                RESULT_ROOT,
                scenario
            ),
            exist_ok=True
        )

# ============================================================
# COPY XML OUTPUTS
# ============================================================

def copy_outputs(target_folder):

    os.makedirs(
        target_folder,
        exist_ok=True
    )

    missing_files = []

    for fname in XML_FILES:

        src = os.path.join(
            SUMO_OUTPUT_FOLDER,
            fname
        )

        if not os.path.isfile(src):

            missing_files.append(fname)
            continue

        dst = os.path.join(
            target_folder,
            fname
        )

        shutil.copy2(
            src,
            dst
        )

        print(
            f"Copied SUMO output:\n"
            f"{src}\n"
            f"-> {dst}"
        )

    if missing_files:

        raise FileNotFoundError(
            "Missing SUMO output files: "
            f"{missing_files}"
        )


# ============================================================
# ETHICS OUTPUT MANAGEMENT
# ============================================================

def remove_old_ethics_outputs():

    for fname in ETHICS_FILES:

        path = os.path.join(
            os.getcwd(),
            fname
        )

        if os.path.exists(path):

            os.remove(path)

            print(
                f"Removed old ethics output: "
                f"{path}"
            )

def remove_old_sumo_outputs():

    for fname in XML_FILES:

        path = os.path.join(
            SUMO_OUTPUT_FOLDER,
            fname
        )

        if os.path.isfile(path):

            os.remove(path)

            print(
                f"Removed old SUMO output: "
                f"{path}"
            )


def copy_ethics_outputs(target_folder):

    os.makedirs(
        target_folder,
        exist_ok=True
    )

    for fname in ETHICS_FILES:

        src = os.path.join(
            os.getcwd(),
            fname
        )

        if not os.path.exists(src):

            raise FileNotFoundError(
                f"Missing ethics output: {src}"
            )

        dst = os.path.join(
            target_folder,
            fname
        )

        shutil.copy2(
            src,
            dst
        )

        print(
            f"Copied ethics output:\n"
            f"{src}\n"
            f"-> {dst}"
        )

def validate_ethics_summary(target_folder):

    summary_path = os.path.join(
        target_folder,
        "ethics_summary.json"
    )

    if not os.path.isfile(summary_path):

        raise FileNotFoundError(
            f"Ethics summary not found:\n"
            f"{summary_path}"
        )

    with open(
        summary_path,
        "r",
        encoding="utf-8"
    ) as file:

        summary = json.load(file)

    required_keys = {
        "N_conflicts",
        "N_classified",
        "N_proactive",
        "N_reactive",
        "N_nonyield",
        "N_unclassified",
        "EthicalAlignment",
        "VRUPriorityScore",
        "EthicalAlignmentClassified",
        "VRUPriorityScoreClassified",
        "ClassificationCoverage",
    }

    missing_keys = (
        required_keys
        - set(summary.keys())
    )

    if missing_keys:

        raise KeyError(
            "Missing keys in ethics_summary.json: "
            f"{sorted(missing_keys)}"
        )

    expected_classified = (
        summary["N_proactive"]
        + summary["N_reactive"]
        + summary["N_nonyield"]
    )

    expected_conflicts = (
        summary["N_classified"]
        + summary["N_unclassified"]
    )

    if (
        summary["N_classified"]
        != expected_classified
    ):

        raise ValueError(
            "Invalid ethics summary: "
            "N_classified must equal "
            "N_proactive + N_reactive + N_nonyield."
        )

    if (
        summary["N_conflicts"]
        != expected_conflicts
    ):

        raise ValueError(
            "Invalid ethics summary: "
            "N_conflicts must equal "
            "N_classified + N_unclassified."
        )

    n_conflicts = summary["N_conflicts"]
    n_classified = summary["N_classified"]

    if n_conflicts > 0:

        expected_coverage = (
            n_classified / n_conflicts
        )

        actual_coverage = summary[
            "ClassificationCoverage"
        ]

        if actual_coverage is None:

            raise ValueError(
                "ClassificationCoverage must not be "
                "None when N_conflicts is greater than zero."
            )

        if not np.isclose(
            actual_coverage,
            expected_coverage,
            rtol=1e-9,
            atol=1e-12
        ):

            raise ValueError(
                "Invalid ClassificationCoverage. "
                f"Expected {expected_coverage}, "
                f"found {actual_coverage}."
            )

    print(
        "\nEthics summary validated successfully."
    )

    print(
        json.dumps(
            summary,
            indent=4,
            ensure_ascii=False
        )
    )

    return summary

        
# ============================================================
# BASELINE
# ============================================================

def run_baseline():

    print("\n================================================")
    print("RUNNING BASELINE")
    print("================================================")

    if not os.path.isfile(SUMOCFG):
        raise FileNotFoundError(
            f"SUMO configuration not found:\n{SUMOCFG}"
        )

    env = ESRMEnv(
        sumocfg_path=SUMOCFG,
        junction_polygon=JUNCTION_POLYGON,
        episode_seconds=7200.0,        
        step_length=0.1,
        vru_inner_edges=VRU_INNER_ALL,
    )

    obs = env.reset()

    done = False
    step_counter = 0
    interaction_counter = 0

    start_time = time.time()

    try:

        while not done:

            interaction_active = bool(obs[0])

            if interaction_active:
                interaction_counter += 1

            # Baseline: no ESRM intervention
            action = 0

            obs, reward, done, info = env.step(
                action
            )

            step_counter += 1

            if step_counter % 10000 == 0:

                progress = (
                    100.0
                    * step_counter
                    / env.horizon
                )

                try:
                    sim_time = (
                        env.traci_conn
                        .simulation
                        .getTime()
                    )
                except Exception:
                    sim_time = -1.0

                print(
                    f"Step={step_counter} | "
                    f"Progress={progress:.1f}% | "
                    f"SUMO Time={sim_time:.1f}"
                )

    finally:

        env.close()

    if not done or step_counter < env.horizon:

        raise RuntimeError(
            "Baseline simulation did not reach the full horizon. "
            f"Completed {step_counter} of "
            f"{env.horizon} steps."
        )

    elapsed = time.time() - start_time

    interaction_rate = (
        100.0 * interaction_counter / step_counter
        if step_counter > 0
        else 0.0
    )

    print(
        f"\nInteractions: {interaction_counter}"
    )

    print(
        f"Interaction rate: "
        f"{interaction_rate:.2f}%"
    )

    print(
        f"\nBaseline simulation completed "
        f"in {elapsed:.1f} seconds."
    )

    return True

# ============================================================
# U1 
# ============================================================

from gym import spaces
from stable_baselines3.common.save_util import load_from_zip_file


def load_repaired_ppo(model_path, env):

    data, params, pytorch_variables = load_from_zip_file(
        model_path,
        device="cpu"
    )

    data["action_space"] = spaces.Discrete(5)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=data["policy_kwargs"],
    )

    model.policy.load_state_dict(
        params["policy"]
    )

    return model

# ============================================================
# U1
# ============================================================

def run_u1():

    print("\n================================================")
    print("RUNNING U1")
    print("================================================")

    if not os.path.isfile(U1_MODEL):
        raise FileNotFoundError(
            f"U1 model not found:\n{U1_MODEL}"
        )

    if not os.path.isfile(SUMOCFG):
        raise FileNotFoundError(
            f"SUMO configuration not found:\n{SUMOCFG}"
        )

    print(f"Loading U1 model:\n{U1_MODEL}")

    env = ESRMEnv(
        sumocfg_path=SUMOCFG,
        junction_polygon=JUNCTION_POLYGON,
        episode_seconds=7200.0,
        step_length=0.1,
        vru_inner_edges=VRU_INNER_ALL,
    )

    model = load_repaired_ppo(
        U1_MODEL,
        env
    )

    obs = env.reset()

    print("U1 rollout started...")
    print("Simulation horizon:", env.horizon)

    done = False
    step_counter = 0
    interaction_counter = 0

    action_counter = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }

    start_time = time.time()

    try:

        while not done:

            interaction_active = bool(obs[0])

            if interaction_active:

                interaction_counter += 1

                action, _ = model.predict(
                    obs,
                    deterministic=False
                )

                action = int(action)

            else:

                action = 0

            action_counter[action] += 1

            obs, reward, done, info = env.step(
                action
            )

            step_counter += 1

            if step_counter % 10000 == 0:

                progress = (
                    100.0
                    * step_counter
                    / env.horizon
                )

                try:
                    sim_time = (
                        env.traci_conn
                        .simulation
                        .getTime()
                    )
                except Exception:
                    sim_time = -1.0

                print(
                    f"Step={step_counter} | "
                    f"Progress={progress:.1f}% | "
                    f"SUMO Time={sim_time:.1f}"
                )

    finally:

        env.close()

    if not done or step_counter < env.horizon:

        raise RuntimeError(
            "U1 simulation did not reach the full horizon. "
            f"Completed {step_counter} of "
            f"{env.horizon} steps."
        )

    elapsed = time.time() - start_time

    print("\nAction distribution:")

    for action_id in sorted(action_counter):

        print(
            f"A{action_id}: "
            f"{action_counter[action_id]}"
        )

    interaction_rate = (
        100.0 * interaction_counter / step_counter
        if step_counter > 0
        else 0.0
    )

    print(
        f"\nInteractions: {interaction_counter}"
    )

    print(
        f"Interaction rate: "
        f"{interaction_rate:.2f}%"
    )

    print(
        f"\nU1 simulation completed "
        f"in {elapsed:.1f} seconds."
    )

    return True
    
# ============================================================
# U2 PLACEHOLDER
# ============================================================

def run_u2():

    print("\n================================================")
    print("RUNNING U2")
    print("================================================")

    if not os.path.isfile(U2_MODEL):
        raise FileNotFoundError(
            f"U2 model not found:\n{U2_MODEL}"
        )

    if not os.path.isfile(SUMOCFG):
        raise FileNotFoundError(
            f"SUMO configuration not found:\n{SUMOCFG}"
        )

    print(f"Loading U2 model:\n{U2_MODEL}")

    env = ESRMEnv(
        sumocfg_path=SUMOCFG,
        junction_polygon=JUNCTION_POLYGON,
        episode_seconds=7200.0,
        step_length=0.1,
        vru_inner_edges=VRU_INNER_ALL,
    )

    base_reward = ESRMRewardU1()

    reward_ethics = ESRMRewardEthics(
        base_reward
    )

    env.rewarder = reward_ethics

    model = load_repaired_ppo(
        U2_MODEL,
        env
    )

    obs = env.reset()

    print("U2 rollout started...")
    print("Simulation horizon:", env.horizon)

    done = False
    step_counter = 0
    interaction_counter = 0

    start_time = time.time()

    action_counter = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }

    try:

        while not done:

            interaction_active = bool(
                obs[0]
            )

            if interaction_active:

                interaction_counter += 1

                action, _ = model.predict(
                    obs,
                    deterministic=False
                )

                action = int(action)

            else:

                action = 0

            action_counter[action] += 1

            obs, reward, done, info = env.step(
                action
            )

            step_counter += 1

            if step_counter % 10000 == 0:

                progress = (
                    100
                    * step_counter
                    / env.horizon
                )

                try:

                    sim_time = (
                        env.traci_conn
                        .simulation
                        .getTime()
                    )

                except Exception:

                    sim_time = -1

                print(
                    f"Step={step_counter} | "
                    f"Progress={progress:.1f}% | "
                    f"SUMO Time={sim_time:.1f}"
                )

    finally:

        env.close()
        
    if not done or step_counter < env.horizon:
        raise RuntimeError(
            "U2 simulation did not reach the full horizon. "
            f"Completed {step_counter} of {env.horizon} steps."
        )

    elapsed = time.time() - start_time

    print("\nAction distribution:")

    for action_id in sorted(action_counter):

        print(
            f"A{action_id}: "
            f"{action_counter[action_id]}"
        )

    print(
        f"\nInteractions: "
        f"{interaction_counter}"
    )

    interaction_rate = (
        100 * interaction_counter / step_counter
        if step_counter > 0
        else 0.0
    )

    print(
        f"Interaction rate: "
        f"{interaction_rate:.2f}%"
    )

    print(
        f"\nU2 simulation completed "
        f"in {elapsed:.1f} seconds."
    )

    return True

# ============================================================
# RUN ESRM PIPELINE
# ============================================================

def evaluate_scenario(result_folder):

    print(
        f"\nESRM evaluation not yet executed.\n"
        f"Result files saved in:\n"
        f"{result_folder}"
    )

   
    # Next integration step:
    
    # update config.SUMO_FOLDER
     #execute esrm_runner.run_pipeline()
    
    pass

# ============================================================
# MASTER ORCHESTRATION
# ============================================================
def execute_scenario(
    scenario_name,
    run_function
):

    result_folder = os.path.join(
        RESULT_ROOT,
        scenario_name
    )

    print("\n################################################")
    print(f"PREPARING SCENARIO: {scenario_name}")
    print("################################################")

    remove_old_ethics_outputs()
    remove_old_sumo_outputs()

    run_function()

    copy_outputs(
        result_folder
    )

    copy_ethics_outputs(
        result_folder
    )

    validate_ethics_summary(
        result_folder
    )

    evaluate_scenario(
        result_folder
    )

    print("\n################################################")
    print(f"SCENARIO COMPLETE: {scenario_name}")
    print("################################################")


def run_all():

    create_folder_structure()

    total_start_time = time.time()

    print("\n================================================")
    print("STARTING OUTCOME SIMULATION SEQUENCE")
    print("================================================")

#    execute_scenario(
#        scenario_name="Baseline",
#        run_function=run_baseline
#    )

 #   execute_scenario(
 #       scenario_name="U1",
 #       run_function=run_u1
 #   )

    execute_scenario(
        scenario_name="U2",
        run_function=run_u2
    )

    total_elapsed = (
        time.time() - total_start_time
    )

    print("\n================================================")
    print("ALL OUTCOME SIMULATIONS COMPLETE")
    print("================================================")

    print(
        f"Total wall-clock time: "
        f"{total_elapsed:.1f} seconds"
    )

    print(
        f"Results saved under:\n"
        f"{RESULT_ROOT}"
    )
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_all()