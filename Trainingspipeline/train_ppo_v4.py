# -*- coding: utf-8 -*-
"""
train_ppo_v4.py

PPO training for ESRM U1.

U1 uses the shared ESRM env_v2 environment with the
base reward consisting of:

- Safety
- Throughput
- Fairness

No additional ethics reward is used.
"""

import os
import random
import time

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from esrm_env.env_V2 import ESRMEnv
from esrm_env.reward import ESRMRewardU1
from esrm_env.stf_callback import STFLoggerCallback


# ======================================================
# PROJECT CONFIGURATION
# ======================================================

PROJECT_PATH = (
    r"C:\Users\49151\Desktop\Master\07_Masterthesis"
    r"\00_InTas_New - RL\Pyscripts"
)


# ======================================================
# REPRODUCIBILITY
# ======================================================

def set_global_seed(seed):

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ======================================================
# DEBUG
# ======================================================

print(
    "Working directory:",
    os.getcwd()
)

print(
    "Project path:",
    PROJECT_PATH
)


# ======================================================
# TRAINING CONFIGURATION
# ======================================================

TOTAL_TIMESTEPS = 500_000

# Final training:
TOTAL_TIMESTEPS = 500_000
#SEEDS = [1, 2, 3]
SEEDS = [3]

PRINT_EVERY_STEPS = max(
    250,
    TOTAL_TIMESTEPS // 10
)

PRINT_EVERY_PERCENT = (
    100.0
    / (
        TOTAL_TIMESTEPS
        // PRINT_EVERY_STEPS
    )
)


# ======================================================
# SUMO / SCENARIO CONFIGURATION
# ======================================================

SUMOCFG = (
    r"C:\Users\49151\Desktop\Master\07_Masterthesis"
    r"\00_InTas_New - RL\02_Midday"
    r"\InTAS_2.0.RL.sumocfg"
)

JUNCTION_POLYGON = [
    (215275.72, 450927.73),
    (215284.45, 450922.08),
    (215284.04, 450921.03),
    (215284.07, 450920.57),
    (215284.26, 450920.14),
    (215284.60, 450919.75),
    (215285.10, 450919.40),
    (215279.90, 450910.39),
    (215277.88, 450910.67),
    (215276.72, 450910.22),
    (215275.47, 450909.38),
    (215274.13, 450908.16),
    (215272.69, 450906.54),
    (215264.65, 450913.15),
    (215265.18, 450914.14),
    (215265.21, 450914.61),
    (215265.08, 450915.06),
    (215264.78, 450915.49),
    (215264.33, 450915.90),
    (215270.63, 450924.18),
    (215272.03, 450923.94),
    (215272.84, 450924.35),
    (215273.73, 450925.12),
    (215274.69, 450926.25),
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


# ======================================================
# ENVIRONMENT FACTORY
# ======================================================

def make_env(
    seed,
    run_root
):

    # U1 reward:
    # Safety + Throughput + Fairness
    base_reward = ESRMRewardU1()

    env = ESRMEnv(
        sumocfg_path=SUMOCFG,
        junction_polygon=JUNCTION_POLYGON,
        episode_seconds=600.0,
        step_length=0.1,
        vru_inner_edges=VRU_INNER_ALL,
        seed=seed,
        rewarder=base_reward,
    )

    monitor_dir = os.path.join(
        run_root,
        "monitor"
    )

    os.makedirs(
        monitor_dir,
        exist_ok=True
    )

    print(
        "ENV: U1 with shared env_v2 environment"
    )

    print(
        "Reward: ESRMRewardU1 "
        "(Safety + Throughput + Fairness)"
    )

    return Monitor(
        env,
        filename=os.path.join(
            monitor_dir,
            "monitor.csv"
        )
    )


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    print(
        f"TOTAL_TIMESTEPS: "
        f"{TOTAL_TIMESTEPS}"
    )

    print(
        f"Seeds: {SEEDS}"
    )

    print(
        f"Print every "
        f"{PRINT_EVERY_STEPS} steps "
        f"(approximately "
        f"{PRINT_EVERY_PERCENT:.1f}%)"
    )

    for seed in SEEDS:

        print(
            "\n=============================="
        )

        print(
            f"RUN U1 WITH SEED {seed}"
        )

        print(
            "==============================\n"
        )

        # --------------------------------------
        # Set Python, NumPy and PyTorch seeds
        # --------------------------------------

        set_global_seed(
            seed
        )

        # --------------------------------------
        # Run-specific directories
        # --------------------------------------

        run_id = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        run_root = os.path.join(
            PROJECT_PATH,
            "logs",
            "U1",
            f"seed_{seed}",
            run_id
        )

        tensorboard_dir = os.path.join(
            run_root,
            "tensorboard"
        )

        model_dir = os.path.join(
            PROJECT_PATH,
            "models",
            "U1",
            f"seed_{seed}"
        )

        os.makedirs(
            tensorboard_dir,
            exist_ok=True
        )

        os.makedirs(
            model_dir,
            exist_ok=True
        )

        print(
            f"Run directory: "
            f"{run_root}"
        )

        print(
            f"TensorBoard directory: "
            f"{tensorboard_dir}"
        )

        print(
            f"Model directory: "
            f"{model_dir}"
        )

        # --------------------------------------
        # Environment
        # --------------------------------------

        env = make_env(
            seed=seed,
            run_root=run_root
        )

        print(
            "Observation space:",
            env.observation_space
        )

        print(
            "Action space:",
            env.action_space
        )

        # --------------------------------------
        # PPO model
        # --------------------------------------

        model = PPO(
            policy="MlpPolicy",
            env=env,

            n_steps=4096,
            batch_size=256,
            n_epochs=12,

            learning_rate=2e-4,
            gamma=0.97,
            gae_lambda=0.95,

            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.7,

            policy_kwargs={
                "net_arch": {
                    "pi": [
                        128,
                        128,
                    ],
                    "vf": [
                        128,
                        128,
                    ],
                }
            },

            verbose=1,
            tensorboard_log=(
                tensorboard_dir
            ),
            seed=seed,
        )

        # --------------------------------------
        # Callback
        # --------------------------------------

        callback = STFLoggerCallback(
            total_timesteps=(
                TOTAL_TIMESTEPS
            ),
            print_every_steps=(
                PRINT_EVERY_STEPS
            ),
            print_every_percent=(
                PRINT_EVERY_PERCENT
            ),
            verbose=0,
        )

        # --------------------------------------
        # Model output path
        # --------------------------------------

        model_path = os.path.join(
            model_dir,
            (
                "ppo_esrm_u1_"
                f"500k_seed_{seed}"
            )
        )

        # --------------------------------------
        # Training
        # --------------------------------------

        try:

            model.learn(
                total_timesteps=(
                    TOTAL_TIMESTEPS
                ),
                callback=callback,
                reset_num_timesteps=True,
            )

            model.save(
                model_path
            )

            print(
                f"Model saved at: "
                f"{model_path}"
            )

            print(
                f"Seed {seed} training "
                "finished successfully."
            )

        finally:

            env.close()

            print(
                f"Environment closed for "
                f"seed {seed}."
            )

    print(
        "\nALL U1 TRAINING RUNS COMPLETED"
    )