# -*- coding: utf-8 -*-
"""
train_ppo_ethics.py
PPO training for ESRM with Ethics (S + T + F + E) --> U2
"""

import os
import sys
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import numpy as np
import torch
import random

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)



def set_global_seed(seed):

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
# ======================================================
# 🔥 FIX IMPORT PATH (robust)
# ======================================================
PROJECT_PATH = r"C:\Users\49151\Desktop\Master\07_Masterthesis\00_InTas_New - RL\Pyscripts"

if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

print("✅ Working dir:", os.getcwd())
print("✅ Project path:", PROJECT_PATH)

# ======================================================
# IMPORTS
# ======================================================
#from esrm_env.env import ESRMEnv #u2a --> only if interaction = active (U1 und U2a)
#from esrm_env.env_u2b import ESRMEnv #u2b --> Globale berechnung --> abgelöst durch env.py: U1 und U2 nutzen selbes environment
from esrm_env.env import ESRMEnv
from stfe_callback import STFELoggerCallback

from esrm_env.reward import ESRMRewardU1
from reward_ethics import ESRMRewardEthics

# ======================================================
# TRAINING CONFIG
# ======================================================
TOTAL_TIMESTEPS = 500_000

PRINT_EVERY_STEPS = max(250, TOTAL_TIMESTEPS // 10)
PRINT_EVERY_PERCENT = 100.0 / (TOTAL_TIMESTEPS // PRINT_EVERY_STEPS)

#SEEDS = [1, 2, 3]
SEEDS = [3]

print(f"✅ TOTAL_TIMESTEPS: {TOTAL_TIMESTEPS}")
print(f"✅ Print every {PRINT_EVERY_STEPS} steps (~{PRINT_EVERY_PERCENT:.1f}%)")

# ======================================================
# SUMO / SCENARIO CONFIG
# ======================================================
#SUMOCFG = r"C:\Users\49151\Desktop\Master\07_Masterthesis\00_InTas_New - RL\01_Morning\InTAS_2.0.RL.sumocfg" #Morning Peak
SUMOCFG = r"C:\Users\49151\Desktop\Master\07_Masterthesis\00_InTas_New - RL\02_Midday\InTAS_2.0.RL.sumocfg" #Midday 12-14

JUNCTION_POLYGON = [
    (215275.72, 450927.73), (215284.45, 450922.08), (215284.04, 450921.03),
    (215284.07, 450920.57), (215284.26, 450920.14), (215284.60, 450919.75),
    (215285.10, 450919.40), (215279.90, 450910.39), (215277.88, 450910.67),
    (215276.72, 450910.22), (215275.47, 450909.38), (215274.13, 450908.16),
    (215272.69, 450906.54), (215264.65, 450913.15), (215265.18, 450914.14),
    (215265.21, 450914.61), (215265.08, 450915.06), (215264.78, 450915.49),
    (215264.33, 450915.90), (215270.63, 450924.18), (215272.03, 450923.94),
    (215272.84, 450924.35), (215273.73, 450925.12), (215274.69, 450926.25),
]

VRU_INNER_ALL = {
    "-201953160#1", "-201953160#4", "-208010459#1", "-40799279",
    "201953160#0", "201953160#3", "208010459#1", "40799279",
}

# ======================================================
# ENV FACTORY
# ======================================================
def make_env(seed):

    env = ESRMEnv(
        sumocfg_path=SUMOCFG,
        junction_polygon=JUNCTION_POLYGON,
        episode_seconds=600.0,
        step_length=0.1,
        vru_inner_edges=VRU_INNER_ALL,
        
    )
    
    env.reset(seed=seed) 
    print("✅ ENV: U2-V2 (shared ESRM environment)")

    # --------------------------------------------------
    # 🔥 REWARD REPLACEMENT (ETHICS INJECTION)
    # --------------------------------------------------
    base_reward = ESRMRewardU1()
    reward_ethics = ESRMRewardEthics(base_reward)

    env.rewarder = reward_ethics

    print("✅ Using ETHICS reward: ESRMRewardEthics (S + T + F + E)")
    print("✅ Ethics scaling W4 = 0.7")

    # Monitor logging
    monitor_dir = os.path.join(
        PROJECT_PATH,
        "logs",
        "monitor_ethics",
        f"seed_{seed}"
    )
    os.makedirs(monitor_dir, exist_ok=True)

    return Monitor(env, monitor_dir)


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    for seed in SEEDS:

        print(f"\n==============================")
        print(f"🚀 RUN WITH SEED {seed}")
        print(f"==============================\n")

        # ✅ Seed setzen
        set_seed(seed)

        env = make_env(seed)

        # ✅ eigener Run-Ordner pro Seed
        run_id = time.strftime("%Y%m%d_%H%M%S")

        TB_LOG_DIR = os.path.join(
            PROJECT_PATH,
            "logs",
            "TB",
            f"run_ethics_seed_{seed}_{run_id}"
        )

        os.makedirs(TB_LOG_DIR, exist_ok=True)

        print(f"✅ TensorBoard log dir: {TB_LOG_DIR}")

        # ==================================================
        # PPO MODEL
        # ==================================================
        model = PPO(
            policy="MlpPolicy",
            env=env,
            n_steps=4096,
            batch_size=256,
            n_epochs = 12,
            learning_rate=2e-4,
            gamma=0.97,
            clip_range=0.2,
            vf_coef=0.7,
            gae_lambda=0.95,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=TB_LOG_DIR,
            seed=seed,  
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128])
            ),

        )

        # ==================================================
        # CALLBACK
        # ==================================================
        stfe_callback = STFELoggerCallback(
            total_timesteps=TOTAL_TIMESTEPS,
            print_every_steps=PRINT_EVERY_STEPS,
            print_every_percent=PRINT_EVERY_PERCENT,
            verbose=0,
        )

        # ==================================================
        # TRAINING
        # ==================================================
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=stfe_callback,
            reset_num_timesteps=True,
        )

        # ==================================================
        # SAVE MODEL
        # ==================================================
        #model_path = os.path.join(
        #    PROJECT_PATH,
        #    f"ppo_esrm_ethics_seed_{seed}.zip"
        #)

        model_path = os.path.join(
            PROJECT_PATH,
            f"ppo_esrm_u2_online_ethics_seed_{seed}.zip"
        )

        model.save(model_path)

        env.close()

        print(f"\n✅ Seed {seed} training finished.")
        print(f"✅ Model saved at: {model_path}")
        print(f"✅ TensorBoard logs: {TB_LOG_DIR}")

 