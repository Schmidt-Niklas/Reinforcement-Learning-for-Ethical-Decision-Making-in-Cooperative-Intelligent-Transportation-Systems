# Reinforcement-Learning-for-Ethical-Decision-Making-in-Cooperative-Intelligent-Transportation-Systems

This repository contains the Python implementation developed for training and evaluating infrastructure-level reinforcement-learning policies in a multimodal SUMO traffic scenario. The project compares an operational agent based on Safety, Throughput, and Fairness with an ethics-enhanced agent that additionally considers interaction criticality and road-user vulnerability. The evaluation investigates both conventional traffic performance and the allocation of mitigation responsibility in vehicle–VRU interactions.

**Trainingspipeline:** 
The Trainingspipeline directory contains the custom reinforcement-learning environment, reward modules, intervention controller, tracking components, callbacks, and PPO training scripts. ![Flowchart of the RL-Trainings Pipeline (Generated with AI assistance)](Trainingspipeline/Flowchart%20RL%20Trainingspipeline.png)


- train_ppo_v4.py
Configures and executes PPO training for the operational agent based on Safety, Throughput, and Fairness.

- train_ppo_ethics_V2.py
Configures and executes PPO training for the ethics-enhanced agent using the additional Ethics-dependent reward modification.

- env_V2.py
Implements the Gym-compatible SUMO environment and coordinates episode initialization, action scheduling, simulation progression, interaction detection, reward gating, and communication with the PPO agent.

- observation.py
Extracts the multimodal traffic observation, including interaction activity, surrogate-safety indicators, participant counts, waiting times, and the critical vehicle–VRU pair.

- controller.py
Translates the discrete action selected by the reinforcement-learning agent into an infrastructure-level traffic intervention and applies the intervention through TraCI.

- reward.py
Calculates the operational reward from Safety, Throughput, and Fairness.

- ethics_tracker.py
Tracks vehicle–VRU interaction pairs and stores pair-based events for the independent Final Outcome Evaluation.

- reward_ethics.py
Applies the Ethics-dependent modification to the complete operational reward of the ethics-enhanced agent.

- ethics.py
Calculates the Ethical Pre-Crash Factor based on interaction criticality, severity, and road-user vulnerability.

- stf_callback.py
Records interaction, Safety, Throughput, Fairness, and objective-contribution metrics during the training of the operational agent.

- stfe_callback.py
Records the corresponding training metrics for the ethics-enhanced agent, including Ethics-specific factors and reward adjustments.



**Outcome Evaluation:**
The Outcome Evaluation directory contains the scripts used to execute and independently evaluate the native SUMO Baseline and the selected operational and ethics-enhanced policies. 
![Python pipeline of the Final Outcome Evaluation](Outcome%20Evaluation/Python%20pipeline%20of%20the%20Final%20Outcome%20Evaluation.png)

- config.py
Defines paths, scenario settings, simulation intervals, selected policy files, and other parameters required by the evaluation pipeline.

- run_outcome_evaluation.py
Provides the main entry point for running the complete Final Outcome Evaluation.

- esrm_runner.py
Executes the configured control strategy in SUMO and coordinates simulation output generation and pair-based interaction tracking.

- safety_output_final.py
Calculates collision and surrogate-safety outcomes, including localized critical events and Time Integrated TTC.

- throughput_output.py
Calculates completed cars, buses, pedestrians, and total traffic participants per simulation minute.


- fairness_waiting_output.py
Calculates within-group waiting-time equality and cross-group Fairness for cars, buses, and vulnerable road users.

- ethics_output_final_V2.py
Calculates the final pair-based Ethics outcomes, including interaction Criticality, Yielding Behaviour, Classification Coverage, mitigation-burden allocation, Ethical Alignment, and the VRU Priority Score V2.

- plot_tb_dashboard_v3.py
Processes logged evaluation or dashboard metrics and creates the corresponding summary plots.

- analyze_ethics_events_v2.py
Analyzes the pair-based vehicle–VRU interaction events and performs additional calculations and consistency checks for the Ethics evaluation.

- compare_selected_agents.py
Loads the evaluation results of Baseline, the operational agent, and the ethics-enhanced agent and generates the comparative result figures.


