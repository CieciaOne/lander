# Configurations

This directory is the canonical place for scenario YAMLs — copy and edit one of the starter files to begin a new experiment. Load them via `python scripts/run_scenario.py --config configs/<name>.yaml`.

- RL training hyperparameters (PPO/SAC), network sizes.
- CL: λ (EWC), replay buffer size (%), strategy (uniform/reservoir).
- After Phases 3 and 5: `best_cl_autonomy.yaml`, `best_cl_security.yaml`.
