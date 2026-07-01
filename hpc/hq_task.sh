#!/bin/bash
# Executed once per HyperQueue task.
# $HQ_TASK_ID is set by HQ to the 0-based task index.
#
# DEPRECATED: prefer the native SLURM job array (hpc/submit.sh / hpc/submit_array.sh) — it needs no
# login-node HQ daemon and takes a named registry. This HQ path reads the DEFAULT registry
# (hpc/registry.json); pass --registry explicitly to target a named one under hpc/registries/.

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source ~/.local/venv/bin/activate

cd ~/Code_v2/

python hpc/run_task.py --expe_id=$HQ_TASK_ID
