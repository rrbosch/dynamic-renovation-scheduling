#!/bin/bash
# Executed once per HyperQueue task.
# $HQ_TASK_ID is set by HQ to the 0-based task index (matches registry.json).

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source ~/.local/venv/bin/activate

cd ~/Code_v2/

python hpc/run_task.py --expe_id=$HQ_TASK_ID
