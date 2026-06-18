#!/bin/bash
# Native SLURM job-array submit — disconnect-proof alternative to the HyperQueue path.
#
# Each array element runs ONE registry entry via hpc/run_task.py, indexed by
# $SLURM_ARRAY_TASK_ID (== the old $HQ_TASK_ID). No HQ server/worker, so there is no
# login-node daemon to keep alive: closing your SSH/app cannot kill the run (sbatch
# allocations are detached from your session).
#
# Why an array works here (see hpc/snellius_reference.md):
#   - Single-node jobs SHARE nodes on Snellius (OverSubscribe=NO, ExclusiveUser=NO),
#     so --cpus-per-task=16 packs up to 8 tasks/node with no whole-node waste.
#   - QOS allows up to 128 concurrent jobs/user >> 29, so the whole array runs at once.
#
# Registry index map (hpc/registry.json, 32 entries):
#   0      = i10p_ppo_curriculum
#   1-24   = the 24 ADP grid configs
#   25-28  = the 4 MC-rollout configs (adaptive, bug-fixed)
#   29-31  = optuna 0A heuristics  ← already done, do NOT re-run
#
# Results are written to results/exp0/<run_name>/ (the configs' run_name is "exp0/...").
#
# Usage (run from the login node; no `hq` needed):
#   cd ~/Code_v2/
#   # Full 0B from scratch (old 0B results were cleared; no resume):
#   sbatch --array=0-28  hpc/submit_array.sh     # PPO + 24 ADP + 4 rollout
#   # Or a single smoke test first:
#   sbatch --array=1     hpc/submit_array.sh
#
# The CLI --array overrides the directive below, so you can submit any subset
# (e.g. --array=1,5,9 for specific configs). NOTE 29-31 (optuna 0A) are done — don't re-run.
#
# Monitor:  squeue -u $USER        |  sacct -j <jobid> --format=JobID,State,Elapsed
# Logs:     hpc/logs/slurm_<arrayjobid>_<taskid>.out / .err

#SBATCH --job-name=rl_infra
#SBATCH --array=0-28
#SBATCH --cpus-per-task=16
#SBATCH --time=28:00:00
# 16 cores = the Snellius minimum shareable slot (1/8 node; a node holds up to 8 jobs), so this is
# the smallest billed unit anyway — match n_workers=16 to use it fully. No --mem-per-cpu: the default
# (1792 MiB/core → ~28 GiB) keeps billing at 16 cores; an explicit 2G tipped it into the 32-core tier.
#SBATCH --partition=rome
#SBATCH --output=hpc/logs/slurm_%A_%a.out
#SBATCH --error=hpc/logs/slurm_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=robbert.bosch@pm.me

# NB: no `set -u` — venv activate / module functions reference unbound vars.
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source ~/.local/venv/bin/activate

cd ~/Code_v2/
mkdir -p hpc/logs

echo "[array] job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname) start=$(date -Is)"
python hpc/run_task.py --expe_id="${SLURM_ARRAY_TASK_ID}"
echo "[array] task=${SLURM_ARRAY_TASK_ID} done=$(date -Is)"
