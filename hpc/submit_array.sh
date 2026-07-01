#!/bin/bash
# Native SLURM job-array submit — disconnect-proof (no HyperQueue daemon to keep alive).
#
# Each array element runs ONE registry entry via hpc/run_task.py, indexed by
# $SLURM_ARRAY_TASK_ID. The registry is passed as the FIRST script argument, so you can
# pre-create many named registries under hpc/registries/ and dispatch each independently.
#
# Job-name convention: registry hpc/registries/<experiment>_<appendix>.json  ->  job name
# rl_<experiment>_<appendix>  ->  logs hpc/logs/rl_<...>_<jobid>_<task>.{out,err}  (%x = job name).
# The thin wrapper hpc/submit.sh derives that name for you.
#
# Usage (run from the login node):
#   cd ~/Code_v2/
#   bash hpc/submit.sh hpc/registries/sf15_0a.json 0-7          # recommended: auto job name rl_sf15_0a
#   # or drive sbatch directly (set the name yourself):
#   sbatch --job-name=rl_sf15_0a --array=0-7 hpc/submit_array.sh hpc/registries/sf15_0a.json
#   # single smoke test:
#   sbatch --job-name=rl_sf15_0a --array=2  hpc/submit_array.sh hpc/registries/sf15_0a.json
#
# Why an array works here (see hpc/snellius_reference.md): single-node jobs SHARE nodes on
# Snellius (--cpus-per-task=16 packs up to 8 tasks/node), and QOS allows up to 128 concurrent
# jobs/user, so the whole array runs at once.
#
# Monitor:  squeue -u $USER        |  sacct -j <jobid> --format=JobID,JobName,State,Elapsed
# Logs:     hpc/logs/<jobname>_<arrayjobid>_<taskid>.out / .err

#SBATCH --job-name=rl_exp
#SBATCH --cpus-per-task=16
#SBATCH --time=28:00:00
# 16 cores = the Snellius minimum shareable slot (1/8 node; a node holds up to 8 jobs), so this is
# the smallest billed unit anyway — match n_workers=16 to use it fully. No --mem-per-cpu: the default
# (1792 MiB/core → ~28 GiB) keeps billing at 16 cores; an explicit 2G tipped it into the 32-core tier.
#SBATCH --partition=rome
#SBATCH --output=hpc/logs/%x_%A_%a.out
#SBATCH --error=hpc/logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=robbert.bosch@pm.me

# First script arg = registry JSON to dispatch (required). --array is supplied on the CLI.
REG="${1:?usage: sbatch --array=A-B hpc/submit_array.sh <registry.json>  (or use hpc/submit.sh)}"

# NB: no `set -u` — venv activate / module functions reference unbound vars.
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source ~/.local/venv/bin/activate

cd ~/Code_v2/
mkdir -p hpc/logs

echo "[array] job=${SLURM_JOB_NAME} id=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} registry=${REG} host=$(hostname) start=$(date -Is)"
python hpc/run_task.py --expe_id="${SLURM_ARRAY_TASK_ID}" --registry "${REG}"
echo "[array] task=${SLURM_ARRAY_TASK_ID} done=$(date -Is)"
