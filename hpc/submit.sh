#!/bin/bash
# SLURM job script — allocates nodes and starts HyperQueue workers.
# Submit with: sbatch hpc/submit.sh
#
# Full execution procedure (run from login node after ssh into Snellius):
#
#   cd ~/Code_v2/
#
#   # 1. Load modules and start HQ server (skip if already running)
#   module load 2023 && module load HyperQueue/0.19.0
#   hq server info || nohup hq server start &
#
#   # 2. Generate experiment registry
#   python hpc/generate_registry.py --configs configs/exp1_*.json --seeds 0 1 2 3 4
#   # → prints: "N experiments queued → update --array 0-{N-1}"
#
#   # 3. Submit task array to HQ (update N to match step 2 output)
#   hq submit --array 0-{N-1} --pin taskset --cpus=1 hpc/hq_task.sh
#
#   # 4. Submit this SLURM job to spawn workers
#   sbatch hpc/submit.sh
#
#   # 5. Monitor
#   hq job list
#   hq job progress <job_id>
#   hq task list <job_id> | grep FAILED | wc -l
#
#   # Cleanup
#   hq job cancel all       # cancel pending/running tasks
#   hq server stop          # shut down when done

#SBATCH --job-name=rl_infra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=hpc/logs/slurm_%j.out
#SBATCH --error=hpc/logs/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=robbert.bosch@pm.me

module load 2023
module load HyperQueue/0.19.0

mkdir -p ~/Code_v2/hpc/logs

# Auto-detect all 128 CPUs; with --cpus=8 per HQ task this gives 16 concurrent tasks per node
hq worker start &
wait
