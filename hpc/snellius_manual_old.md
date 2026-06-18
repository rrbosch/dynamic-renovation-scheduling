# Snellius Run Manual

Step-by-step commands for running experiments on Snellius via SLURM + HyperQueue.

---

## Quick Start

All commands from a login node, assuming venv and code are already set up:

```bash
module load 2023
module load HyperQueue/0.19.0
cd ~/Code_v2/
nohup hq server start &
python3 -c "import json; r=json.load(open('hpc/registry.json')); print(len(r), 'experiments'); print(f'  -> array index: 0-{len(r)-1}')"
hq submit --array 0-19 --cpus=8 --pin taskset hpc/hq_task.sh   # replace 19 with last index
sbatch hpc/submit.sh
```

---

## 1. One-time: Python environment

Only needed the very first time (or after a Python module change).

```bash
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Create venv (skip if ~/.local/venv/ already exists)
python3 -m venv ~/.local/venv

# Activate and install dependencies
source ~/.local/venv/bin/activate
cd ~/Code_v2/
pip install -r requirements.txt
```

---

## 2. Start the HyperQueue server

The HQ server runs on the login node and persists across sessions.

```bash
module load 2023
module load HyperQueue/0.19.0

nohup hq server start &
hq server info            # verify
```

---

## 3. Verify all code is up-to-date, upload configs and registry

Make sure the latest code, config files, and registry are synced to Snellius:

```bash
# Run this on your laptop:
scp -r agents/ experiments/ env/ training/ utils/ hpc/ configs/ <username>@snellius.surf.nl:~/Code_v2/
scp hpc/registry.json <username>@snellius.surf.nl:~/Code_v2/hpc/registry.json
```

Alternatively, regenerate the registry directly on Snellius (picks up any status changes):

```bash
cd ~/Code_v2/
source ~/.local/venv/bin/activate
python hpc/generate_registry.py   # interactive: select configs + seeds
```

---

## 4. Count experiments

```bash
python3 -c "import json; r=json.load(open('hpc/registry.json')); print(len(r), 'experiments'); print(f'  -> array index: 0-{len(r)-1}')"
```

Example output:
```
25 experiments
  -> array index: 0-24
```

---

## 5. Submit the HQ task array

Replace `24` with your actual last index (N-1):

```bash
hq submit --array 0-24 --cpus=8 --pin taskset hpc/hq_task.sh
```

Note the returned `<job_id>` — you'll use it for monitoring.

---

## 6. Submit the SLURM job (spawns workers)

```bash
sbatch hpc/submit.sh
```

Check it was queued:

```bash
squeue -u $USER
```

Once SLURM starts the job, the HQ worker connects to the server and tasks begin executing.

---

## 7. Monitor progress

```bash
hq job list                              # all HQ jobs and their status
hq job progress <job_id>                 # live progress bar for a specific job
hq task list <job_id> | grep FAILED      # show failed tasks
hq task list <job_id> | grep WAITING     # show tasks still in queue
```

SLURM logs (stdout/stderr per SLURM job):

```bash
ls hpc/logs/
tail -f hpc/logs/slurm_<slurm_job_id>.out
```

Per-task output goes to `results/<run_name>/` as usual.

---

## 8. After all tasks complete

```bash
hq job list --all                        # confirm all tasks finished
ls results/                              # check result directories exist
hq server stop                           # shut down HQ server when fully done
```

---

## 9. Resuming after timeout or partial failure

If the SLURM job times out before all tasks finish (or some tasks failed):

```bash
# Re-generate registry — already-finished runs are skipped automatically
python hpc/generate_registry.py   # interactive, or:
python hpc/generate_registry.py --configs configs/exp1_*.json --seeds 0 1 2 3 4

# Count remaining experiments and resubmit (steps 4-6 above)
```

The trainer auto-resumes from the latest checkpoint, so no work is lost.

To inspect which tasks failed and why:

```bash
hq task list <job_id> --filter failed
# Then check the corresponding results/ directory for partial output
```

---

## Quick reference

| Action | Command |
|---|---|
| Check HQ server | `hq server info` |
| Start HQ server | `nohup hq server start &` |
| Count registry entries | `python3 -c "import json; print(len(json.load(open('hpc/registry.json'))))"` |
| Submit task array | `hq submit --array 0-{N-1} --cpus=8 --pin taskset hpc/hq_task.sh` |
| Submit SLURM workers | `sbatch hpc/submit.sh` |
| Monitor | `hq job progress <job_id>` |
| Cancel all tasks | `hq job cancel all` |
| Stop HQ server | `hq server stop` |
