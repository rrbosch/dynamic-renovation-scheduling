# Snellius Run Manual

Step-by-step commands for running experiments on Snellius via a **native SLURM job array**
(`hpc/submit_array.sh`). No HyperQueue: there is no server/worker daemon to keep alive, so the run
**survives SSH/app disconnects** (a login-node HQ server dying is what killed an earlier run). The
older HQ-based workflow is preserved in `snellius_manual_old.md`.

Partition / sharing / QOS facts (why a 29-task array runs fully concurrent with no node waste) are in
`hpc/snellius_reference.md`.

---

## Quick Start

From a login node, with the venv and code already in place:

```bash
cd ~/Code_v2/
sbatch --array=1     hpc/submit_array.sh    # smoke test: one ADP config
sbatch --array=0-28  hpc/submit_array.sh    # full Exp-0B (PPO + 24 ADP + 4 rollout)
squeue -u $USER                             # watch it run
```

Results land in `results/exp0/<run_name>/`. That's it — no `hq`, no separate worker job, no server.

---

## 1. One-time: Python environment

Only the first time (or after a Python-module change):

```bash
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

python3 -m venv ~/.local/venv          # skip if ~/.local/venv/ already exists
source ~/.local/venv/bin/activate
cd ~/Code_v2/
pip install -r requirements.txt
```

(The array script loads these modules and activates this venv itself per task — you don't need to
activate it just to submit.)

---

## 2. Sync code, configs, and registry from your laptop

Run on your **laptop**. `rsync` is preferred (skips `results/` and caches):

```bash
rsync -av --delete \
  --exclude 'results/' --exclude '__pycache__/' --exclude '.git/' \
  agents/ experiments/ env/ training/ utils/ hpc/ configs/ instances/ \
  <username>@snellius.surf.nl:~/Code_v2/
```

Make sure these in particular are current: all `configs/i10p_*.json` (their `run_name` is `exp0/…`),
`hpc/registry.json`, `hpc/submit_array.sh`, and any code you changed.

---

## 3. The registry and the index map

`hpc/submit_array.sh` runs one registry entry per array element via
`python hpc/run_task.py --expe_id=$SLURM_ARRAY_TASK_ID`. The index → config mapping
(`hpc/registry.json`, 32 entries):

| Index | Configs |
|---|---|
| `0` | `i10p_ppo_curriculum` |
| `1-24` | the 24 ADP grid configs |
| `25-28` | the 4 MC-rollout configs |
| `29-31` | optuna 0A heuristics — **already done, do NOT re-run** |

Count / inspect:

```bash
python3 -c "import json; r=json.load(open('hpc/registry.json')); print(len(r),'entries'); [print(i, r[i]['config']) for i in (0,24,25,28)]"
```

To regenerate the registry (single replication for Exp 0):

```bash
source ~/.local/venv/bin/activate
python hpc/generate_registry.py --configs "configs/i10p_*.json"      # add --seeds 0 1 2 3 4 for Exp 1+
```

---

## 4. Submit the array

The `#SBATCH` defaults are baked into `hpc/submit_array.sh`
(`--array=0-28`, `--cpus-per-task=16`, `--time=28:00:00`, `--partition=rome`; no `--mem-per-cpu`).
16 cores = the Snellius minimum shareable slot (1/8 node), so it's the smallest billed unit anyway —
`n_workers=16` uses it fully. (See `hpc/registry_conventions.md`.)
A CLI `--array` overrides the directive, so:

```bash
# Smoke test ONE config first (confirm it starts fresh + writes results/exp0/...):
sbatch --array=1 hpc/submit_array.sh

# Full run:
sbatch --array=0-28 hpc/submit_array.sh

# Any subset is fine, e.g. just the rollout configs:
sbatch --array=25-28 hpc/submit_array.sh
```

`sbatch` prints a `<jobid>`. All 29 elements run concurrently (QOS allows up to 128/user; single-node
jobs share nodes, so no whole-node waste). **Walltime is 28h** — longer than the 24h training
`time_budget` so the final evaluation (`experiments/run.py` → `trainer.evaluate(save_episodes=True)`)
completes and writes `eval_episodes.csv`. A run that is cut off *before* that final eval produces no
eval file.

---

## 5. Monitor

```bash
squeue -u $USER                                              # queued / running array elements
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS%12     # per-element state (RUNNING/COMPLETED/TIMEOUT/FAILED)
tail -f "hpc/logs/slurm_<jobid>_1.out"                       # live log for array task 1
grep -l "Resuming from checkpoint" hpc/logs/slurm_<jobid>_*.out   # which tasks resumed vs started fresh
```

Per-task stdout/stderr: `hpc/logs/slurm_%A_%a.out` / `.err` (`%A`=array job id, `%a`=task index).
Per-run output: `results/exp0/<run_name>/` (config, checkpoints, `training_log.csv`,
`eval_episodes.csv`).

**Disconnect test:** after submitting, close your session and reconnect — `squeue -u $USER` still
shows the array. That's the whole point of dropping the HQ server.

---

## 6. Resume after a timeout or partial failure

Runs checkpoint periodically (default every 30 min) and auto-resume on the **same config**: resubmit
the array and each incomplete run continues from its latest checkpoint (`auto_resume=True`, keyed on
`config_hash`). To avoid re-running ones that already finished, resubmit only the unfinished indices:

```bash
# Identify finished runs (have a complete checkpoint / eval_episodes.csv):
ls results/exp0/*/eval_episodes.csv 2>/dev/null

# Resubmit only the indices that didn't finish, e.g.:
sbatch --array=3,7,12-14 hpc/submit_array.sh
```

No work is lost — resumed runs pick up their episode counter and elapsed time from the checkpoint.

---

## 7. Pull results back

On your **laptop**:

```bash
rsync -av <username>@snellius.surf.nl:~/Code_v2/results/exp0/ "results/exp0/"
```

Then analyze locally (paired-CRN comparison vs the tuned reactive heuristic baseline).

---

## Quick reference

| Action | Command |
|---|---|
| Sync code up (laptop) | `rsync -av --exclude results/ … snellius:~/Code_v2/` |
| Count registry entries | `python3 -c "import json; print(len(json.load(open('hpc/registry.json'))))"` |
| Smoke test one config | `sbatch --array=1 hpc/submit_array.sh` |
| Submit full 0B | `sbatch --array=0-28 hpc/submit_array.sh` |
| Watch queue | `squeue -u $USER` |
| Per-element state | `sacct -j <jobid> --format=JobID,State,Elapsed` |
| Cancel | `scancel <jobid>` (or `scancel -u $USER`) |
| Pull results (laptop) | `rsync -av snellius:~/Code_v2/results/exp0/ results/exp0/` |
