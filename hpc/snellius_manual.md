# Snellius Run Manual

Step-by-step commands for running experiments on Snellius via a **native SLURM job array**
(`hpc/submit_array.sh`, driven by `hpc/submit.sh`). No HyperQueue: there is no server/worker daemon to
keep alive, so the run **survives SSH/app disconnects** (a login-node HQ server dying is what killed an
earlier run). The older HQ-based workflow is preserved in `snellius_manual_old.md`.

Dispatch is organised around **named per-experiment registries** under `hpc/registries/`. Each registry
is one batch of runs; the filename sets the SLURM job name:

```
hpc/registries/<experiment>_<appendix>.json  ->  job rl_<experiment>_<appendix>
                                             ->  logs hpc/logs/rl_<...>_<jobid>_<task>.{out,err}
```

Partition / sharing / QOS facts (why a whole array runs concurrent with no node waste) are in
`hpc/snellius_reference.md`; the registry/job-name convention is in `hpc/registry_conventions.md`.

---

## Quick Start

From a login node, with the venv and code already in place (current batch = **sf24 0A**, 8 heuristics):

```bash
cd ~/Code_v2/
bash hpc/submit.sh hpc/registries/sf24_0a.json 2      # smoke test: one config (task 2 = perasset)
bash hpc/submit.sh hpc/registries/sf24_0a.json 0-7    # full batch -> job rl_sf24_0a
squeue -u $USER                                        # watch it run (JobName column shows rl_sf24_0a)
```

Results land in `results/exp0/<run_name>/`. No `hq`, no separate worker job, no server.

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

## 2. Sync code, configs, and registries from your laptop

Run on your **laptop**. `rsync` skips `results/` and caches. **`env/` is now tracked and must be
synced** (the bidirectional asset model lives there):

```bash
rsync -av --delete \
  --exclude 'results/' --exclude '__pycache__/' --exclude '.git/' \
  agents/ experiments/ env/ training/ utils/ hpc/ configs/ instances/ \
  <username>@snellius.surf.nl:~/Code_v2/
```

Make sure these are current: the `configs/` for the batch you're running, `instances/` (e.g.
`instance_sf24.json`), `hpc/registries/`, `hpc/submit.sh`, `hpc/submit_array.sh`, and any changed code.
After a big code change (e.g. the bidirectional model), run the **login-node canary** in §3 before
submitting.

---

## 3. Registries, the index map, and a build canary

`hpc/submit_array.sh <registry.json>` runs one registry entry per array element via
`python hpc/run_task.py --expe_id=$SLURM_ARRAY_TASK_ID --registry <registry.json>`. Each registry is a
flat JSON list; the array index is the 0-based position in that list.

Inspect / dry-run a registry (no compute):

```bash
python3 -c "import json; r=json.load(open('hpc/registries/sf24_0a.json')); print(len(r),'entries'); [print(i,e['config']) for i,e in enumerate(r)]"
python hpc/run_task.py --registry hpc/registries/sf24_0a.json --expe_id 2 --dry-run   # prints the resolved config, no run
```

**Login-node build canary** (catches a stale/partial upload before you burn an array):

```bash
source ~/.local/venv/bin/activate
python -c "from experiments.configs import ExperimentConfig, build_experiment as b; \
b(ExperimentConfig.from_file('configs/sf24_optuna_perasset.json')); print('build OK')"
```

Create a new named registry for a batch (prints the exact submit line):

```bash
python hpc/generate_registry.py --configs "configs/sf24_optuna_*.json" \
    --output hpc/registries/sf24_0a.json          # add --seeds 0 1 2 3 4 for Exp 1+
```

---

## 4. Submit the array

Use the wrapper — it derives the job name from the registry filename and passes the registry through:

```bash
# Smoke ONE config first (confirm it starts + writes results/exp0/...):
bash hpc/submit.sh hpc/registries/sf24_0a.json 2

# Full batch (job name rl_sf24_0a):
bash hpc/submit.sh hpc/registries/sf24_0a.json 0-7

# Any subset works, e.g. a few indices:
bash hpc/submit.sh hpc/registries/sf24_0a.json 1,3,5
```

Equivalent explicit form (set the name yourself):
`sbatch --job-name=rl_sf24_0a --array=0-7 hpc/submit_array.sh hpc/registries/sf24_0a.json`.

`sbatch` prints a `<jobid>`. All elements run concurrently (QOS allows up to 128/user; single-node jobs
share nodes, so no whole-node waste). `#SBATCH` defaults in `submit_array.sh`: `--cpus-per-task=16`,
`--time=28:00:00`, `--partition=rome`, no `--mem-per-cpu` (16 cores = the min shareable/billed slot;
`n_workers=16` uses it fully — see `hpc/registry_conventions.md`). **Walltime is 28h** — longer than the
24h training `time_budget` so the final evaluation completes and writes `eval_episodes.csv`; a run cut
off before that produces no eval file. (Optuna 0A is single-threaded — 16 cores is billed regardless,
so no change needed.)

---

## 5. Monitor

```bash
squeue -u $USER --format="%.18i %.20j %.8T %.10M"           # JobName column shows rl_sf24_0a etc.
sacct -j <jobid> --format=JobID,JobName%20,State,Elapsed,MaxRSS%12
tail -f "hpc/logs/rl_sf24_0a_<jobid>_2.out"                 # live log for task 2
grep -l "Resuming from checkpoint" hpc/logs/rl_sf24_0a_<jobid>_*.out   # which tasks resumed vs fresh
```

Per-task stdout/stderr: `hpc/logs/<jobname>_%A_%a.out` / `.err` (`%x`=job name, `%A`=array job id,
`%a`=task index). Per-run output: `results/exp0/<run_name>/` (config, checkpoints, `training_log.csv`,
`eval_episodes.csv`).

**Disconnect test:** after submitting, close your session and reconnect — `squeue -u $USER` still shows
the array. That's the whole point of dropping the HQ server.

---

## 6. Resume after a timeout or partial failure

Runs checkpoint periodically (default every 30 min) and auto-resume on the **same config** (keyed on
`config_hash`): resubmit the batch and each incomplete run continues from its latest checkpoint. To
skip finished ones, resubmit only the unfinished indices:

```bash
ls results/exp0/*/eval_episodes.csv 2>/dev/null            # finished runs
bash hpc/submit.sh hpc/registries/sf24_0a.json 3,5-7       # resubmit only what's left
```

No work is lost — resumed runs pick up their episode counter and elapsed time from the checkpoint.

---

## 7. Pull results back

On your **laptop**:

```bash
rsync -av <username>@snellius.surf.nl:~/Code_v2/results/exp0/ "results/exp0/"
```

Then analyze locally (paired-CRN comparison vs the tuned reactive/per-asset heuristic and the
clairvoyant lower bound).

---

## Quick reference

| Action | Command |
|---|---|
| Sync code up (laptop) | `rsync -av --exclude results/ … snellius:~/Code_v2/` |
| Inspect a registry | `python3 -c "import json;print(len(json.load(open('hpc/registries/sf24_0a.json'))))"` |
| Dry-run an entry | `python hpc/run_task.py --registry hpc/registries/sf24_0a.json --expe_id 0 --dry-run` |
| Build canary | `python -c "from experiments.configs import ExperimentConfig,build_experiment as b; b(ExperimentConfig.from_file('configs/sf24_optuna_perasset.json'))"` |
| Smoke one config | `bash hpc/submit.sh hpc/registries/sf24_0a.json 2` |
| Submit a batch | `bash hpc/submit.sh hpc/registries/sf24_0a.json 0-7` |
| Watch queue | `squeue -u $USER --format="%.18i %.20j %.8T %.10M"` |
| Per-element state | `sacct -j <jobid> --format=JobID,JobName%20,State,Elapsed` |
| Cancel | `scancel <jobid>` (or `scancel -u $USER`) |
| Pull results (laptop) | `rsync -av snellius:~/Code_v2/results/exp0/ results/exp0/` |
