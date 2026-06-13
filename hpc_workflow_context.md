# HPC Workflow Documentation: `python_restructured/` Project

> **Note for LLM:** This document describes how the `python_restructured/` project on Snellius (SURF national HPC) is structured for running batched experiments. Use this as a reference when adapting the workflow to a new experiment setup. Do not assume this is a generic HPC template — the specifics below are particular to this project.

---

## Infrastructure Stack

- **Cluster:** Snellius (SURF, Netherlands)
- **Job scheduler:** SLURM
- **Task scheduler:** HyperQueue (HQ) v0.19.0, running on top of SLURM
- **Python environment:** `Python/3.11.3-GCCcore-12.3.0`, activated from a venv at `~/.local/venv/`
- **Software environment module:** `2023` (must be loaded before other modules)

HyperQueue is used as a meta-scheduler: SLURM allocates compute nodes, HQ distributes individual tasks across those nodes. This avoids SLURM overhead for large arrays of short-to-medium jobs.

---

## Project Directory

All relevant files live in `~/python_restructured/` on the login node. The key files for the HPC workflow are:

| File | Role |
|---|---|
| `to_do_experiments.json` | Registry of all experiments to run |
| `hq_task.sh` | Shell script executed once per HQ task |
| `hyperq_job_hpc.sh` | SLURM job script that spawns HQ workers |
| `run_on_hpc.py` | Python entry point, called by `hq_task.sh` |

---

## Experiment Registry: `to_do_experiments.json`

This JSON file is a **flat list** of experiment configurations. Each entry corresponds to exactly one experiment run. Example entry:

```json
{
    "experiment_name": "Population size experiment",
    "parameters": {
        "pop_size": 30,
        "evaluator": "StandardEvaluator",
        "algo_seed": 0
    }
}
```

The list index (0-based) is the experiment ID. HyperQueue passes the task index via `$HQ_TASK_ID`, which `run_on_hpc.py` uses to look up the corresponding entry in this JSON.

**To count the number of experiments (= number of HQ tasks needed):**
```bash
python3 -c "import json; print(len(json.load(open('to_do_experiments.json'))))"
```

When adapting to a new experiment: regenerate `to_do_experiments.json` with your new parameter combinations and seeds, keeping the same flat-list structure.

---

## Task Script: `hq_task.sh`

Executed once per experiment by HyperQueue. It:
1. Loads the software environment and Python module
2. Activates the virtualenv at `~/.local/venv/`
3. Calls `run_on_hpc.py` with the task index as `--expe_id`

```bash
#!/bin/bash
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source ~/.local/venv/bin/activate
python3 run_on_hpc.py --expe_id=$HQ_TASK_ID
```

> **Known issue in current file:** the script calls `python3.9` but loads `Python/3.11.3`. This is a bug — use `python3` or `python3.11` to match the loaded module.

`$HQ_TASK_ID` is an integer in the range `[0, N-1]` where N is the total number of tasks submitted. `run_on_hpc.py` uses this integer as an index into `to_do_experiments.json`.

When adapting: only change the `python3` call if your entry point script has a different name or argument interface.

---

## Python Entry Point: `run_on_hpc.py`

Accepts `--expe_id` as a command-line argument. Internally, it loads `to_do_experiments.json`, indexes into it at position `expe_id`, and runs the experiment with those parameters.

When adapting: update `run_on_hpc.py` to handle whatever parameters your new `to_do_experiments.json` contains.

---

## SLURM/HQ Job Script: `hyperq_job_hpc.sh`

Submitted via `sbatch`. It allocates compute nodes and starts HQ workers on them. Current configuration:

```
--nodes=1
--tasks-per-node=15    # 15 parallel HQ workers on the node
--time=24:00:00        # 24-hour walltime
```

The script also copies a SQLite database (`traffic_results.db`) to `$TMPDIR` (fast local scratch) before starting workers — this is specific to the current project's traffic assignment component.

```bash
cp "$HOME/python_restructured/Environments/input/Sioux Falls Expanded/traffic_results.db" "$TMPDIR"
```

When adapting: adjust `--nodes`, `--tasks-per-node`, and `--time` based on the expected runtime per task and total number of tasks. Remove or replace the `cp` line if your experiment does not need this database.

---

## Full Execution Procedure

Run from the login node after SSH-ing into Snellius:

```bash
# 1. Navigate to project
cd python_restructured/

# 2. Load modules
module load 2023
module load HyperQueue/0.19.0

# 3. Start HQ server (skip if already running)
hq server info             # check status
nohup hq server start &    # start if not running

# 4. Count experiments
python3 -c "import json; print(len(json.load(open('to_do_experiments.json'))))"
# e.g. 210 → array index is 0-209

# 5. Submit task array to HQ queue
hq submit --array 0-209 --pin taskset --cpus=1 hq_task.sh

# 6. Submit SLURM job to spawn workers
sbatch hyperq_job_hpc.sh

# 7. Monitor
hq job list
hq job progress <job_id>
hq task list <job_id> | grep "WAITING" | wc -l
```

---

## Monitoring & Cleanup

```bash
hq job list --all                     # show all jobs (including finished)
hq job cancel all                     # cancel all pending/running tasks
hq task list <job_id> | grep "FAILED" # inspect failures
hq server stop                        # shut down server when done
```

SLURM output logs are written to `slurm_out/slurmout_<jobid>.out` and `.errarray`.

---

## Adapting to a New Experiment: Checklist

1. **Regenerate `to_do_experiments.json`** with new parameter combinations (keep flat list structure, 0-indexed)
2. **Update `run_on_hpc.py`** to parse and use the new parameter fields
3. **Update `hq_task.sh`** if the Python entry point name or arguments change; fix the `python3.9` → `python3` bug if not already done
4. **Update `hyperq_job_hpc.sh`** — adjust walltime, node count, and remove/replace the `traffic_results.db` copy if not needed
5. **Recount experiments** and update the `--array 0-N` argument accordingly
6. Re-run the full execution procedure above
