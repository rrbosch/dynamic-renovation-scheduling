# Experiment Setup — Computational Budget

## Decisions

| Parameter | Value | Rationale |
|---|---|---|
| Independent repetitions (seeds) | **5** | Minimum for defensible statistics; see literature below |
| Workers per run (`n_workers`) | **8** | Covers parallel episode collection + evaluation without over-subscribing |
| Wallclock time per run | **24 h** | Safe upper bound for all agent classes including MonteCarloRolloutAgent |

---

## Literature Basis

### Seed count

Agarwal et al. (2021), **"Deep Reinforcement Learning at the Edge of the Statistical Precipice"**
(NeurIPS 2021 Outstanding Paper) shows that point estimates (mean/median) over 1–3 seeds are
statistically unreliable. Their main recommendation:

> *"3–10 runs per task is practical when using robust aggregate metrics
> (interquartile mean, stratified bootstrap confidence intervals)."*

**5 seeds** is the de-facto minimum in recent OR/scheduling RL papers and is what most reviewers
expect. Final results should be reported using the `rliable` library:
stratified bootstrap CIs + performance profiles rather than plain mean ± std.

- Paper: https://arxiv.org/abs/2108.13264
- Library: https://github.com/google-research/rliable

### Worker count and walltime

The Open RL Benchmark (NeurIPS 2024) tracks 25 000+ CPU-only RL runs and finds 4–16 parallel
workers per run typical for numpy/compiled-backend agents. 8 workers balances episode throughput
against node sharing overhead on Snellius.

Walltime norms from OR papers on infrastructure/network scheduling: 4–8 h for DQN/ADP,
up to 24 h for Monte Carlo rollout agents. 24 h is chosen as a single safe upper bound so all
agent types share one SLURM configuration.

---

## Mapping to Code

### `n_workers` in config JSON

Set `"training": { "n_workers": 8 }` in every config file before running on Snellius.
On laptop (development), keep `"n_workers": 1` to avoid multiprocessing overhead.

### Seeds via `generate_registry.py`

```bash
python hpc/generate_registry.py --configs configs/exp1_*.json --seeds 0 1 2 3 4
```

This produces one registry entry per (config × seed), with auto-named run directories
(`<base>_s0` … `<base>_s4`). The seed is injected into the config at task dispatch time
by `run_task.py` — no separate config files needed per seed.

---

## Snellius Node Sizing

Snellius standard nodes have **128 cores** (AMD GENOA/ROME).

```
128 cores / 8 workers per run = 16 concurrent runs per node
```

With 5 seeds and N algorithms:

| Algorithms | Total runs | Nodes needed (24 h slot) |
|---|---|---|
| 5 | 25 | 2 |
| 10 | 50 | 4 |
| 16 | 80 | 5 |

### Recommended `submit.sh` flags for a full experiment batch

```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --time=24:00:00
```

```bash
# HQ task submission (update N to match generate_registry.py output)
hq submit --array 0-{N-1} --pin taskset --cpus=8 hpc/hq_task.sh
```

Note: `--cpus=8` tells HQ to reserve 8 cores per task, matching `n_workers=8`.
