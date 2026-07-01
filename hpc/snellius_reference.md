# Snellius Reference (offline)

Compiled scheduling/partition/accounting reference for Snellius (SURF), so we don't have to
re-derive or re-look-up these facts. Complements `hpc/snellius_manual.md` (the SLURM-array run
manual) and `hpc/registry_conventions.md` (named registries + job-name convention).

**Sources:** SURF User Knowledge Base (links at bottom) + **live `scontrol`/`sacctmgr` queries run
2026-06-17** on the `ttsei13069` account (raw output in the Appendix). Where the live cluster config
and the web docs disagree, the live query wins and the discrepancy is flagged.

---

## 1. Hardware / partitions

| Partition | CPU/node | Nodes | Notes |
|---|---|---|---|
| **rome** (= "thin") | **128** | 521 (`tcn[4-524]`) | **Default partition.** AMD Rome. `MaxCPUsPerNode=128`, `DefMemPerCPU=1792 MB` (~1.75 GiB/core → ~224 GiB/node usable, 251 GiB physical). Max walltime **5 days**. |
| **genoa** | **192** | — | AMD Genoa (2023). |
| fat / himem | 128/192 | — | High-memory nodes (himem 4 TB / 8 TB QOS exist). |
| gpu | — | — | GPU nodes (separate QOS). |
| staging | — | — | Data staging / transfers. |

A worker in our last run reported `cpus: 8x16 = 128, mem 251.67 GiB` → a full rome node.

---

## 2. Node sharing vs exclusive access  ⚠️ (corrected)

- **Single-node jobs SHARE nodes by default.** A job that fits on one node may request a *subset* of
  cores/memory, and SLURM co-locates multiple such jobs on the same node. Live confirmation on `rome`:
  `OverSubscribe=NO` (cores are not over-subscribed — each core goes to one job — but different jobs
  occupy different cores on the same node) and `ExclusiveUser=NO` (a node is not reserved to one user).
- **Multi-node jobs are ALWAYS exclusive.** Per SURF: "Jobs requesting more than 1 node will get
  exclusive access … independent of the amount of core/memory requested." (Enforced by SURF's
  job-submit policy, not visible in `scontrol`.)
- **You only get a whole node to yourself** by requesting all its cores (e.g. `--cpus-per-task=128`)
  or by passing `--exclusive`. Our array uses `--cpus-per-task=16` (a shared 1/8-node slot), so it
  never reserves a whole node.

**Implication:** an array of single-node tasks (`--cpus-per-task=16`) does **not** waste whole nodes —
SLURM packs up to 8 per node. The earlier "16× waste" worry was wrong; nodes are shared. (16, not 8,
because 16 cores is the minimum shareable slot — see §3.)

---

## 3. Billing / accounting  ✅ (confirmed by live `sbatch` notices, 2026-06-18)

- **Minimum billed slot = 16 cores (1/8 node).** A rome node "can be shared by up to **8 jobs**" →
  128/8 = **16 cores**. Requesting fewer (e.g. `--cpus-per-task=8`) still **allocates and bills a
  16-core slot** — the extra cores sit idle. So 16 is the practical floor; we set `n_workers=16` /
  `--cpus-per-task=16` to use the whole slot (see [registry_conventions.md](registry_conventions.md)).
- **Memory can push you into a higher tier.** Default is **1792 MiB/core**; the 16-core slot's default
  is ~28 GiB. With `--mem-per-cpu`, "all *allocated* CPUs are counted", so `--mem-per-cpu=2G` × 16 =
  32 GiB exceeds the slot and bills the **next tier (32 cores / 1.4 node)** — a silent 2× overcharge.
  **Do not set `--mem-per-cpu`** unless a config truly needs >28 GiB; the default keeps you at 16.
- Within a tier, `TRESBillingWeights = cpu=1.0` → linear per core. A full node = 128 cores, 229376 MiB,
  shareable by ≤8 jobs.
- Live confirmation: an `--cpus-per-task=8 --mem-per-cpu=2G` job printed *"You will be charged for 32
  CPUs"*; dropping `--mem-per-cpu` and using `--cpus-per-task=16` bills 16.

---

## 4. QOS & concurrency limits (account `ttsei13069`)

`sacctmgr show assoc user=$USER` → Account `ttsei13069`, QOS `normal`, **no `MaxJobs`/`MaxSubmit`** set
on the association.

`sacctmgr show qos` (relevant rows; columns = MaxJobsPU / MaxSubmitPU / MaxTRESPU):

| QOS | MaxJobsPU | Meaning |
|---|---|---|
| `normal` | — (none) | default; no explicit cap |
| `rome` | **128** | ≤128 running jobs/user on rome |
| `genoa` | **128** | ≤128 running jobs/user on genoa |
| `thin_ondemand` | 3 (MaxSubmit) | short/interactive |
| `maxjobs0` / `maxjobs1` | 0 / 1 | special caps |

**Bottom line: you can run up to ~128 concurrent jobs.** A 29-task array (`--array=0-28`) runs **fully
concurrent** — no QOS bottleneck. This removes the main reason to prefer HQ's block reservation for
*this* batch size.

---

## 5. Key job-script flags (rome defaults)

```bash
#SBATCH --partition=rome            # default anyway; rome = thin = 128 c/node
#SBATCH --cpus-per-task=16          # 16 = min shareable slot (1/8 node, ≤8 jobs/node) = min billed unit
                                    # → match n_workers=16; do NOT set --mem-per-cpu (default ~28 GiB
                                    #   keeps billing at 16 cores; an explicit 2G tips it to 32)
#SBATCH --time=24:00:00             # max 5 days on rome
#SBATCH --array=0-28%128            # %N caps concurrency; 128 is the QOS ceiling anyway
#SBATCH --exclusive                 # ONLY if you want the whole node to yourself
```
- DefaultTime is 5 min — always set `--time` (`submit_array.sh` sets 28h).
- Implemented in `hpc/submit_array.sh`: each array task runs one registry entry via
  `run_task.py --expe_id=$SLURM_ARRAY_TASK_ID --registry <file>`; `--cpus-per-task=16` matches
  `n_workers=16` (the min billed slot). `--array` is supplied on the CLI (via `hpc/submit.sh`),
  not baked into the script.

---

## 6. Implication for our HQ-vs-SLURM-array decision

| | HQ (current) | Native SLURM array |
|---|---|---|
| Packing | HQ packs 16×8c/node inside whole-node allocations | SLURM packs shared single-node jobs (no waste) |
| Concurrency for 29 tasks | guaranteed by block reservation | up to 128/user → all 29 at once |
| Disconnect safety | **server on login node = single point of failure** (killed our run) | **none — no daemon to keep alive** |
| Billing | whole nodes (used fully) | ~linear per core |
| Extra work | keep server alive (tmux / in-batch) | rewrite submit + resumption keying |

Given no QOS bottleneck and node sharing, the **SLURM array is a viable, disconnect-proof
alternative** whose only real costs are the resumption-keying rewrite (acceptable) and confirming the
billing increment. HQ remains fine **if** the server is moved off the login node (tmux on a pinned
`intN`, or run server+worker inside the batch job).

**Implemented:** `hpc/submit_array.sh` is the native-array path (no HQ server/worker). It runs one
**named-registry** entry per task via `run_task.py --expe_id=$SLURM_ARRAY_TASK_ID --registry <file>`;
dispatch a batch with `bash hpc/submit.sh hpc/registries/<name>.json <array>` (derives job name
`rl_<name>`, e.g. `bash hpc/submit.sh hpc/registries/sf15_0a.json 0-7`). Only `hq_task.sh` remains as
the deprecated HQ fallback — `submit.sh` is now the wrapper. Full workflow: `hpc/snellius_manual.md`.

---

## 7. Appendix — raw live query output (2026-06-17, int6)

```
$ sacctmgr show assoc user=$USER format=Account,QOS,MaxJobs,MaxSubmit
   Account                  QOS MaxJobs MaxSubmit
---------- -------------------- ------- ---------
ttsei13069               normal

$ sacctmgr show qos format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU
      Name MaxJobsPU MaxSubmitPU     MaxTRESPU
    normal
      rome       128
     genoa       128
 thin_onde+         3              (MaxSubmitPU)
  maxjobs0         0
  maxjobs1         1
   (… other special QOS omitted …)

$ scontrol show partition rome
PartitionName=rome  Default=YES  QoS=rome
   ExclusiveUser=NO  OverSubscribe=NO
   MaxNodes=UNLIMITED  MaxTime=5-00:00:00  MinNodes=1
   MaxCPUsPerNode=128  Nodes=tcn[4-524]
   TotalCPUs=66688  TotalNodes=521
   DefMemPerCPU=1792  MaxMemPerNode=UNLIMITED
   TRES=cpu=66688,mem=116704G,node=521,billing=66688
   TRESBillingWeights=gres/cpu=1.0,cpu=1.0
   DefaultTime=00:05:00
```

---

## Sources
- Snellius partitions and accounting — https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius+partitions+and+accounting
- Snellius partitions — https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius+partitions
- SLURM batch system — https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660221/SLURM+batch+system
- Writing a job script — https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660220/Writing+a+job+script
- Getting started: https://edu.nl/cq7yn · Filesystem/quota: https://edu.nl/7gvhk · Status/maintenance: https://edu.nl/brncf
