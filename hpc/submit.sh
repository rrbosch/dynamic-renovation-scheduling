#!/bin/bash
# Thin submit wrapper: derive a descriptive SLURM job name from the registry filename,
# then dispatch the array via hpc/submit_array.sh.
#
# Job-name convention: hpc/registries/<experiment>_<appendix>.json  ->  rl_<experiment>_<appendix>
#   (appendix optional, e.g. v2, sf15). Logs land at hpc/logs/rl_<...>_<jobid>_<task>.{out,err}.
#
# Usage:
#   bash hpc/submit.sh <registry.json> <array-spec>
#   e.g.  bash hpc/submit.sh hpc/registries/sf15_0a.json 0-7      # job rl_sf15_0a, tasks 0..7
#         bash hpc/submit.sh hpc/registries/sf15_0a.json 2        # single smoke task
#
# (Equivalent explicit form:
#    sbatch --job-name=rl_sf15_0a --array=0-7 hpc/submit_array.sh hpc/registries/sf15_0a.json )
#
# The HQ worker-spawn script that used to live here is deprecated — the SLURM job array
# (submit_array.sh) needs no login-node daemon. See git history for the old HyperQueue path.
set -e

reg="${1:?usage: bash hpc/submit.sh <registry.json> <array-spec>   e.g. hpc/registries/sf15_0a.json 0-7}"
array="${2:?missing array spec, e.g. 0-7 or 2 or 1,5,9}"

if [[ ! -f "$reg" ]]; then
    echo "ERROR: registry not found: $reg" >&2
    exit 1
fi

name="rl_$(basename "${reg%.json}")"
echo "Submitting: job-name=$name  registry=$reg  array=$array"
sbatch --job-name="$name" --array="$array" hpc/submit_array.sh "$reg"
