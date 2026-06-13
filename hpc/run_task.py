"""HyperQueue entry point — called once per task by hq_task.sh.

Usage:
    python hpc/run_task.py --expe_id=$HQ_TASK_ID

Reads hpc/registry.json, looks up entry[expe_id], applies any seed/run_name
overrides, then runs the experiment with auto-resume enabled.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from experiments.run import _run_one  # noqa: E402
from utils.logging import enable_timestamped_stdout  # noqa: E402

_REGISTRY_PATH = _project_root / "hpc" / "registry.json"


def main() -> None:
    enable_timestamped_stdout()

    parser = argparse.ArgumentParser(description="Run one experiment from hpc/registry.json")
    parser.add_argument("--expe_id", type=int, required=True,
                        help="0-based index into hpc/registry.json (= $HQ_TASK_ID)")
    cli = parser.parse_args()

    with open(_REGISTRY_PATH) as f:
        registry = json.load(f)

    if cli.expe_id < 0 or cli.expe_id >= len(registry):
        print(f"ERROR: expe_id={cli.expe_id} out of range "
              f"(registry has {len(registry)} entries)")
        sys.exit(1)

    entry = registry[cli.expe_id]
    base_config_path = str(_project_root / entry["config"])
    seed_override = entry.get("seed")
    run_name_override = entry.get("run_name")

    print(f"[HPC] expe_id={cli.expe_id}  config={entry['config']}"
          f"  run_name={run_name_override}  seed={seed_override}")

    tmp_path = None
    try:
        if seed_override is not None or run_name_override is not None:
            # Write a patched config to a temp file so _run_one picks up overrides
            with open(base_config_path) as f:
                raw = json.load(f)
            if seed_override is not None:
                raw["seed"] = seed_override
            if run_name_override is not None:
                raw["run_name"] = run_name_override

            # Create temp file alongside the original so relative paths resolve correctly
            tmp_fd, tmp_path = tempfile.mkstemp(
                suffix=".json", dir=str(Path(base_config_path).parent)
            )
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(raw, f)
            config_path = tmp_path
        else:
            config_path = base_config_path

        # Build args namespace matching what _run_one expects
        args = SimpleNamespace(
            config=config_path,
            run_name=None,   # already embedded in config JSON if needed
            workers=None,
            resume=None,
        )
        _run_one(config_path, args, auto_resume=True)

    finally:
        if tmp_path and Path(tmp_path).exists():
            os.unlink(tmp_path)


if __name__ == "__main__":
    main()
