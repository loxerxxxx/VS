from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def run_step(command: List[str], *, env: dict[str, str]) -> None:
    print(f"\n>>> Running: {' '.join(command)}")
    subprocess.run(command, check=True, env=env)


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    cache_root = Path(".cache/huggingface").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    env.setdefault("HF_HOME", str(cache_root))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
    return env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VS startup bench pipeline end-to-end."
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip API generation and reuse existing JSON outputs.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip diversity metric computation.",
    )
    args = parser.parse_args()

    env = build_env()
    python = sys.executable

    if not args.skip_generate:
        run_step([python, "scripts/generate_ideas.py"], env=env)

    run_step([python, "scripts/parse_vs_output.py"], env=env)

    if not args.skip_metrics:
        run_step([python, "compute_diversity_metrics.py"], env=env)

    print("\nPipeline finished.")
    print("Launch dashboard with: streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
