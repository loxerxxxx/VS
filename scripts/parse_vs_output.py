from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from parse_vs_responses import main as run_parser

    run_parser()


if __name__ == "__main__":
    main()
