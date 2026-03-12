from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    # Allow importing the project-level script from /scripts.
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from generate_startup_ideas import main as run_generator

    run_generator()


if __name__ == "__main__":
    main()
