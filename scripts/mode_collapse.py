from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.research_utils import compute_embeddings, compute_mode_collapse_score


PARSED_VS_PATH = Path("data/parsed_vs_ideas.json")
DIRECT_OUTPUTS_PATH = Path("data/direct_outputs.json")

RESPONSE_BLOCK_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)
TEXT_RE = re.compile(r"<text>(.*?)</text>", re.DOTALL | re.IGNORECASE)
IDEA_LINE_RE = re.compile(r"startup idea\s*:\s*(.+)", re.IGNORECASE)


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def extract_idea(text: str) -> str:
    for line in text.splitlines():
        match = IDEA_LINE_RE.search(line.strip())
        if match:
            return match.group(1).strip()
    return text.strip()


def parse_direct_output(raw_output: str) -> List[str]:
    ideas: List[str] = []
    blocks = RESPONSE_BLOCK_RE.findall(raw_output or "")
    for block in blocks:
        text_match = TEXT_RE.search(block)
        if text_match:
            idea = extract_idea(text_match.group(1))
            if idea:
                ideas.append(idea)
    if ideas:
        return ideas
    fallback = extract_idea(raw_output or "")
    return [fallback] if fallback else []


def get_vs_ideas(vs_items: Any) -> List[str]:
    if not isinstance(vs_items, list):
        return []
    ideas: List[str] = []
    for item in vs_items:
        if not isinstance(item, dict):
            continue
        idea = str(item.get("idea", "")).strip()
        if idea:
            ideas.append(idea)
    return ideas


def get_direct_ideas(direct_items: Any) -> List[str]:
    if not isinstance(direct_items, list):
        return []
    ideas: List[str] = []
    for item in direct_items:
        if not isinstance(item, dict):
            continue
        ideas.extend(parse_direct_output(str(item.get("raw_output", ""))))
    return [idea for idea in ideas if idea]


def compute_scores(direct_ideas: List[str], vs_ideas: List[str]) -> Dict[str, float]:
    direct_embeddings = compute_embeddings(direct_ideas)
    vs_embeddings = compute_embeddings(vs_ideas)

    return {
        "direct_prompt_mode_collapse_score": compute_mode_collapse_score(direct_embeddings),
        "verbalized_sampling_mode_collapse_score": compute_mode_collapse_score(vs_embeddings),
    }


def main() -> None:
    direct_items = load_json(DIRECT_OUTPUTS_PATH)
    vs_items = load_json(PARSED_VS_PATH)

    direct_ideas = get_direct_ideas(direct_items)
    vs_ideas = get_vs_ideas(vs_items)

    scores = compute_scores(direct_ideas, vs_ideas)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
