from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


INPUT_PATH = Path("data/vs_outputs.json")
OUTPUT_PATH = Path("data/parsed_vs_ideas.json")

RESPONSE_BLOCK_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)
TEXT_RE = re.compile(r"<text>(.*?)</text>", re.DOTALL | re.IGNORECASE)
PROBABILITY_RE = re.compile(
    r"<probability>\s*([0-9]*\.?[0-9]+)\s*</probability>", re.IGNORECASE
)
IDEA_LINE_RE = re.compile(r"startup idea\s*:\s*(.+)", re.IGNORECASE)


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_idea(text_block: str) -> str:
    cleaned = text_block.strip()
    for line in cleaned.splitlines():
        match = IDEA_LINE_RE.search(line.strip())
        if match:
            return match.group(1).strip()
    return cleaned


def parse_response_block(block: str) -> Optional[Dict[str, Any]]:
    text_match = TEXT_RE.search(block)
    prob_match = PROBABILITY_RE.search(block)

    if not text_match or not prob_match:
        return None

    text_value = text_match.group(1).strip()
    probability_raw = prob_match.group(1).strip()

    try:
        probability = float(probability_raw)
    except ValueError:
        return None

    if probability < 0:
        return None

    idea = extract_idea(text_value)
    if not idea:
        return None

    return {
        "idea": idea,
        "probability": probability,
    }


def parse_vs_outputs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    skipped_blocks = 0

    for item in items:
        topic = str(item.get("topic", "")).strip()
        raw_output = str(item.get("raw_output", ""))
        if not topic or not raw_output:
            continue

        blocks = RESPONSE_BLOCK_RE.findall(raw_output)
        if not blocks:
            skipped_blocks += 1
            continue

        for block in blocks:
            parsed_block = parse_response_block(block)
            if not parsed_block:
                skipped_blocks += 1
                continue

            parsed.append(
                {
                    "topic": topic,
                    "idea": parsed_block["idea"],
                    "probability": parsed_block["probability"],
                }
            )

    if skipped_blocks:
        print(f"Skipped {skipped_blocks} malformed response block(s).")

    return parsed


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    raw_items = load_json(INPUT_PATH)
    if not isinstance(raw_items, list):
        raise ValueError("Expected a JSON array in data/vs_outputs.json")

    parsed = parse_vs_outputs(raw_items)
    save_json(OUTPUT_PATH, parsed)
    print(f"Saved {len(parsed)} parsed ideas to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
