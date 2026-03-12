from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


PARSED_VS_PATH = Path("data/parsed_vs_ideas.json")
DIRECT_OUTPUTS_PATH = Path("data/direct_outputs.json")
OUTPUT_PATH = Path("data/diversity_metrics.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

RESPONSE_BLOCK_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)
TEXT_RE = re.compile(r"<text>(.*?)</text>", re.DOTALL | re.IGNORECASE)
IDEA_LINE_RE = re.compile(r"startup idea\s*:\s*(.+)", re.IGNORECASE)
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_idea_from_text(text: str) -> str:
    stripped = text.strip()
    for line in stripped.splitlines():
        match = IDEA_LINE_RE.search(line.strip())
        if match:
            return match.group(1).strip()
    return stripped


def parse_direct_item(raw_output: str) -> List[str]:
    ideas: List[str] = []
    content = raw_output.strip()
    if not content:
        return ideas

    blocks = RESPONSE_BLOCK_RE.findall(content)
    if blocks:
        for block in blocks:
            text_match = TEXT_RE.search(block)
            if text_match:
                idea = extract_idea_from_text(text_match.group(1))
                if idea:
                    ideas.append(idea)
        if ideas:
            return ideas

    for line in content.splitlines():
        match = IDEA_LINE_RE.search(line.strip())
        if match:
            idea = match.group(1).strip()
            if idea:
                ideas.append(idea)
    if ideas:
        return ideas

    return [content]


def normalize_vs_ideas(raw_items: Any) -> List[str]:
    if not isinstance(raw_items, list):
        raise ValueError("Expected a JSON array for parsed_vs_ideas.json")
    ideas: List[str] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        idea = str(item.get("idea", "")).strip()
        if idea:
            ideas.append(idea)
    return ideas


def normalize_direct_ideas(raw_items: Any) -> List[str]:
    if not isinstance(raw_items, list):
        raise ValueError("Expected a JSON array for direct_outputs.json")
    ideas: List[str] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        raw_output = str(item.get("raw_output", ""))
        ideas.extend(parse_direct_item(raw_output))
    return [idea for idea in ideas if idea.strip()]


def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def distinct_n(texts: Iterable[str], n: int) -> float:
    ngrams = set()
    total = 0
    for text in texts:
        tokens = tokenize(text)
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ngrams.add(tuple(tokens[i : i + n]))
            total += 1
    if total == 0:
        return 0.0
    return len(ngrams) / total


def semantic_diversity(embeddings: np.ndarray) -> float:
    n = len(embeddings)
    if n < 2:
        return 0.0
    sim_matrix = cosine_similarity(embeddings)
    upper = sim_matrix[np.triu_indices(n, k=1)]
    avg_similarity = float(np.mean(upper)) if upper.size else 0.0
    return 1.0 - avg_similarity


def concept_diversity(embeddings: np.ndarray) -> Dict[str, Any]:
    n = len(embeddings)
    if n == 0:
        return {
            "n_clusters": 0,
            "cluster_distribution": {},
            "concept_diversity_ratio": 0.0,
        }

    # Heuristic: small but meaningful number of clusters.
    n_clusters = max(2, int(np.sqrt(n)))
    n_clusters = min(n_clusters, 10, n)
    if n_clusters < 2:
        n_clusters = 1

    if n_clusters == 1:
        return {
            "n_clusters": 1,
            "cluster_distribution": {"0": n},
            "concept_diversity_ratio": 1.0,
        }

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(embeddings)
    unique, counts = np.unique(labels, return_counts=True)
    cluster_distribution = {str(int(k)): int(v) for k, v in zip(unique, counts)}

    nonempty_clusters = int(np.count_nonzero(counts))
    concept_diversity_ratio = nonempty_clusters / n_clusters
    return {
        "n_clusters": n_clusters,
        "cluster_distribution": cluster_distribution,
        "concept_diversity_ratio": concept_diversity_ratio,
    }


def compute_metrics(ideas: List[str], encoder: SentenceTransformer) -> Dict[str, Any]:
    if not ideas:
        return {
            "num_ideas": 0,
            "semantic_diversity": 0.0,
            "lexical_diversity": {"distinct_1": 0.0, "distinct_2": 0.0},
            "concept_diversity": {
                "n_clusters": 0,
                "cluster_distribution": {},
                "concept_diversity_ratio": 0.0,
            },
        }

    embeddings = encoder.encode(ideas, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings)

    return {
        "num_ideas": len(ideas),
        "semantic_diversity": semantic_diversity(embeddings),
        "lexical_diversity": {
            "distinct_1": distinct_n(ideas, n=1),
            "distinct_2": distinct_n(ideas, n=2),
        },
        "concept_diversity": concept_diversity(embeddings),
    }


def print_summary_table(vs_metrics: Dict[str, Any], direct_metrics: Dict[str, Any]) -> None:
    rows = [
        {
            "method": "VS prompting",
            "num_ideas": vs_metrics["num_ideas"],
            "semantic_diversity": round(vs_metrics["semantic_diversity"], 4),
            "distinct_1": round(vs_metrics["lexical_diversity"]["distinct_1"], 4),
            "distinct_2": round(vs_metrics["lexical_diversity"]["distinct_2"], 4),
            "n_clusters": vs_metrics["concept_diversity"]["n_clusters"],
            "concept_diversity_ratio": round(
                vs_metrics["concept_diversity"]["concept_diversity_ratio"], 4
            ),
        },
        {
            "method": "Direct prompting",
            "num_ideas": direct_metrics["num_ideas"],
            "semantic_diversity": round(direct_metrics["semantic_diversity"], 4),
            "distinct_1": round(direct_metrics["lexical_diversity"]["distinct_1"], 4),
            "distinct_2": round(direct_metrics["lexical_diversity"]["distinct_2"], 4),
            "n_clusters": direct_metrics["concept_diversity"]["n_clusters"],
            "concept_diversity_ratio": round(
                direct_metrics["concept_diversity"]["concept_diversity_ratio"], 4
            ),
        },
    ]
    df = pd.DataFrame(rows)
    print("\nDiversity metrics comparison")
    print(df.to_string(index=False))


def save_metrics(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def configure_local_hf_cache() -> None:
    """Ensure model downloads can write within project workspace."""
    project_cache = Path(".cache/huggingface").resolve()
    project_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(project_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(project_cache / "hub"))


def main() -> None:
    configure_local_hf_cache()
    parsed_vs_raw = load_json(PARSED_VS_PATH)
    direct_raw = load_json(DIRECT_OUTPUTS_PATH)

    vs_ideas = normalize_vs_ideas(parsed_vs_raw)
    direct_ideas = normalize_direct_ideas(direct_raw)

    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    vs_metrics = compute_metrics(vs_ideas, encoder)
    direct_metrics = compute_metrics(direct_ideas, encoder)

    result = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "vs_prompting": vs_metrics,
        "direct_prompting": direct_metrics,
    }

    save_metrics(OUTPUT_PATH, result)
    print_summary_table(vs_metrics, direct_metrics)
    print(f"\nSaved metrics to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
