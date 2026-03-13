from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _configure_local_hf_cache() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cache_root = project_root / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_root / "sentence_transformers"))
    cache_root.mkdir(parents=True, exist_ok=True)


_configure_local_hf_cache()
_ENCODER = None


def _encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer

        _ENCODER = SentenceTransformer(EMBEDDING_MODEL)
    return _ENCODER


def compute_embeddings(ideas: Sequence[str]) -> np.ndarray:
    if not ideas:
        return np.zeros((0, 384), dtype=float)
    vectors = _encoder().encode(list(ideas), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=float)


def mode_collapse_score(embeddings: np.ndarray) -> float:
    if embeddings.shape[0] < 2:
        return 0.0
    sims = cosine_similarity(embeddings)
    iu = np.triu_indices_from(sims, k=1)
    if len(iu[0]) == 0:
        return 0.0
    return float(np.mean(sims[iu]))


def get_long_tail_ideas(
    ideas: Sequence[str],
    embeddings: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    if len(ideas) == 0 or embeddings.shape[0] == 0:
        return pd.DataFrame(columns=["idea", "distance"])
    centroid = embeddings.mean(axis=0, keepdims=True)
    centroid = centroid / np.maximum(np.linalg.norm(centroid, axis=1, keepdims=True), 1e-12)
    distances = 1.0 - cosine_similarity(embeddings, centroid).reshape(-1)
    df = pd.DataFrame({"idea": list(ideas), "distance": distances})
    return df.sort_values("distance", ascending=False).head(top_k).reset_index(drop=True)


def plot_idea_map(
    ideas: Sequence[str],
    embeddings: np.ndarray,
    *,
    title: str = "Startup Idea Universe",
):
    if len(ideas) == 0:
        return px.scatter(title=title)

    if embeddings.shape[0] == 1:
        df = pd.DataFrame({"x": [0.0], "y": [0.0], "cluster": ["0"], "idea": list(ideas)})
        return px.scatter(df, x="x", y="y", color="cluster", hover_data=["idea"], title=title)

    coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    k = min(5, len(ideas))
    if k <= 1:
        clusters = np.zeros(len(ideas), dtype=int)
    else:
        clusters = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings)

    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cluster": clusters.astype(str),
            "idea": list(ideas),
        }
    )
    return px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        hover_data=["idea"],
        title=title,
        opacity=0.85,
    )
