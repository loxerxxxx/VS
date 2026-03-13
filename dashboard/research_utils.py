from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

CACHE_ROOT = os.path.abspath(".cache/huggingface")
os.makedirs(CACHE_ROOT, exist_ok=True)
os.environ["HF_HOME"] = CACHE_ROOT
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_ROOT, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_ROOT, "transformers")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(CACHE_ROOT, "sentence_transformers")

from sentence_transformers import SentenceTransformer


def _configure_local_hf_cache() -> None:
    os.makedirs(CACHE_ROOT, exist_ok=True)
    os.environ["HF_HOME"] = CACHE_ROOT
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_ROOT, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_ROOT, "transformers")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(CACHE_ROOT, "sentence_transformers")


def compute_embeddings(
    texts: Iterable[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    text_list = [str(t).strip() for t in texts if str(t).strip()]
    if not text_list:
        return np.empty((0, 0), dtype=float)
    _configure_local_hf_cache()
    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(
        text_list,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=float)


def compute_mode_collapse_score(embeddings: np.ndarray) -> float:
    if embeddings.size == 0 or len(embeddings) < 2:
        return 0.0
    sim_matrix = cosine_similarity(embeddings)
    upper = sim_matrix[np.triu_indices(len(embeddings), k=1)]
    if upper.size == 0:
        return 0.0
    return float(np.mean(upper))


def compute_long_tail_ideas(
    ideas_df: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int = 10,
) -> pd.DataFrame:
    if ideas_df.empty or embeddings.size == 0:
        return pd.DataFrame(columns=list(ideas_df.columns) + ["distance_from_centroid"])

    centroid = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)

    out = ideas_df.reset_index(drop=True).copy()
    out["distance_from_centroid"] = distances
    out = out.sort_values("distance_from_centroid", ascending=False).head(top_k)
    return out


def plot_embedding_map(
    ideas_df: pd.DataFrame,
    embeddings: np.ndarray,
):
    if ideas_df.empty or embeddings.size == 0:
        return px.scatter(title="Idea Landscape Map (no data)")

    if len(embeddings) == 1:
        plot_df = ideas_df.reset_index(drop=True).copy()
        plot_df["pc1"] = 0.0
        plot_df["pc2"] = 0.0
    else:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embeddings)
        plot_df = ideas_df.reset_index(drop=True).copy()
        plot_df["pc1"] = coords[:, 0]
        plot_df["pc2"] = coords[:, 1]

    color_col = "prompt_type" if "prompt_type" in plot_df.columns else None
    fig = px.scatter(
        plot_df,
        x="pc1",
        y="pc2",
        color=color_col,
        hover_data=[c for c in ["topic", "idea", "probability", "prompt_type"] if c in plot_df.columns],
        title="Idea Landscape Map",
    )
    fig.update_layout(height=520, xaxis_title="PC1", yaxis_title="PC2")
    return fig
