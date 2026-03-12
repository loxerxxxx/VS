from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from chart_helpers import (
    diversity_bar_chart,
    embedding_scatter_plot,
    probability_distribution_chart,
)

PARSED_VS_PATH = Path("data/parsed_vs_ideas.json")
METRICS_PATH = Path("data/diversity_metrics.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_data() -> tuple[pd.DataFrame, Dict]:
    vs_data = _load_json(PARSED_VS_PATH)
    metrics_data = _load_json(METRICS_PATH)

    if not isinstance(vs_data, list):
        raise ValueError("Expected a list in data/parsed_vs_ideas.json")
    if not isinstance(metrics_data, dict):
        raise ValueError("Expected an object in data/diversity_metrics.json")

    df = pd.DataFrame(vs_data)
    if df.empty:
        df = pd.DataFrame(columns=["topic", "idea", "probability"])

    for col in ["topic", "idea", "probability"]:
        if col not in df.columns:
            df[col] = np.nan

    df["topic"] = df["topic"].fillna("").astype(str)
    df["idea"] = df["idea"].fillna("").astype(str)
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    df = df[df["idea"].str.strip().astype(bool)].copy()
    return df, metrics_data


@st.cache_resource
def load_encoder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_data(show_spinner=False)
def build_embedding_frame(ideas: List[str]) -> pd.DataFrame:
    if not ideas:
        return pd.DataFrame(columns=["pc1", "pc2", "cluster"])

    encoder = load_encoder()
    embeddings = encoder.encode(ideas, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings)

    if len(embeddings) == 1:
        return pd.DataFrame({"pc1": [0.0], "pc2": [0.0], "cluster": [0]})

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    n_clusters = max(2, min(8, int(np.sqrt(len(embeddings)))))
    n_clusters = min(n_clusters, len(embeddings))
    if n_clusters < 2:
        clusters = np.zeros(len(embeddings), dtype=int)
    else:
        clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(
            embeddings
        )

    return pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1], "cluster": clusters})


@st.cache_data(show_spinner=False)
def compute_uniqueness_scores(ideas: List[str]) -> np.ndarray:
    if not ideas:
        return np.array([])
    if len(ideas) == 1:
        return np.array([1.0])

    encoder = load_encoder()
    embeddings = encoder.encode(ideas, normalize_embeddings=True, show_progress_bar=False)
    sim = cosine_similarity(np.asarray(embeddings))
    np.fill_diagonal(sim, -1.0)
    max_similarity = sim.max(axis=1)
    uniqueness = 1.0 - max_similarity
    return uniqueness


def diversity_comparison_frame(metrics: Dict) -> pd.DataFrame:
    vs = metrics.get("vs_prompting", {})
    direct = metrics.get("direct_prompting", {})
    return pd.DataFrame(
        [
            {
                "method": "VS prompting",
                "semantic_diversity": vs.get("semantic_diversity", 0.0),
                "distinct_1": vs.get("lexical_diversity", {}).get("distinct_1", 0.0),
                "distinct_2": vs.get("lexical_diversity", {}).get("distinct_2", 0.0),
                "concept_diversity_ratio": vs.get("concept_diversity", {}).get(
                    "concept_diversity_ratio", 0.0
                ),
            },
            {
                "method": "Direct prompting",
                "semantic_diversity": direct.get("semantic_diversity", 0.0),
                "distinct_1": direct.get("lexical_diversity", {}).get("distinct_1", 0.0),
                "distinct_2": direct.get("lexical_diversity", {}).get("distinct_2", 0.0),
                "concept_diversity_ratio": direct.get("concept_diversity", {}).get(
                    "concept_diversity_ratio", 0.0
                ),
            },
        ]
    )


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .metric-card {
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            border: 1px solid #374151;
            border-radius: 12px;
            padding: 0.8rem 1rem;
            color: #f9fafb;
        }
        .subtle {
            color: #6b7280;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="VS Startup Bench Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
    )
    apply_custom_style()

    st.title("VS Startup Bench Dashboard")
    st.caption("Experiment analytics for diversity in AI startup ideation")

    try:
        ideas_df, metrics = load_data()
    except Exception as exc:
        st.error(f"Failed to load experiment data: {exc}")
        st.stop()

    # Section 1: Overview
    st.markdown("## 1) Overview")
    st.write(
        "This experiment compares **Verbalized Sampling (VS) prompting** against **direct prompting** "
        "for generating AI-native startup ideas. The analysis tracks semantic, lexical, and concept "
        "diversity, then explores idea-level structure and long-tail novelty."
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Parsed VS ideas", f"{len(ideas_df)}")
    col_b.metric(
        "VS semantic diversity",
        f"{metrics.get('vs_prompting', {}).get('semantic_diversity', 0.0):.3f}",
    )
    col_c.metric(
        "Direct semantic diversity",
        f"{metrics.get('direct_prompting', {}).get('semantic_diversity', 0.0):.3f}",
    )

    # Section 2: Diversity Metrics
    st.markdown("## 2) Diversity Metrics")
    comparison_df = diversity_comparison_frame(metrics)
    fig_metrics = diversity_bar_chart(comparison_df)
    st.plotly_chart(fig_metrics, use_container_width=True)

    # Section 3: Startup Idea Explorer
    st.markdown("## 3) Startup Idea Explorer")
    topics = sorted(t for t in ideas_df["topic"].dropna().unique() if t)
    selected_topics = st.multiselect(
        "Filter by topic",
        options=topics,
        default=topics,
        placeholder="Select one or more topics",
    )
    query = st.text_input("Search idea text", placeholder="e.g. diagnostics, agentic, compliance")

    filtered = ideas_df.copy()
    if selected_topics:
        filtered = filtered[filtered["topic"].isin(selected_topics)]
    if query.strip():
        filtered = filtered[filtered["idea"].str.contains(query, case=False, na=False)]

    st.dataframe(
        filtered.sort_values(["topic", "probability"], ascending=[True, True]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Showing {len(filtered)} ideas.")
    prob_fig = probability_distribution_chart(
        filtered.assign(method="VS prompting"),
        title="Probability Distribution (Filtered VS Ideas)",
    )
    st.plotly_chart(prob_fig, use_container_width=True)

    # Section 4: Embedding Visualization
    st.markdown("## 4) Embedding Visualization")
    st.markdown(
        "<p class='subtle'>PCA projection over sentence-transformer embeddings with KMeans cluster labels.</p>",
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.info("No ideas available for embedding visualization with current filters.")
    else:
        emb_df = build_embedding_frame(filtered["idea"].tolist())
        plot_df = filtered.reset_index(drop=True).join(emb_df)
        fig_scatter = embedding_scatter_plot(
            plot_df, hover_cols=["topic", "idea", "probability"]
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Section 5: Long Tail Ideas
    st.markdown("## 5) Long Tail Ideas")
    st.write(
        "Ideas below are ranked by **uniqueness score** (`1 - max cosine similarity` to other ideas), "
        "highlighting concepts farthest from the dense center of idea space."
    )

    if ideas_df.empty:
        st.info("No ideas available for long-tail analysis.")
    else:
        uniqueness = compute_uniqueness_scores(ideas_df["idea"].tolist())
        long_tail = ideas_df.copy()
        long_tail["uniqueness_score"] = uniqueness
        long_tail = long_tail.sort_values("uniqueness_score", ascending=False)

        top_k = st.slider("Number of long-tail ideas", min_value=5, max_value=30, value=10, step=1)
        st.dataframe(
            long_tail.head(top_k)[["topic", "idea", "probability", "uniqueness_score"]],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
