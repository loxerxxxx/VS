from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from openai import APIConnectionError, APIError, NotFoundError, OpenAI, RateLimitError
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from chart_helpers import (
    diversity_bar_chart,
    embedding_scatter_plot,
    probability_distribution_chart,
)
from dashboard.research_utils import (
    compute_embeddings,
    get_long_tail_ideas,
    mode_collapse_score,
    plot_idea_map,
)

PARSED_VS_PATH = Path("data/parsed_vs_ideas.json")
METRICS_PATH = Path("data/diversity_metrics.json")
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
FREE_MODEL_FALLBACKS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-3-4b-it:free",
    "qwen/qwen3-4b:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
]
KIMI_MODEL = "moonshotai/kimi-k2"

RESPONSE_BLOCK_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)
TEXT_RE = re.compile(r"<text>(.*?)</text>", re.DOTALL | re.IGNORECASE)


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _get_secret_or_env(name: str) -> str | None:
    if name in st.secrets:
        value = str(st.secrets[name]).strip()
        return value if value else None
    value = os.getenv(name)
    return value.strip() if value else None


def _extract_content(completion: object) -> str:
    choices = getattr(completion, "choices", None)
    if not choices:
        raise RuntimeError("Model returned no choices.")
    msg = getattr(choices[0], "message", None)
    if msg is None:
        raise RuntimeError("Model response message was empty.")
    content = getattr(msg, "content", "")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, dict):
                out.append(str(item.get("text", "")))
            else:
                out.append(str(getattr(item, "text", "")))
        joined = "\n".join(x for x in out if x).strip()
        if joined:
            return joined
    raise RuntimeError("Model returned empty content.")


def _model_candidates(model: str) -> List[str]:
    configured_fallbacks = _get_secret_or_env("OPENAI_MODEL_FALLBACKS")
    models: List[str] = [model.strip()]
    if configured_fallbacks:
        models.extend(x.strip() for x in configured_fallbacks.split(",") if x.strip())
    models.extend(FREE_MODEL_FALLBACKS)
    deduped: List[str] = []
    seen = set()
    for m in models:
        if m and m not in seen:
            deduped.append(m)
            seen.add(m)
    return deduped


def _client() -> OpenAI:
    api_key = _get_secret_or_env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit secrets.")
    base_url = _get_secret_or_env("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    return OpenAI(api_key=api_key, base_url=base_url)


def _chat_with_fallback(model: str, system: str, prompt: str) -> str:
    client = _client()
    last_error: Exception | None = None
    for candidate in _model_candidates(model):
        for attempt in range(2):
            try:
                completion = client.chat.completions.create(
                    model=candidate,
                    temperature=0.9,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                return _extract_content(completion)
            except NotFoundError as exc:
                last_error = exc
                break
            except (RateLimitError, APIConnectionError, APIError) as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                break
    raise RuntimeError(f"All model attempts failed. Last error: {last_error}")


def _parse_ideas(raw: str) -> List[str]:
    ideas: List[str] = []
    blocks = RESPONSE_BLOCK_RE.findall(raw)
    if blocks:
        for block in blocks:
            match = TEXT_RE.search(block)
            if match:
                txt = match.group(1).strip()
                if txt:
                    ideas.append(txt)
    if ideas:
        return ideas
    lines = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
    return lines[:10]


def generate_ideas(topic: str, mode: str, model: str, n: int = 10) -> tuple[List[str], str]:
    if mode == "verbalized_sampling":
        prompt = (
            f"Think of 20 possible startup ideas for {topic}. Internally consider the probability "
            "distribution and sample from unusual/long-tail regions. Return exactly "
            f"{n} diverse ideas in this XML format only:\n"
            "<response><text>...</text><probability>0.01</probability></response>"
        )
        system = "You generate unconventional but plausible startup ideas."
    else:
        prompt = (
            f"Generate {n} startup ideas and go-to-market strategies for: {topic}. "
            "Return exactly this XML format only:\n"
            "<response><text>...</text><probability>0.01</probability></response>"
        )
        system = "You generate clear practical startup ideas."
    raw = _chat_with_fallback(model, system, prompt)
    return _parse_ideas(raw), raw


@st.cache_data
def load_data() -> tuple[pd.DataFrame, dict]:
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


@st.cache_data(show_spinner=False)
def build_embedding_frame(ideas: List[str]) -> pd.DataFrame:
    if not ideas:
        return pd.DataFrame(columns=["pc1", "pc2", "cluster"])
    embeddings = compute_embeddings(ideas)
    if len(embeddings) == 1:
        return pd.DataFrame({"pc1": [0.0], "pc2": [0.0], "cluster": [0]})

    coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
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
    embeddings = compute_embeddings(ideas)
    sim = cosine_similarity(np.asarray(embeddings))
    np.fill_diagonal(sim, -1.0)
    max_similarity = sim.max(axis=1)
    return 1.0 - max_similarity


def diversity_comparison_frame(metrics: dict) -> pd.DataFrame:
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
        .stApp { background: #050a18; color: #e5e7eb; }
        .hero-title { font-size: 2.3rem; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }
        .hero-sub { color: #a6b0c4; margin-top: -8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="VS Startup Bench", page_icon=":rocket:", layout="wide")
    apply_custom_style()
    st.markdown('<div class="hero-title">VS STARTUP BENCH DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">BY A DESPERATE PSYCHOPATH AKA PRINCE</div>', unsafe_allow_html=True)

    mode_ui = st.sidebar.selectbox("Mode", ["Brainstorm Mode", "Experiment Mode"])
    preset = st.sidebar.selectbox(
        "Model preset",
        [
            "NVIDIA free (recommended)",
            "Kimi (requires your access)",
            "Custom model",
        ],
    )
    if preset == "NVIDIA free (recommended)":
        model_default = DEFAULT_MODEL
    elif preset == "Kimi (requires your access)":
        model_default = KIMI_MODEL
    else:
        model_default = _get_secret_or_env("OPENAI_MODEL") or DEFAULT_MODEL

    st.markdown("## Live Experiment Generator")
    with st.form("generate_form"):
        topic = st.text_input("Topic", value="AI healthcare")
        n_ideas = st.slider("Ideas per method", min_value=5, max_value=15, value=10)
        model = st.text_input("Model", value=model_default)
        submitted = st.form_submit_button("Generate")

    if not submitted:
        st.info("Select mode, enter topic, and click Generate.")
        return
    if not topic.strip():
        st.warning("Topic is required.")
        return

    with st.spinner("Generating ideas..."):
        try:
            direct_ideas, direct_raw = generate_ideas(topic, "direct", model, n=n_ideas)
            vs_ideas: List[str] = []
            vs_raw = ""
            if mode_ui == "Experiment Mode":
                vs_ideas, vs_raw = generate_ideas(
                    topic, "verbalized_sampling", model, n=n_ideas
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Generation failed: {exc}")
            st.caption(
                "Tip: NVIDIA free models are available now. Kimi may not be free/available on every account."
            )
            return

    if mode_ui == "Brainstorm Mode":
        st.success(f"Generated {len(direct_ideas)} ideas.")
        st.dataframe(pd.DataFrame({"idea": direct_ideas}), use_container_width=True, hide_index=True)
        emb = compute_embeddings(direct_ideas)
        long_tail = get_long_tail_ideas(direct_ideas, emb, top_k=5)
        st.markdown("### 🔥 Long Tail Ideas Discovered")
        st.dataframe(long_tail, use_container_width=True, hide_index=True)
        st.markdown("### 🌌 Startup Idea Universe")
        st.plotly_chart(plot_idea_map(direct_ideas, emb), use_container_width=True)
    else:
        direct_emb = compute_embeddings(direct_ideas)
        vs_emb = compute_embeddings(vs_ideas)
        d_score = mode_collapse_score(direct_emb)
        v_score = mode_collapse_score(vs_emb)
        c1, c2 = st.columns(2)
        c1.metric("Direct Mode Collapse", f"{d_score:.4f}")
        c2.metric("VS Mode Collapse", f"{v_score:.4f}")
        st.plotly_chart(
            px.bar(
                pd.DataFrame(
                    {"method": ["Direct", "Verbalized Sampling"], "score": [d_score, v_score]}
                ),
                x="method",
                y="score",
                title="Mode Collapse Score (lower is better)",
            ),
            use_container_width=True,
        )

        st.markdown("### Direct Prompting")
        st.dataframe(pd.DataFrame({"idea": direct_ideas}), use_container_width=True, hide_index=True)
        st.markdown("### Verbalized Sampling")
        st.dataframe(pd.DataFrame({"idea": vs_ideas}), use_container_width=True, hide_index=True)

        combined_ideas = direct_ideas + vs_ideas
        combined_emb = compute_embeddings(combined_ideas)
        st.markdown("### 🔥 Long Tail Ideas Discovered")
        st.dataframe(
            get_long_tail_ideas(combined_ideas, combined_emb, top_k=5),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("### 🌌 Startup Idea Universe")
        st.plotly_chart(plot_idea_map(combined_ideas, combined_emb), use_container_width=True)

        rows = []
        now = datetime.now(timezone.utc).isoformat()
        for i, idea in enumerate(direct_ideas):
            rows.append(
                {
                    "topic": topic,
                    "idea": idea,
                    "method": "direct",
                    "embedding": direct_emb[i].tolist() if len(direct_emb) > i else [],
                    "timestamp": now,
                }
            )
        for i, idea in enumerate(vs_ideas):
            rows.append(
                {
                    "topic": topic,
                    "idea": idea,
                    "method": "verbalized_sampling",
                    "embedding": vs_emb[i].tolist() if len(vs_emb) > i else [],
                    "timestamp": now,
                }
            )
        st.download_button(
            "Download Experiment Dataset",
            data=json.dumps(rows, indent=2),
            file_name="experiment_dataset.json",
            mime="application/json",
        )

        with st.expander("Raw model outputs"):
            st.markdown("#### Direct")
            st.code(direct_raw)
            st.markdown("#### Verbalized Sampling")
            st.code(vs_raw)

    st.divider()
    st.markdown("## Existing Dataset Dashboard")
    try:
        ideas_df, metrics = load_data()
    except Exception as exc:
        st.error(f"Failed to load experiment data: {exc}")
        return

    st.markdown("### 1) Overview")
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

    st.markdown("### 2) Diversity Metrics")
    comparison_df = diversity_comparison_frame(metrics)
    st.plotly_chart(diversity_bar_chart(comparison_df), use_container_width=True)

    st.markdown("### 3) Startup Idea Explorer")
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
    st.plotly_chart(
        probability_distribution_chart(
            filtered.assign(method="VS prompting"),
            title="Probability Distribution (Filtered VS Ideas)",
        ),
        use_container_width=True,
    )

    st.markdown("### 4) Embedding Visualization")
    if filtered.empty:
        st.info("No ideas available for embedding visualization with current filters.")
    else:
        emb_df = build_embedding_frame(filtered["idea"].tolist())
        plot_df = filtered.reset_index(drop=True).join(emb_df)
        st.plotly_chart(
            embedding_scatter_plot(plot_df, hover_cols=["topic", "idea", "probability"]),
            use_container_width=True,
        )

    st.markdown("### 5) Long Tail Ideas")
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
