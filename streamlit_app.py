from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from openai import APIConnectionError, APIError, NotFoundError, OpenAI, RateLimitError
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
DEFAULT_LIVE_MODEL = "openai/gpt-oss-20b:free"
FREE_MODEL_FALLBACKS = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-3-4b-it:free",
    "qwen/qwen3-4b:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
]

RESPONSE_BLOCK_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)
TEXT_RE = re.compile(r"<text>(.*?)</text>", re.DOTALL | re.IGNORECASE)
PROBABILITY_RE = re.compile(
    r"<probability>\s*([0-9]*\.?[0-9]+)\s*</probability>", re.IGNORECASE
)
IDEA_LINE_RE = re.compile(r"startup idea\s*:\s*(.+)", re.IGNORECASE)
CUSTOMER_LINE_RE = re.compile(r"target customer\s*:\s*(.+)", re.IGNORECASE)
GTM_LINE_RE = re.compile(r"go-to-market strategy\s*:\s*(.+)", re.IGNORECASE)


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


def _extract_line_value(pattern: re.Pattern[str], text: str) -> str:
    for line in text.splitlines():
        match = pattern.search(line.strip())
        if match:
            return match.group(1).strip()
    return ""


def _extract_content_from_completion(completion: object) -> str:
    """Safely read text content from provider response variants."""
    choices = getattr(completion, "choices", None)
    if not choices:
        raise RuntimeError("Model returned no choices. Please retry or switch model.")

    first = choices[0] if len(choices) > 0 else None
    if first is None:
        raise RuntimeError("Model returned an empty first choice. Please retry.")

    message = getattr(first, "message", None)
    if message is None:
        raise RuntimeError("Model response is missing message content. Please retry.")

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                text = getattr(item, "text", "")
                if text:
                    parts.append(str(text))
        joined = "\n".join(part for part in parts if part).strip()
        if joined:
            return joined

    # Fallback for provider/tool outputs where content is empty but object exists.
    alt = str(message).strip()
    if alt:
        return alt
    raise RuntimeError("Model returned empty content. Try another free model.")


def _model_candidates(preferred_model: str) -> List[str]:
    candidates: List[str] = []
    if preferred_model.strip():
        candidates.append(preferred_model.strip())
    configured_fallbacks = _get_secret_or_env("OPENAI_MODEL_FALLBACKS")
    if configured_fallbacks:
        candidates.extend(
            model.strip() for model in configured_fallbacks.split(",") if model.strip()
        )
    candidates.extend(FREE_MODEL_FALLBACKS)

    deduped: List[str] = []
    seen = set()
    for model in candidates:
        if model not in seen:
            deduped.append(model)
            seen.add(model)
    return deduped


def _create_completion_with_fallback(
    client: OpenAI,
    *,
    preferred_model: str,
    system_message: str,
    prompt: str,
    retries_per_model: int = 1,
) -> object:
    last_error: Exception | None = None
    for model in _model_candidates(preferred_model):
        for attempt in range(retries_per_model + 1):
            try:
                return client.chat.completions.create(
                    model=model,
                    temperature=0.9,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                )
            except NotFoundError as exc:
                last_error = exc
                break
            except (RateLimitError, APIConnectionError, APIError) as exc:
                last_error = exc
                if attempt < retries_per_model:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                break
    raise RuntimeError(
        f"All model candidates failed. Last error: {last_error}"
    ) from last_error


def parse_live_response(raw_output: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    blocks = RESPONSE_BLOCK_RE.findall(raw_output)
    for idx, block in enumerate(blocks, start=1):
        text_match = TEXT_RE.search(block)
        prob_match = PROBABILITY_RE.search(block)
        if not text_match:
            continue

        text = text_match.group(1).strip()
        try:
            probability = float(prob_match.group(1)) if prob_match else np.nan
        except ValueError:
            probability = np.nan

        rows.append(
            {
                "rank": idx,
                "idea": _extract_line_value(IDEA_LINE_RE, text) or text,
                "target_customer": _extract_line_value(CUSTOMER_LINE_RE, text),
                "go_to_market_strategy": _extract_line_value(GTM_LINE_RE, text),
                "probability": probability,
                "raw_text": text,
            }
        )

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(
        [{"rank": 1, "idea": raw_output.strip(), "target_customer": "", "go_to_market_strategy": "", "probability": np.nan, "raw_text": raw_output.strip()}]
    )


def generate_live_ideas(topic: str, num_responses: int, model: str) -> tuple[str, pd.DataFrame]:
    api_key = _get_secret_or_env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Add it in Streamlit Secrets or environment variables."
        )
    base_url = _get_secret_or_env("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    prompt = f"""
Generate exactly {num_responses} startup ideas for topic: {topic}

Each response must be:
<response>
<text>startup idea: ...
target customer: ...
go-to-market strategy: ...</text>
<probability>...</probability>
</response>

Rules:
- probability must be numeric and < 0.10
- sample from the tails (uncommon, non-obvious concepts)
- no markdown or additional commentary
""".strip()

    completion = _create_completion_with_fallback(
        client,
        preferred_model=model,
        system_message="You are an expert AI startup strategist.",
        prompt=prompt,
    )

    raw_output = _extract_content_from_completion(completion)
    return raw_output, parse_live_response(raw_output)


def generate_live_ideas_by_mode(
    *,
    topic: str,
    num_responses: int,
    model: str,
    mode: str,
) -> tuple[str, pd.DataFrame]:
    api_key = _get_secret_or_env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Add it in Streamlit Secrets or environment variables."
        )
    base_url = _get_secret_or_env("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    if mode == "verbalized":
        prompt = f"""
Generate exactly {num_responses} startup ideas for topic: {topic}.
Sample from the tails of the distribution and prioritize non-obvious ideas.

Each response must be:
<response>
<text>startup idea: ...
target customer: ...
go-to-market strategy: ...</text>
<probability>...</probability>
</response>

Rules:
- probability must be numeric and < 0.10
- no markdown or additional commentary
""".strip()
        system_message = "You are an expert AI startup strategist focused on unconventional ideas."
    else:
        prompt = f"""
Generate exactly {num_responses} practical startup ideas for topic: {topic}.

Each response must be:
<response>
<text>startup idea: ...
target customer: ...
go-to-market strategy: ...</text>
<probability>...</probability>
</response>

Rules:
- probability must be numeric and < 0.10
- no markdown or additional commentary
""".strip()
        system_message = "You are an expert AI startup strategist."

    completion = _create_completion_with_fallback(
        client,
        preferred_model=model,
        system_message=system_message,
        prompt=prompt,
    )

    raw_output = _extract_content_from_completion(completion)
    return raw_output, parse_live_response(raw_output)


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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Teko:wght@500;600;700&display=swap');

        :root {
            --bg-main: #060816;
            --bg-panel: #0f172a;
            --bg-panel-soft: #101a31;
            --text-main: #e5e7eb;
            --text-muted: #9ca3af;
            --accent-1: #4f46e5;
            --accent-2: #06b6d4;
            --accent-3: #f43f5e;
            --border: #25314f;
        }

        .stApp {
            background:
                radial-gradient(1200px 600px at 20% -10%, rgba(79, 70, 229, 0.24), transparent 60%),
                radial-gradient(1000px 500px at 90% 0%, rgba(6, 182, 212, 0.18), transparent 55%),
                linear-gradient(180deg, #070b1a 0%, var(--bg-main) 100%);
            color: var(--text-main);
            font-family: "Inter", system-ui, -apple-system, sans-serif;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        label[data-testid="stWidgetLabel"] p {
            color: #e5e7eb !important;
            font-weight: 600 !important;
        }

        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {
            color: #f8fafc !important;
            -webkit-text-fill-color: #f8fafc !important;
        }

        div[data-baseweb="input"] {
            background-color: #0f172a !important;
            border-color: #334155 !important;
        }

        button[kind="secondaryFormSubmit"] {
            color: #ffffff !important;
            border-color: #475569 !important;
            background: #1e293b !important;
        }

        [data-testid="stCaptionContainer"], small, .stCaption, .stMarkdown p {
            color: #cbd5e1 !important;
        }

        .subtle {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        div[data-testid="stVerticalBlock"] > div:has(> div > div > .stMarkdown h2) {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(10, 15, 30, 0.92));
            border: 1px solid rgba(37, 49, 79, 0.85);
            border-radius: 16px;
            padding: 1rem 1rem 0.5rem 1rem;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.28);
            backdrop-filter: blur(4px);
        }

        [data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(14, 22, 44, 0.85), rgba(12, 20, 38, 0.9));
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.5rem 0.75rem;
        }

        .hero-title {
            font-family: "Teko", "Inter", sans-serif;
            font-size: clamp(2.3rem, 5vw, 4rem);
            letter-spacing: 0.08em;
            line-height: 0.95;
            margin-bottom: 0.15rem;
            text-transform: uppercase;
            color: #f8fafc;
            text-shadow:
                0 0 24px rgba(79, 70, 229, 0.45),
                0 0 40px rgba(6, 182, 212, 0.25);
        }

        .hero-subtitle {
            color: var(--text-muted);
            margin-top: 0;
            margin-bottom: 0.25rem;
            font-size: 0.95rem;
        }

        .hero-tag {
            color: #c7d2fe;
            font-size: 0.9rem;
            margin-top: 0;
            margin-bottom: 1.25rem;
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

    st.markdown(
        """
        <div class="hero-title">VS Startup Bench Dashboard</div>
        <p class="hero-subtitle">Experiment analytics for diversity in AI startup ideation</p>
        <p class="hero-tag">by a desperate psychopath aka Prince</p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Live Idea Generator")
    st.write(
        "Single prompt ingestion with dual generation: **Direct Prompting** and "
        "**Verbalized Prompting** side-by-side."
    )
    with st.form("live_generation_form"):
        c1, c2 = st.columns([2, 1])
        topic_input = c1.text_input(
            "Topic / Prompt",
            value="AI healthcare",
            placeholder="e.g. AI legal ops, AI climate risk, AI creator tools",
        )
        count_input = c2.slider("Responses", min_value=3, max_value=8, value=5, step=1)
        model_input = st.text_input(
            "Model",
            value=_get_secret_or_env("OPENAI_MODEL") or DEFAULT_LIVE_MODEL,
            help="Set default via OPENAI_MODEL secret/env.",
        )
        generate_clicked = st.form_submit_button("Generate Ideas")

    if generate_clicked:
        if not topic_input.strip():
            st.warning("Please provide a topic.")
        else:
            with st.spinner("Generating ideas..."):
                try:
                    direct_raw, direct_df = generate_live_ideas_by_mode(
                        topic=topic_input.strip(),
                        num_responses=count_input,
                        model=model_input.strip(),
                        mode="direct",
                    )
                    verbalized_raw, verbalized_df = generate_live_ideas_by_mode(
                        topic=topic_input.strip(),
                        num_responses=count_input,
                        model=model_input.strip(),
                        mode="verbalized",
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Live generation failed: {exc}")
                else:
                    st.success(
                        f"Generated {len(direct_df)} direct ideas and {len(verbalized_df)} verbalized ideas."
                    )

                    left, right = st.columns(2)
                    with left:
                        st.markdown("### Direct Prompting")
                        st.dataframe(
                            direct_df[
                                [
                                    "rank",
                                    "idea",
                                    "target_customer",
                                    "go_to_market_strategy",
                                    "probability",
                                ]
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.download_button(
                            label="Download direct ideas (JSON)",
                            data=json.dumps(direct_df.to_dict(orient="records"), indent=2),
                            file_name="direct_live_generated_ideas.json",
                            mime="application/json",
                        )
                        with st.expander("Direct raw model output"):
                            st.code(direct_raw)

                    with right:
                        st.markdown("### Verbalized Prompting")
                        st.dataframe(
                            verbalized_df[
                                [
                                    "rank",
                                    "idea",
                                    "target_customer",
                                    "go_to_market_strategy",
                                    "probability",
                                ]
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.download_button(
                            label="Download verbalized ideas (JSON)",
                            data=json.dumps(verbalized_df.to_dict(orient="records"), indent=2),
                            file_name="verbalized_live_generated_ideas.json",
                            mime="application/json",
                        )
                        with st.expander("Verbalized raw model output"):
                            st.code(verbalized_raw)

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
