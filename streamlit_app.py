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
DEFAULT_MODEL = "gpt-4o-mini"
FREE_MODEL_FALLBACKS = [
    "gpt-4o",
    "openai/gpt-oss-20b:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "qwen/qwen3-4b:free",
]
KIMI_MODEL = "moonshotai/kimi-k2"
UI_STATE_VERSION = "v4"

RESPONSE_BLOCK_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)
TEXT_RE = re.compile(r"<text>(.*?)</text>", re.DOTALL | re.IGNORECASE)
LIST_ITEM_RE = re.compile(
    r"(?:^|\n)\s*(?:\d+[\).:-]|[-*•])\s+(.+?)(?=(?:\n\s*(?:\d+[\).:-]|[-*•])\s+)|\Z)",
    re.DOTALL,
)
INLINE_COMPOUND_ITEM_RE = re.compile(
    r"(?:^|\s)(?:\d+[\).:-]|[-*•])\s+(.+?)(?=(?:\s(?:\d+[\).:-]|[-*•])\s+)|$)",
    re.DOTALL,
)


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


def _max_model_candidates() -> int:
    raw = _get_secret_or_env("OPENAI_MAX_MODEL_CANDIDATES")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return 1
    # Free-tier safe default: avoid broad probing that triggers rate limits.
    return 1


def _max_attempts_per_model() -> int:
    raw = _get_secret_or_env("OPENAI_MAX_ATTEMPTS_PER_MODEL")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return 1
    # Free-tier safe default.
    return 1


def _format_rate_limit_error(exc: Exception) -> str:
    text = str(exc)
    reset_match = re.search(r"X-RateLimit-Reset':\s*'(\d+)'", text)
    if reset_match:
        try:
            reset_ms = int(reset_match.group(1))
            reset_dt = datetime.fromtimestamp(reset_ms / 1000, tz=timezone.utc).astimezone()
            reset_str = reset_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            return (
                "Rate limit exceeded for free models. "
                f"Retry after {reset_str}."
            )
        except Exception:  # noqa: BLE001
            pass
    return "Rate limit exceeded for free models. Please wait about a minute and retry."


def _client() -> OpenAI:
    api_key = _get_secret_or_env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit secrets.")
    base_url = _get_secret_or_env("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    return OpenAI(api_key=api_key, base_url=base_url)


def _chat_with_fallback(
    model: str,
    system: str,
    prompt: str,
    *,
    temperature: float = 0.7,
) -> str:
    client = _client()
    return _chat_with_fallback_using_client(
        client,
        model,
        system,
        prompt,
        temperature=temperature,
    )


def _chat_with_fallback_using_client(
    client: OpenAI,
    model: str,
    system: str,
    prompt: str,
    *,
    temperature: float = 0.7,
) -> str:
    last_error: Exception | None = None
    candidates = _model_candidates(model)[: _max_model_candidates()]
    for candidate in candidates:
        for attempt in range(_max_attempts_per_model()):
            try:
                completion = client.chat.completions.create(
                    model=candidate,
                    temperature=temperature,
                    max_tokens=1400,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                return _extract_content(completion)
            except NotFoundError as exc:
                last_error = exc
                break
            except RateLimitError as exc:
                # Fail fast on global free-tier limits to avoid long spinner loops.
                raise RuntimeError(_format_rate_limit_error(exc)) from exc
            except (APIConnectionError, APIError) as exc:
                last_error = exc
                time.sleep(1.2 * (attempt + 1))
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                break
    raise RuntimeError(f"All model attempts failed. Last error: {last_error}")


def _normalize_idea(text: str) -> str:
    cleaned = re.sub(r"</?[^>]+>", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        key = item.lower().strip()
        if key and key not in seen:
            out.append(item.strip())
            seen.add(key)
    return out


def _split_compound_idea(text: str) -> List[str]:
    parts = [_normalize_idea(m.group(1)) for m in INLINE_COMPOUND_ITEM_RE.finditer(text)]
    parts = [p for p in parts if p]
    if len(parts) >= 2:
        return parts
    return [text]


def _parse_ideas(raw: str, limit: int = 10) -> List[str]:
    ideas: List[str] = []
    blocks = RESPONSE_BLOCK_RE.findall(raw)
    if blocks:
        for block in blocks:
            match = TEXT_RE.search(block)
            if match:
                txt = _normalize_idea(match.group(1))
                if txt:
                    ideas.append(txt)

    if not ideas:
        list_items = [_normalize_idea(m.group(1)) for m in LIST_ITEM_RE.finditer(raw)]
        ideas.extend(x for x in list_items if x)

    if not ideas:
        paras = [_normalize_idea(p) for p in re.split(r"\n\s*\n+", raw) if p.strip()]
        ideas.extend(x for x in paras if x and len(x.split()) > 3)

    if not ideas:
        lines = [_normalize_idea(line.strip("- ").strip()) for line in raw.splitlines() if line.strip()]
        ideas.extend(x for x in lines if x)

    # If a single parsed item contains multiple numbered/bulleted ideas, split it.
    if len(ideas) == 1:
        ideas = _split_compound_idea(ideas[0])

    ideas = _dedupe_keep_order(ideas)
    return ideas[:limit]


def generate_ideas(topic: str, mode: str, model: str, n: int = 10) -> tuple[List[str], str]:
    if mode == "verbalized_sampling":
        prompt = (
            f"Think of 20 startup ideas for {topic}. Sample unusual/long-tail regions.\n"
            f"Return exactly {n} responses and NOTHING else.\n"
            "STRICT FORMAT (repeat this block exactly 5 times):\n"
            "<response><text>[one idea only, single sentence, no numbering]</text>"
            "<probability>[numeric < 0.10]</probability></response>\n"
            "Rules: one idea per response block, no combined ideas, no markdown, no prose."
        )
        system = "You generate unconventional but plausible startup ideas."
    else:
        prompt = (
            f"Generate exactly {n} startup ideas for: {topic}.\n"
            "Return exactly this XML format and NOTHING else:\n"
            "<response><text>[one idea only, single sentence, no numbering]</text>"
            "<probability>[numeric < 0.10]</probability></response>\n"
            "Rules: output exactly 5 separate <response> blocks, one idea per block, "
            "do not combine multiple ideas in one block, no markdown/prose."
        )
        system = "You generate clear practical startup ideas."
    temperature = 0.85 if mode == "verbalized_sampling" else 0.6
    client = _client()

    best_ideas: List[str] = []
    best_raw = ""
    last_error: Exception | None = None

    # Try model candidates until one produces the requested count.
    candidates = _model_candidates(model)[: _max_model_candidates()]
    for candidate in candidates:
        try:
            raw = _chat_with_fallback_using_client(
                client,
                candidate,
                system,
                prompt,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

        ideas = _parse_ideas(raw, limit=n)
        if len(ideas) < n:
            repair_prompt = (
                f"Convert the content below into exactly {n} distinct startup ideas. "
                "Output one idea per line, no numbering, no markdown.\n\n"
                f"CONTENT:\n{raw}"
            )
            try:
                repaired_raw = _chat_with_fallback_using_client(
                    client,
                    candidate,
                    "You are a strict output formatter.",
                    repair_prompt,
                    temperature=0.2,
                )
                repaired_ideas = _parse_ideas(repaired_raw, limit=n)
                if len(repaired_ideas) > len(ideas):
                    ideas = repaired_ideas
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        if len(ideas) > len(best_ideas):
            best_ideas = ideas
            best_raw = raw
        if len(ideas) >= n:
            return ideas[:n], raw

    if best_ideas:
        return best_ideas[:n], best_raw
    raise RuntimeError(f"All model candidates failed to produce ideas. Last error: {last_error}")


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

    st.sidebar.markdown("**Mode: Experiment (Direct + VS)**")
    preset = st.sidebar.selectbox(
        "Model preset",
        [
            "OpenAI gpt-4o-mini (research-safe)",
            "NVIDIA free (fast fallback)",
            "Kimi (requires your access)",
            "Custom model",
        ],
        key=f"model_preset_{UI_STATE_VERSION}",
    )
    if preset == "OpenAI gpt-4o-mini (research-safe)":
        model_default = DEFAULT_MODEL
    elif preset == "NVIDIA free (fast fallback)":
        model_default = "nvidia/nemotron-3-nano-30b-a3b:free"
    elif preset == "Kimi (requires your access)":
        model_default = KIMI_MODEL
    else:
        model_default = _get_secret_or_env("OPENAI_MODEL") or DEFAULT_MODEL

    st.markdown("## Live Experiment Generator")
    st.caption("Experiment mode runs Direct + VS with 5 ideas each for reliability.")
    with st.form("generate_form"):
        topic = st.text_input("Topic", value="AI healthcare", key=f"topic_{UI_STATE_VERSION}")
        n_ideas = st.slider(
            "Ideas per method",
            min_value=5,
            max_value=15,
            value=5,
            key=f"ideas_per_method_{UI_STATE_VERSION}",
        )
        model = st.text_input("Model", value=model_default, key=f"model_{UI_STATE_VERSION}")
        submitted = st.form_submit_button("Generate")

    if not submitted:
        st.info("Select mode, enter topic, and click Generate.")
    elif not topic.strip():
        st.warning("Topic is required.")
    else:
        # Force consistent behavior near deadline: always run both methods with 5 outputs each.
        if n_ideas != 5:
            st.info("Using fixed value: 5 ideas per method (deadline reliability mode).")
        n_ideas = 5

        with st.spinner("Generating ideas..."):
            try:
                direct_ideas, direct_raw = generate_ideas(topic, "direct", model, n=n_ideas)
                vs_ideas: List[str] = []
                vs_raw = ""
                vs_ideas, vs_raw = generate_ideas(
                    topic, "verbalized_sampling", model, n=n_ideas
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Generation failed: {exc}")
                if "spend limit exceeded" in str(exc).lower() or "error code: 402" in str(exc).lower():
                    st.info(
                        "Provider spend cap reached (402). Update OpenRouter key spend limit "
                        "or use another key/provider route."
                    )
                if "Rate limit exceeded" in str(exc):
                    st.info(
                        "Free-tier limit reached. Quick options: wait for reset and retry, "
                        "or add a paid key/provider for faster and more stable generation."
                    )
                st.caption(
                    "Tip: use OpenAI gpt-4o-mini (research-safe) for stable Direct vs VS comparison."
                )
            else:
                if len(direct_ideas) < n_ideas:
                    st.warning(
                        f"Direct parsing returned {len(direct_ideas)}/{n_ideas} ideas. "
                        "Model format was inconsistent; retry for fuller output."
                    )
                    st.markdown("#### Direct raw output (auto-shown due to short parse)")
                    st.code(direct_raw)
                if len(vs_ideas) < n_ideas:
                    st.warning(
                        f"VS parsing returned {len(vs_ideas)}/{n_ideas} ideas. "
                        "Model format was inconsistent; retry for fuller output."
                    )
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
                            {
                                "method": ["Direct", "Verbalized Sampling"],
                                "score": [d_score, v_score],
                            }
                        ),
                        x="method",
                        y="score",
                        title="Mode Collapse Score (lower is better)",
                    ),
                    use_container_width=True,
                )

                st.markdown("### Direct Prompting")
                st.dataframe(
                    pd.DataFrame({"idea": direct_ideas}),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown("### Verbalized Sampling")
                st.dataframe(
                    pd.DataFrame({"idea": vs_ideas}),
                    use_container_width=True,
                    hide_index=True,
                )

                combined_ideas = direct_ideas + vs_ideas
                combined_emb = compute_embeddings(combined_ideas)
                st.markdown("### 🔥 Long Tail Ideas Discovered")
                st.dataframe(
                    get_long_tail_ideas(combined_ideas, combined_emb, top_k=5),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown("### 🌌 Startup Idea Universe")
                st.plotly_chart(
                    plot_idea_map(combined_ideas, combined_emb),
                    use_container_width=True,
                )

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
