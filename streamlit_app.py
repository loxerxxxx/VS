from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI
from sklearn.cluster import KMeans

from dashboard.research_utils import (
    compute_embeddings,
    compute_long_tail_ideas,
    compute_mode_collapse_score,
    plot_embedding_map,
)


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LIVE_MODEL = "openai/gpt-oss-20b:free"

RESPONSE_BLOCK_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)
TEXT_RE = re.compile(r"<text>(.*?)</text>", re.DOTALL | re.IGNORECASE)
PROBABILITY_RE = re.compile(
    r"<probability>\s*([0-9]*\.?[0-9]+)\s*</probability>", re.IGNORECASE
)
IDEA_LINE_RE = re.compile(r"startup idea\s*:\s*(.+)", re.IGNORECASE)
CUSTOMER_LINE_RE = re.compile(r"target customer\s*:\s*(.+)", re.IGNORECASE)
GTM_LINE_RE = re.compile(r"go-to-market strategy\s*:\s*(.+)", re.IGNORECASE)


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
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        joined = "\n".join(part for part in parts if part).strip()
        if joined:
            return joined
    alt = str(message).strip()
    if alt:
        return alt
    raise RuntimeError("Model returned empty content. Try another free model.")


def parse_live_response(raw_output: str, prompt_type: str, topic: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    blocks = RESPONSE_BLOCK_RE.findall(raw_output or "")
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
                "topic": topic,
                "prompt_type": prompt_type,
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
        [
            {
                "rank": 1,
                "topic": topic,
                "prompt_type": prompt_type,
                "idea": raw_output.strip(),
                "target_customer": "",
                "go_to_market_strategy": "",
                "probability": np.nan,
                "raw_text": raw_output.strip(),
            }
        ]
    )


def _build_client() -> OpenAI:
    api_key = _get_secret_or_env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit secrets/environment.")
    base_url = _get_secret_or_env("OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)


def _call_model(client: OpenAI, model: str, system_message: str, user_prompt: str) -> str:
    completion = client.chat.completions.create(
        model=model,
        temperature=0.9,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
    )
    return _extract_content_from_completion(completion)


def generate_for_mode(
    *,
    topic: str,
    model: str,
    num_responses: int,
    mode: str,
) -> Tuple[str, pd.DataFrame]:
    client = _build_client()
    if mode == "vs":
        system = "You are a startup researcher maximizing idea diversity and tail-sampling."
        user = f"""
Generate exactly {num_responses} startup and go-to-market ideas for topic: {topic}.
Use verbalized sampling: intentionally sample from low-probability tails.

Output format:
<response>
<text>startup idea: ...
target customer: ...
go-to-market strategy: ...</text>
<probability>...</probability>
</response>

Rules:
- Return exactly {num_responses} response blocks
- probability must be numeric and < 0.10
- no markdown and no extra commentary
""".strip()
        prompt_type = "vs"
    else:
        system = "You are a startup strategist."
        user = f"""
Generate exactly {num_responses} startup and go-to-market ideas for topic: {topic}.

Output format:
<response>
<text>startup idea: ...
target customer: ...
go-to-market strategy: ...</text>
<probability>...</probability>
</response>

Rules:
- Return exactly {num_responses} response blocks
- probability must be numeric and < 0.10
- no markdown and no extra commentary
""".strip()
        prompt_type = "direct"

    raw_output = _call_model(client, model, system, user)
    return raw_output, parse_live_response(raw_output, prompt_type=prompt_type, topic=topic)


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #070b1a 0%, #060816 100%);
        }
        .block-container { max-width: 1200px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def compute_sidebar_metrics(df: pd.DataFrame, embeddings: np.ndarray) -> Dict[str, float]:
    total_ideas = int(len(df))
    if total_ideas == 0 or embeddings.size == 0:
        return {
            "total_ideas": 0,
            "unique_clusters": 0,
            "mode_collapse_score": 0.0,
            "long_tail_discovery_score": 0.0,
        }

    n_clusters = max(1, min(int(np.sqrt(total_ideas)), total_ideas))
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(embeddings)
    unique_clusters = int(len(np.unique(labels)))

    return {
        "total_ideas": total_ideas,
        "unique_clusters": unique_clusters,
        "mode_collapse_score": compute_mode_collapse_score(embeddings),
        "long_tail_discovery_score": unique_clusters / total_ideas,
    }


def export_dataset_payload(df: pd.DataFrame, embeddings: np.ndarray) -> str:
    if df.empty or embeddings.size == 0:
        return "[]"
    payload = []
    rows = df.reset_index(drop=True)
    for i, row in rows.iterrows():
        emb = embeddings[i].tolist() if i < len(embeddings) else []
        payload.append(
            {
                "topic": row.get("topic", ""),
                "idea": row.get("idea", ""),
                "probability": row.get("probability", np.nan),
                "embedding": emb,
                "prompt_type": row.get("prompt_type", ""),
            }
        )
    return json.dumps(payload, indent=2)


def main() -> None:
    st.set_page_config(page_title="A Desperate Psychopath’s Idea Machine", layout="wide")
    apply_custom_style()

    # HEADER
    st.title("A Desperate Psychopath’s Idea Machine")
    st.caption("(Streamlit ideation tool generating startup/GTM ideas)")

    st.sidebar.header("Mode")
    mode = st.sidebar.selectbox("Mode", ["Brainstorm Mode", "Experiment Mode"])

    model = st.sidebar.text_input(
        "Model",
        value=_get_secret_or_env("OPENAI_MODEL") or DEFAULT_LIVE_MODEL,
    )
    responses = st.sidebar.slider("Responses", min_value=3, max_value=8, value=5, step=1)

    # Section 1: Idea Generator
    st.markdown("## Section 1: Idea Generator")
    topic = st.text_input(
        "Topic / Prompt",
        value="AI healthcare",
        placeholder="Enter a startup theme/topic",
    )
    run_clicked = st.button("Generate Ideas")

    direct_df = pd.DataFrame()
    vs_df = pd.DataFrame()
    direct_raw = ""
    vs_raw = ""

    if run_clicked:
        if not topic.strip():
            st.warning("Please provide a topic.")
        else:
            with st.spinner("Running generation..."):
                try:
                    if mode == "Brainstorm Mode":
                        direct_raw, direct_df = generate_for_mode(
                            topic=topic.strip(),
                            model=model.strip(),
                            num_responses=responses,
                            mode="direct",
                        )
                    else:
                        direct_raw, direct_df = generate_for_mode(
                            topic=topic.strip(),
                            model=model.strip(),
                            num_responses=responses,
                            mode="direct",
                        )
                        vs_raw, vs_df = generate_for_mode(
                            topic=topic.strip(),
                            model=model.strip(),
                            num_responses=responses,
                            mode="vs",
                        )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Generation failed: {exc}")

    # Section 2: Experiment Comparison
    st.markdown("## Section 2: Experiment Comparison")
    if mode == "Experiment Mode" and (not direct_df.empty or not vs_df.empty):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Direct Prompt")
            st.dataframe(direct_df, width="stretch", hide_index=True)
        with c2:
            st.subheader("Verbalized Sampling Prompt")
            st.dataframe(vs_df, width="stretch", hide_index=True)
    elif mode == "Brainstorm Mode" and not direct_df.empty:
        st.dataframe(direct_df, width="stretch", hide_index=True)
        with st.expander("Raw output"):
            st.code(direct_raw)
    else:
        st.info("Generate ideas to compare prompting strategies.")

    combined_df = pd.concat([direct_df, vs_df], ignore_index=True)
    idea_texts = combined_df["idea"].tolist() if not combined_df.empty else []
    embeddings = compute_embeddings(idea_texts, model_name=EMBEDDING_MODEL)

    # Section 3: Mode Collapse Visualization
    st.markdown("## Section 3: Mode Collapse Visualization")
    if mode == "Experiment Mode" and not direct_df.empty and not vs_df.empty:
        direct_embeddings = compute_embeddings(direct_df["idea"].tolist(), model_name=EMBEDDING_MODEL)
        vs_embeddings = compute_embeddings(vs_df["idea"].tolist(), model_name=EMBEDDING_MODEL)
        direct_score = compute_mode_collapse_score(direct_embeddings)
        vs_score = compute_mode_collapse_score(vs_embeddings)

        mode_df = pd.DataFrame(
            [
                {"prompt": "Direct Prompt", "mode_collapse_score": direct_score},
                {"prompt": "Verbalized Sampling", "mode_collapse_score": vs_score},
            ]
        )
        fig = px.bar(
            mode_df,
            x="prompt",
            y="mode_collapse_score",
            color="prompt",
            title="Mode Collapse Score Comparison",
            text_auto=".4f",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Mode Collapse chart appears in Experiment Mode after generation.")

    # Section 4: Long Tail Ideas
    st.markdown("## Section 4: Long Tail Ideas")
    st.markdown("### 🔥 Long Tail Ideas Discovered")
    if not combined_df.empty and embeddings.size > 0:
        long_tail_df = compute_long_tail_ideas(combined_df, embeddings, top_k=10)
        for _, row in long_tail_df.iterrows():
            st.markdown(
                f"- **{row.get('idea', '')}**  \n"
                f"  target: {row.get('target_customer', '')}  \n"
                f"  gtm: {row.get('go_to_market_strategy', '')}"
            )
    else:
        st.info("Generate ideas to discover long-tail ideas.")

    # Section 5: Idea Landscape Map
    st.markdown("## Section 5: Idea Landscape Map")
    if not combined_df.empty and embeddings.size > 0:
        map_fig = plot_embedding_map(combined_df, embeddings)
        st.plotly_chart(map_fig, width="stretch")
    else:
        st.info("Generate ideas to view the embedding map.")

    # Sidebar: Research Insights
    st.sidebar.markdown("## Research Insights")
    summary = compute_sidebar_metrics(combined_df, embeddings)
    st.sidebar.metric("Total Ideas Generated", summary["total_ideas"])
    st.sidebar.metric("Unique Idea Clusters", summary["unique_clusters"])
    st.sidebar.metric("Mode Collapse Score", f"{summary['mode_collapse_score']:.4f}")
    st.sidebar.metric(
        "Long Tail Discovery Score", f"{summary['long_tail_discovery_score']:.4f}"
    )

    # Export dataset
    export_json = export_dataset_payload(combined_df, embeddings)
    st.download_button(
        "Download Experiment Dataset",
        data=export_json,
        file_name="experiment_dataset.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
