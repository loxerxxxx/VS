from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st
from openai import APIConnectionError, APIError, NotFoundError, OpenAI, RateLimitError

from dashboard.research_utils import (
    compute_embeddings,
    get_long_tail_ideas,
    mode_collapse_score,
    plot_idea_map,
)

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


if __name__ == "__main__":
    main()
