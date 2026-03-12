from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

from openai import APIConnectionError, APIError, NotFoundError, OpenAI, RateLimitError
from tqdm import tqdm


TOPICS = [
    "AI healthcare",
    "AI developer tools",
    "AI education",
    "AI fintech",
    "AI marketplaces",
    "AI marketing",
    "AI logistics",
    "AI robotics",
]

PROMPT_FILES = {
    "vs": Path("prompts/vs_prompt.txt"),
    "direct": Path("prompts/direct_prompt.txt"),
}

OUTPUT_FILES = {
    "vs": Path("data/vs_outputs.json"),
    "direct": Path("data/direct_outputs.json"),
}


@dataclass
class GenerationResult:
    topic: str
    prompt_type: str
    raw_output: str
    timestamp: str


def read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_user_prompt(base_prompt: str, topic: str) -> str:
    return f"{base_prompt}\n\nTopic: {topic}"


def create_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def parse_model_candidates() -> List[str]:
    primary = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    fallback_raw = os.getenv("OPENAI_MODEL_FALLBACKS", "").strip()
    candidates: List[str] = [primary] if primary else []
    if fallback_raw:
        candidates.extend(model.strip() for model in fallback_raw.split(",") if model.strip())

    # De-duplicate while preserving order.
    deduped: List[str] = []
    seen = set()
    for model in candidates:
        if model not in seen:
            deduped.append(model)
            seen.add(model)
    if not deduped:
        raise ValueError("No valid model candidates found. Set OPENAI_MODEL.")
    return deduped


def generate_raw_output(
    client: OpenAI,
    *,
    model: str,
    user_prompt: str,
    retries: int,
    backoff_seconds: float,
) -> str:
    attempt = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful startup strategy assistant."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.9,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except (RateLimitError, APIConnectionError, APIError) as exc:
            if attempt >= retries:
                raise RuntimeError(
                    f"Model '{model}' failed after {retries + 1} attempts: {exc}"
                ) from exc
            sleep_seconds = backoff_seconds * (2**attempt)
            print(
                f"[retry] model='{model}' attempt={attempt + 1}/{retries + 1} "
                f"sleep={sleep_seconds:.1f}s reason={type(exc).__name__}"
            )
            time.sleep(sleep_seconds)
            attempt += 1
        except NotFoundError as exc:
            # Model not found on endpoint; no retry for this model.
            raise RuntimeError(f"Model '{model}' is not available: {exc}") from exc


def run_prompt_batch(
    client: OpenAI,
    *,
    model_candidates: Sequence[str],
    prompt_type: str,
    prompt_text: str,
    topics: List[str],
    output_path: Path,
    retries_per_model: int,
    backoff_seconds: float,
) -> List[GenerationResult]:
    results: List[GenerationResult] = []
    progress_label = f"Generating {prompt_type} outputs"

    for topic in tqdm(topics, desc=progress_label):
        user_prompt = build_user_prompt(prompt_text, topic)
        last_error: Exception | None = None
        raw_output: str | None = None
        for model in model_candidates:
            try:
                raw_output = generate_raw_output(
                    client,
                    model=model,
                    user_prompt=user_prompt,
                    retries=retries_per_model,
                    backoff_seconds=backoff_seconds,
                )
                break
            except Exception as exc:  # noqa: BLE001 - bubble up after trying all candidates
                last_error = exc
                print(f"[model-fallback] topic='{topic}' model='{model}' failed: {exc}")

        if raw_output is None:
            raise RuntimeError(
                f"All model candidates failed for topic '{topic}' and prompt '{prompt_type}'."
            ) from last_error

        results.append(
            GenerationResult(
                topic=topic,
                prompt_type=prompt_type,
                raw_output=raw_output,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )
        # Persist each step to avoid losing long-running progress.
        save_results(output_path, results)

    return results


def save_results(path: Path, results: List[GenerationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) for item in results]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    model_candidates = parse_model_candidates()
    retries_per_model = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
    backoff_seconds = float(os.getenv("OPENAI_RETRY_BACKOFF_SECONDS", "2.0"))
    client = create_client()

    prompt_texts = {key: read_text_file(path) for key, path in PROMPT_FILES.items()}

    vs_results = run_prompt_batch(
        client,
        model_candidates=model_candidates,
        prompt_type="vs",
        prompt_text=prompt_texts["vs"],
        topics=TOPICS,
        output_path=OUTPUT_FILES["vs"],
        retries_per_model=retries_per_model,
        backoff_seconds=backoff_seconds,
    )
    direct_results = run_prompt_batch(
        client,
        model_candidates=model_candidates,
        prompt_type="direct",
        prompt_text=prompt_texts["direct"],
        topics=TOPICS,
        output_path=OUTPUT_FILES["direct"],
        retries_per_model=retries_per_model,
        backoff_seconds=backoff_seconds,
    )

    save_results(OUTPUT_FILES["vs"], vs_results)
    save_results(OUTPUT_FILES["direct"], direct_results)

    print(f"Saved {len(vs_results)} rows to {OUTPUT_FILES['vs']}")
    print(f"Saved {len(direct_results)} rows to {OUTPUT_FILES['direct']}")


if __name__ == "__main__":
    main()
