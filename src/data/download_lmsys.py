"""Stream, filter, and cache the lmarena-ai/arena-human-preference-140k dataset.

This standalone script pulls the LMArena human-preference battles directly from
the Hugging Face Hub in streaming mode (no full local materialization required),
applies two coarse filters, and writes a clean intermediate parquet shard to the
gitignored ``data/raw/`` directory.

Filters applied (in order):
    1. Language gate     - keep English-only rows (multilingual handling deferred).
    2. Model family gate - keep rows where *both* ``model_a`` and ``model_b``
                           belong to one of the target proprietary class pools
                           (ChatGPT / Claude / Gemini). Rows containing any
                           open-source family model (Llama, Qwen, Mistral, ...)
                           are dropped because they fall outside every target pool.

Example
-------
    python src/data/download_lmsys.py                  # full stream -> data/raw/
    python src/data/download_lmsys.py --limit 5000     # quick smoke test
    python src/data/download_lmsys.py --include-non-english
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# --------------------------------------------------------------------------- #
# Project wiring: reuse the centralized paths/secrets from config/config.py.
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import HF_TOKEN, RAW_DATA_DIR  # noqa: E402

DATASET_ID = "lmarena-ai/arena-human-preference-140k"
DEFAULT_SPLIT = "train"

# --------------------------------------------------------------------------- #
# Model family signatures.
#
# Each target family maps to a list of regex signatures matched (case-insensitive)
# against the raw ``model_a`` / ``model_b`` strings. The patterns intentionally
# match by family prefix rather than exact version so that newer checkpoints
# (e.g. ``chatgpt-4o-latest``, ``claude-3-5-sonnet``, ``gemini-1.5-pro``) are all
# captured without code changes.
# --------------------------------------------------------------------------- #
TARGET_FAMILY_SIGNATURES: dict[str, list[str]] = {
    # ChatGPT / OpenAI: gpt-3.5-turbo, gpt-4-*, gpt-4o, chatgpt-4o-latest, o1/o3/o4-*.
    "chatgpt": [r"gpt", r"chatgpt", r"^o[134]\b", r"^o[134]-"],
    # Anthropic: claude-2, claude-3-*, claude-sonnet-4-*, etc.
    "claude": [r"claude"],
    # Google: gemini-1.0-*, gemini-1.5-*, gemini-2.*.
    "gemini": [r"gemini"],
}

# Open-source families that must never survive the filter. The target gate above
# already excludes anything outside the proprietary pools, but this explicit
# blocklist acts as a defensive second check and powers the audit counters.
OPEN_SOURCE_SIGNATURES: list[str] = [
    r"llama",
    r"qwen",
    r"qwq",
    r"mistral",
    r"mixtral",
]

# Pre-compile for the hot loop.
_TARGET_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    family: [re.compile(sig, re.IGNORECASE) for sig in sigs]
    for family, sigs in TARGET_FAMILY_SIGNATURES.items()
}
_OPEN_SOURCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(sig, re.IGNORECASE) for sig in OPEN_SOURCE_SIGNATURES
]


def classify_model(name: str | None) -> str | None:
    """Return the target family for ``name``, or ``None`` if out of scope.

    Open-source matches short-circuit to ``None`` so a stray naming collision
    can never leak a Llama/Qwen/Mistral row into a target pool.
    """
    if not name:
        return None
    if any(p.search(name) for p in _OPEN_SOURCE_PATTERNS):
        return None
    for family, patterns in _TARGET_PATTERNS.items():
        if any(p.search(name) for p in patterns):
            return family
    return None


def extract_text(conversation: list | None, role: str) -> str:
    """Flatten a nested arena conversation into plain text for one role.

    Each turn looks like ``{"role": ..., "content": [{"type", "text", ...}]}``.
    Only ``type == "text"`` parts are concatenated; image parts are skipped.
    """
    if not conversation:
        return ""
    chunks: list[str] = []
    for turn in conversation:
        if turn.get("role") != role:
            continue
        content = turn.get("content") or []
        if isinstance(content, str):
            chunks.append(content)
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                chunks.append(part["text"])
    return "\n".join(chunks).strip()


def build_record(row: dict, family_a: str, family_b: str) -> dict:
    """Assemble a flat, downstream-friendly record from a raw streamed row."""
    return {
        "id": row.get("id"),
        "model_a": row.get("model_a"),
        "model_b": row.get("model_b"),
        "family_a": family_a,
        "family_b": family_b,
        "winner": row.get("winner"),
        "language": row.get("language"),
        "is_code": row.get("is_code"),
        "prompt": extract_text(row.get("conversation_a"), role="user"),
        "response_a": extract_text(row.get("conversation_a"), role="assistant"),
        "response_b": extract_text(row.get("conversation_b"), role="assistant"),
        "timestamp": row.get("timestamp"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream + filter the LMArena 140k human-preference dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after inspecting N source rows (smoke testing). Default: full stream.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_DATA_DIR / "arena_140k_chatgpt_claude_gemini_en.parquet",
        help="Destination parquet path (defaults under the gitignored data/raw/).",
    )
    parser.add_argument(
        "--include-non-english",
        action="store_true",
        help="Disable the English-only language gate (keeps all languages).",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Dataset split to stream (default: {DEFAULT_SPLIT}).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10_000,
        help="Emit a progress line every N inspected rows.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f"Streaming '{DATASET_ID}' [{args.split}] from the Hugging Face Hub...")
    stream = load_dataset(
        DATASET_ID,
        split=args.split,
        streaming=True,
        token=HF_TOKEN or None,
    )

    kept: list[dict] = []
    inspected = 0
    dropped_language = 0
    dropped_family = 0
    family_pairs: dict[tuple[str, str], int] = {}

    for row in stream:
        inspected += 1

        if not args.include_non_english and (row.get("language") or "").lower() != "en":
            dropped_language += 1
        else:
            family_a = classify_model(row.get("model_a"))
            family_b = classify_model(row.get("model_b"))
            if family_a is None or family_b is None:
                dropped_family += 1
            else:
                kept.append(build_record(row, family_a, family_b))
                pair = tuple(sorted((family_a, family_b)))
                family_pairs[pair] = family_pairs.get(pair, 0) + 1

        if args.log_every and inspected % args.log_every == 0:
            print(
                f"  inspected={inspected:,} kept={len(kept):,} "
                f"dropped_lang={dropped_language:,} dropped_family={dropped_family:,}"
            )

        if args.limit is not None and inspected >= args.limit:
            break

    print("\n--- Filter summary ---")
    print(f"Rows inspected      : {inspected:,}")
    print(f"Dropped (non-en)    : {dropped_language:,}")
    print(f"Dropped (off-pool)  : {dropped_family:,}")
    print(f"Rows kept           : {len(kept):,}")
    if family_pairs:
        print("Kept family match-ups:")
        for pair, count in sorted(family_pairs.items(), key=lambda kv: -kv[1]):
            print(f"  {pair[0]:>8} vs {pair[1]:<8}: {count:,}")

    if not kept:
        print("\nNo rows survived the filters; nothing written.")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(kept)
    df.to_parquet(args.output, index=False)
    print(f"\nSaved {len(df):,} filtered rows -> {args.output}")
    return 0


if __name__ == "__main__":
    _code = main()
    # The `datasets` streaming reader can leave a non-daemon prefetch thread
    # alive, which blocks a normal interpreter shutdown and makes the script
    # appear to hang after the parquet file is already flushed and closed.
    # All real work is done by this point, so force a clean, immediate exit.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
