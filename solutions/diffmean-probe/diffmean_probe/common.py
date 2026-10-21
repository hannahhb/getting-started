"""Shared prompt formatting and identifiers for extraction and inference.

Anything that changes the hidden states (prompt layout, reasoning-effort
handling, pooling sites) lives here so training-time extraction and
evaluation-time inference cannot drift apart.
"""

from __future__ import annotations

import re
from typing import Any

# Representation sites read from every layer.
SITES: tuple[str, ...] = ("last", "mean")

# gpt-oss runs at "medium" when no effort is requested.
DEFAULT_EFFORTS: dict[str, str] = {"openai/gpt-oss-120b": "medium"}


def safe_model_id(model_id: str) -> str:
    """Turn a HF model id into a filesystem-safe token."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def normalize_effort(model_id: str, reasoning_effort: str | None) -> str:
    """Map ``"default"``/missing efforts to the model's effective effort."""
    effort = (reasoning_effort or "default").strip().lower() or "default"
    if effort == "default":
        return DEFAULT_EFFORTS.get(model_id, "default")
    return effort


def format_problem(tokenizer: Any, problem_text: str, model_id: str, effort: str) -> str:
    """Format a problem as a single user turn with the generation prompt.

    The effort is forwarded to chat templates that understand it (gpt-oss);
    other templates ignore unknown keyword arguments.
    """
    if not getattr(tokenizer, "chat_template", None):
        return problem_text
    kwargs: dict[str, Any] = {}
    if effort in {"low", "medium", "high"}:
        kwargs["reasoning_effort"] = effort
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": problem_text}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def problem_token_mask(offsets: list[tuple[int, int]], start: int, end: int) -> list[bool]:
    """Mark tokens whose character span overlaps ``[start, end)``."""
    return [token_end > start and token_start < end for token_start, token_end in offsets]
