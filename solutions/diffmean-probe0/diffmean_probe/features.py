"""Hidden-state extraction: last prompt token and problem-span mean per layer."""

from __future__ import annotations

import gc
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import SITES, format_problem, problem_token_mask


def load_model(model_id: str) -> tuple[Any, Any]:
    """Load tokenizer and model from the offline HF cache."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map: Any = "auto"
    else:
        dtype, device_map = torch.float32, None
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def release(model: Any) -> None:
    """Drop a model and return its memory to the allocator."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


@torch.no_grad()
def encode_problem(
    tokenizer: Any,
    model: Any,
    problem_text: str,
    model_id: str,
    effort: str,
    layers: Sequence[int] | None = None,
    max_length: int = 4096,
) -> dict[str, np.ndarray]:
    """Return ``{site: (n_layers, hidden)}`` float32 features for one problem.

    ``layers`` indexes ``output.hidden_states`` (0 = embeddings); ``None``
    keeps every layer. ``"mean"`` averages the tokens that overlap the problem
    text, falling back to all prompt tokens if the span cannot be located.
    """
    prompt = format_problem(tokenizer, problem_text, model_id, effort)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")[0].tolist()]
    start = prompt.find(problem_text)
    mask = (
        problem_token_mask(offsets, start, start + len(problem_text))
        if start >= 0
        else [True] * len(offsets)
    )
    if not any(mask):
        mask = [True] * len(offsets)

    device = _input_device(model)
    output = model(
        **{name: value.to(device) for name, value in encoded.items()},
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    hidden_states = output.hidden_states
    selected = range(len(hidden_states)) if layers is None else layers
    mask_tensor = torch.tensor(mask, dtype=torch.bool)

    last, mean = [], []
    for index in selected:
        states = hidden_states[index][0].float().cpu()
        last.append(states[-1])
        mean.append(states[mask_tensor].mean(dim=0))
    return {"last": torch.stack(last).numpy(), "mean": torch.stack(mean).numpy()}


def encode_problems(
    tokenizer: Any,
    model: Any,
    problems: Iterable[str],
    model_id: str,
    effort: str,
    layers: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """Stack :func:`encode_problem` outputs into ``{site: (n, n_layers, hidden)}``."""
    rows = [encode_problem(tokenizer, model, text, model_id, effort, layers) for text in problems]
    return {site: np.stack([row[site] for row in rows]) for site in SITES}
