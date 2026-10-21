"""Hidden-state extraction: last prompt token and problem-span mean per layer."""

from __future__ import annotations

import gc
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import SITES, format_problem, problem_token_mask

# GPU memory left free for activations when ``device_map="auto"`` places
# weights; without it the placement can fill a card and the first forward
# pass runs out of memory.
ACTIVATION_HEADROOM_BYTES = 6 * 1024**3


def _max_memory() -> dict[Any, int] | None:
    """Per-GPU weight budget: currently free memory minus activation headroom."""
    budget: dict[Any, int] = {}
    for index in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(index)
        budget[index] = max(free - ACTIVATION_HEADROOM_BYTES, 0)
    if not budget:
        return None
    # Weights that don't fit spill to CPU RAM (slow, but works on smaller GPUs).
    budget["cpu"] = 1024**4
    return budget


def load_model(model_id: str) -> tuple[Any, Any]:
    """Load tokenizer and model from the offline HF cache."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map: Any = "auto"
        max_memory = _max_memory()
    else:
        dtype, device_map, max_memory = torch.float32, None, None
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
        max_memory=max_memory,
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
        # Only hidden states are needed; full-sequence logits over a ~200k
        # vocabulary are the largest activation and caused OOM on gpt-oss.
        logits_to_keep=1,
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
