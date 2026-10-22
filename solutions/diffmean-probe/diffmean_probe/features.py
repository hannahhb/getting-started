"""Model signals: prompt hidden states per layer, and short-resample behaviour.

* :func:`encode_problem` runs one forward pass and returns the last prompt
  token and the problem-span mean at every layer.
* :func:`behaviour_features` samples a few short continuations of the prompt
  and summarizes how much they disagree (see ``BEHAVIOUR_FEATURES``).
"""

from __future__ import annotations

import gc
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import SITES, format_problem, problem_token_mask

# Short-resample settings, shared by extraction and inference.
BEHAVIOUR_SAMPLES = 8
BEHAVIOUR_NEW_TOKENS = 64
BEHAVIOUR_EARLY_TOKENS = 16
BEHAVIOUR_SEED = 0
BEHAVIOUR_FEATURES = (
    "dispersion",  # 1 - mean pairwise cosine of continuation mean states (mid layer)
    "early_dispersion",  # the same over the first BEHAVIOUR_EARLY_TOKENS tokens
    "entropy",  # mean next-token entropy over sampled steps
    "nll",  # mean negative log-likelihood of the sampled tokens
)

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


def release(*_unused: Any) -> None:
    """Return freed GPU memory to the device.

    Call after the caller has dropped its own references (``del model``):
    deleting a parameter here cannot free a model the caller still holds.
    """
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


def _decoder_layers(model: Any) -> torch.nn.ModuleList:
    """The transformer block list, found by name so it works across architectures."""
    for name, module in model.named_modules():
        if name.endswith("layers") and isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            return module
    raise RuntimeError("could not locate the model's decoder layers")


def _dispersion(states: torch.Tensor, mask: torch.Tensor) -> float:
    """1 - mean pairwise cosine between masked mean states, (n, t, d) and (n, t)."""
    weights = mask.float().unsqueeze(-1)
    means = (states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
    unit = torch.nn.functional.normalize(means, dim=-1)
    similarity = unit @ unit.T
    n = similarity.shape[0]
    off_diagonal = (similarity.sum() - similarity.diagonal().sum()) / max(n * (n - 1), 1)
    return float(1.0 - off_diagonal)


@torch.no_grad()
def behaviour_features(
    tokenizer: Any,
    model: Any,
    problem_text: str,
    model_id: str,
    effort: str,
    max_length: int = 4096,
) -> np.ndarray:
    """Sample short continuations and return ``BEHAVIOUR_FEATURES`` as float32.

    Sampling is at temperature 1 with no top-k/top-p, from a fixed seed, so
    the features are reproducible on the same hardware.
    """
    prompt = format_problem(tokenizer, problem_text, model_id, effort)
    encoded = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_length
    )
    device = _input_device(model)
    encoded = {name: value.to(device) for name, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]

    layers = _decoder_layers(model)
    captured: list[torch.Tensor] = []

    def keep_last_position(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[:, -1, :].float().cpu())

    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    handle = layers[len(layers) // 2].register_forward_hook(keep_last_position)
    try:
        torch.manual_seed(BEHAVIOUR_SEED)
        output = model.generate(
            **encoded,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            num_return_sequences=BEHAVIOUR_SAMPLES,
            max_new_tokens=BEHAVIOUR_NEW_TOKENS,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=pad,
        )
    finally:
        handle.remove()

    tokens = output.sequences[:, prompt_length:].cpu()
    steps = tokens.shape[1]
    # Each forward call adds one capture; the first (prefill) state predicts token 0.
    states = torch.stack(captured[:steps], dim=1)
    # Valid positions: up to and including the first end-of-sequence token.
    eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
    is_eos = torch.zeros_like(tokens, dtype=torch.bool)
    for token_id in eos_ids:
        is_eos |= tokens == token_id
    mask = (is_eos.cumsum(dim=1) - is_eos.long()) == 0

    logits = torch.stack(output.logits, dim=1).float().cpu()  # (n, steps, vocab)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    nll = -log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    valid = mask.float()
    total = valid.sum().clamp(min=1.0)
    early = mask.clone()
    early[:, BEHAVIOUR_EARLY_TOKENS:] = False
    return np.asarray(
        [
            _dispersion(states, mask),
            _dispersion(states, early),
            float((entropy * valid).sum() / total),
            float((nll * valid).sum() / total),
        ],
        dtype=np.float32,
    )


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
