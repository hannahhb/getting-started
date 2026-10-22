"""Forced-answer perturbation probe.

For a problem X and its four code-made perturbations dX (``perturbations.py``):

1. Sample ``n`` reasoning traces per prompt (X and each dX), stopped at the
   model's end-of-thinking marker or after ``budget`` new tokens.
2. Force an answer: close the reasoning with the model's own marker and append
   "The final answer is $\\boxed{", then read the answer greedily.
3. Score the reference answer r (the most common forced answer on X) by its
   log-probability after the same forced prefix, for every trace.

Features compare X with dX: agreement of forced answers with r, and the drop
in log p(r) from X to dX. No gold answer is used.
"""

from __future__ import annotations

import gc
import math
import re
from collections import Counter
from statistics import mean
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .perturbations import make_perturbations

ALIASES = {
    "gpt-oss-120b:low": ("openai/gpt-oss-120b", "low"),
    "qwen3-8b:low": ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "default"),
}
SOLVE = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
ANSWER_LEAD = "The final answer is $\\boxed{"
SUFFIX = {
    "think": "\n</think>\n\n" + ANSWER_LEAD,
    "harmony": "<|end|><|start|>assistant<|channel|>final<|message|>" + ANSWER_LEAD,
    "plain": "\n\n" + ANSWER_LEAD,
}
FEATURES = ("x_consistency", "agree_mean", "agree_min", "lp_x", "lp_dx", "dlogp", "dlogp_max",
            "finish_x", "finish_dx", "tokens_x")
HEADROOM_BYTES = 6 * 1024**3


def resolve(model_id: str, effort: str | None) -> tuple[str, str]:
    """Map Codabench aliases and default efforts to (checkpoint, effort)."""
    if model_id in ALIASES:
        return ALIASES[model_id]
    effort = (effort or "default").strip().lower() or "default"
    if effort == "default" and "gpt-oss" in model_id:
        effort = "medium"
    return model_id, effort


def load_model(model_id: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"local_files_only": True}
    if torch.cuda.is_available():
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"
        budget = {i: max(torch.cuda.mem_get_info(i)[0] - HEADROOM_BYTES, 0) for i in range(torch.cuda.device_count())}
        kwargs["max_memory"] = {**budget, "cpu": 1024**4}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
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


def style_of(tokenizer) -> str:
    vocab = tokenizer.get_vocab()
    if "<|channel|>" in vocab:
        return "harmony"
    if "</think>" in vocab or "<think>" in (tokenizer.chat_template or ""):
        return "think"
    return "plain"


def chat_prompt(tokenizer, text: str, effort: str) -> str:
    kwargs = {"reasoning_effort": effort} if effort in ("low", "medium", "high") else {}
    return tokenizer.apply_chat_template([{"role": "user", "content": text + SOLVE}], tokenize=False,
                                         add_generation_prompt=True, **kwargs)


def _device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def _eos_ids(tokenizer, model) -> list[int]:
    ids = {tokenizer.eos_token_id}
    configured = getattr(model.generation_config, "eos_token_id", None)
    ids.update(configured if isinstance(configured, list) else [configured])
    for token in ("<|end|>", "<|return|>", "<|call|>", "<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
            ids.add(token_id)
    return sorted(i for i in ids if i is not None)


def _cut(text: str, style: str, tokenizer) -> tuple[str, bool]:
    """Reasoning text up to the end-of-thinking marker, and whether it was reached."""
    marker = {"think": "</think>", "harmony": "<|end|>"}.get(style)
    finished = False
    if marker and marker in text:
        text, finished = text.split(marker, 1)[0], True
    for special in {tokenizer.pad_token, tokenizer.eos_token, "<|return|>", "<|im_end|>", "<|endoftext|>"} - {None}:
        if special in text:
            text, finished = text.split(special, 1)[0], True
    return text, finished


@torch.no_grad()
def sample_reasoning(model, tokenizer, prompts: list[str], n: int, budget: int, style: str, seed: int,
                     x_processor=None):
    """n traces per prompt. Returns (texts, finished, n_tokens, x_raw) with x_raw for prompt 0."""
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(_device(model))
    eos = _eos_ids(tokenizer, model)
    kwargs = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, num_return_sequences=n,
                  max_new_tokens=budget, pad_token_id=tokenizer.pad_token_id, eos_token_id=eos)
    if style == "think":
        kwargs.update(stop_strings=["</think>"], tokenizer=tokenizer)
    if x_processor is not None:
        from transformers.generation.logits_process import LogitsProcessorList
        kwargs["logits_processor"] = LogitsProcessorList([x_processor])
    torch.manual_seed(seed)
    out = model.generate(**encoded, **kwargs)
    prompt_len = encoded["input_ids"].shape[1]
    generated = out[:, prompt_len:]
    texts, finished, n_tokens = [], [], []
    for row in generated:
        text, done = _cut(tokenizer.decode(row, skip_special_tokens=False), style, tokenizer)
        texts.append(text)
        finished.append(done)
        n_tokens.append(int((row != tokenizer.pad_token_id).sum()))
    return texts, finished, n_tokens, (out, prompt_len)


def _parse_answer(text: str) -> int | None:
    head = text.split("}", 1)[0].replace("\\!", "").replace("{,}", "").replace("\\,", "").strip().strip("$")
    if "=" in head:
        head = head.rsplit("=", 1)[1]
    match = re.fullmatch(r"\s*([+-]?\d[\d,]*)(?:\.0+)?\s*", head)
    if not match:
        return None
    digits = match.group(1)
    if "," in digits and not re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+", digits):
        return None
    return int(digits.replace(",", ""))


@torch.no_grad()
def forced_answers(model, tokenizer, contexts: list[str]) -> list[int | None]:
    encoded = tokenizer(contexts, return_tensors="pt", padding=True, add_special_tokens=False).to(_device(model))
    out = model.generate(**encoded, do_sample=False, max_new_tokens=16, pad_token_id=tokenizer.pad_token_id)
    texts = tokenizer.batch_decode(out[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    return [_parse_answer(t) for t in texts]


@torch.no_grad()
def answer_logprobs(model, tokenizer, contexts: list[str], answer: int, batch: int = 8) -> list[float]:
    """log p("<answer>}" | context) for each context."""
    continuation = tokenizer(f"{answer}}}", add_special_tokens=False)["input_ids"]
    scores = []
    for start in range(0, len(contexts), batch):
        rows = [tokenizer(c, add_special_tokens=False)["input_ids"] + continuation for c in contexts[start:start + batch]]
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, width - len(row):] = torch.tensor(row)
            mask[i, width - len(row):] = 1
        logits = model(input_ids=ids.to(_device(model)), attention_mask=mask.to(_device(model))).logits.float()
        logp = torch.log_softmax(logits[:, -len(continuation) - 1:-1], dim=-1)
        target = torch.tensor(continuation, device=logp.device)
        scores += logp.gather(-1, target.view(1, -1, 1).expand(len(rows), -1, 1)).squeeze(-1).sum(-1).tolist()
    return scores


def probe_problem(model, tokenizer, model_id: str, effort: str, text: str, *, n: int = 4, budget: int = 4096,
                  seed: int = 0, x_processor=None) -> dict:
    """Features for one problem (see FEATURES); x_raw holds X's raw generation for extra collectors."""
    style = style_of(tokenizer)
    variants = [("original", text)] + make_perturbations(text, seed)
    prompts = [chat_prompt(tokenizer, t, effort) for _, t in variants]
    # X first and alone when a collector needs its generation; otherwise everything in one batch.
    if x_processor is not None:
        x = sample_reasoning(model, tokenizer, prompts[:1], n, budget, style, seed, x_processor)
        rest = sample_reasoning(model, tokenizer, prompts[1:], n, budget, style, seed + 1)
        texts, finished, tokens = x[0] + rest[0], x[1] + rest[1], x[2] + rest[2]
        x_raw = x[3]
    else:
        texts, finished, tokens, _ = sample_reasoning(model, tokenizer, prompts, n, budget, style, seed)
        x_raw = None
    owner = [i for i in range(len(prompts)) for _ in range(n)]
    contexts = [prompts[owner[j]] + texts[j] + SUFFIX[style] for j in range(len(texts))]
    answers = forced_answers(model, tokenizer, contexts)

    x_idx = [j for j in range(len(contexts)) if owner[j] == 0]
    x_answers = [answers[j] for j in x_idx if answers[j] is not None]
    reference = Counter(x_answers).most_common(1)[0][0] if x_answers else None
    features = {
        "finish_x": mean(finished[j] for j in x_idx),
        "finish_dx": mean(finished[j] for j in range(len(contexts)) if owner[j] != 0),
        "tokens_x": math.log(mean(tokens[j] for j in x_idx) + 1),
        "perturbations": [kind for kind, _ in variants[1:]],
        "reference": reference,
        "answers": answers,
    }
    if reference is None:
        features.update({k: float("nan") for k in ("x_consistency", "agree_mean", "agree_min", "lp_x", "lp_dx",
                                                    "dlogp", "dlogp_max")})
        return {**features, "x_raw": x_raw}
    logps = answer_logprobs(model, tokenizer, contexts, reference)
    per_variant = {i: [j for j in range(len(contexts)) if owner[j] == i] for i in range(1, len(prompts))}
    agree = {i: mean(answers[j] == reference for j in js) for i, js in per_variant.items()}
    lp = {i: mean(logps[j] for j in js) for i, js in per_variant.items()}
    lp_x = mean(logps[j] for j in x_idx)
    features.update({
        "x_consistency": mean(answers[j] == reference for j in x_idx),
        "agree_mean": mean(agree.values()),
        "agree_min": min(agree.values()),
        "lp_x": lp_x,
        "lp_dx": mean(lp.values()),
        "dlogp": lp_x - mean(lp.values()),
        "dlogp_max": max(lp_x - v for v in lp.values()),
        "logps": logps,
    })
    return {**features, "x_raw": x_raw}


def feature_vector(features: dict, names=FEATURES) -> np.ndarray:
    return np.array([float(features.get(name, float("nan"))) for name in names])
