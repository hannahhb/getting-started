#!/usr/bin/env python3
"""Prompt-level robustness direction with refusal-style interventions (Arditi et al.).

Each prompt (an original problem or one of its perturbed versions in
``work/``) is one example, labelled by the model's own accuracy on it from
``datagen/sample.py``: failed (accuracy <= --failed-max) or solved
(accuracy >= --solved-min); prompts in between are left out.

1. Candidates: for every layer and position (problem-token mean, last problem
   token, each chat-template token after the problem),
       r[layer, position] = mean(h | failed) - mean(h | solved).
2. Selection: leave-one-PROBLEM-out. Directions are fitted on the other
   problems' prompts and the held-out problem's prompts are scored (centred on
   the training mean); the AUC over all held-out scores asks whether failed
   prompts project higher than solved ones on unseen problems.
3. Interventions (``--causal``), at the best cell or ``--layer/--position``,
   with sampled generation graded against the gold answer:
     * ablate: remove the direction from every layer's output at every token
       while the model works on FAILED prompts (does accuracy recover?);
     * add: add the raw direction to that layer's output on SOLVED prompts
       (does accuracy drop?).
   Baseline accuracies come from the sampling run.

Caveat: failed prompts are mostly organizer rewrites and solved ones are the
original and light edits, so a direction can partly encode "heavily rewritten";
the interventions are what test whether it matters for solving.

Run on a GPU from the repo root, after sampling the prompts::

    python datagen/sample.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \\
        --work-dir solutions/diffmean-probe/research/direction_pilot/work \\
        --samples-dir data/diffmean-probe/direction_pilot/samples --base-samples 4 --variant-samples 4
    python solutions/diffmean-probe/research/direction_pilot/prompt_level.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
    python solutions/diffmean-probe/research/direction_pilot/prompt_level.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --causal
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUTPUTS = ROOT / "data" / "diffmean-probe" / "direction_pilot"
sys.path[:0] = [str(HERE), str(ROOT / "datagen"), str(ROOT / "solutions" / "diffmean-probe")]

from common import SOLVE_SUFFIX, extract_answer, read_jsonl  # noqa: E402
from diffmean_probe.common import normalize_effort, safe_model_id  # noqa: E402
from problem_level import auc, extract  # noqa: E402


def load_prompts(work: Path, samples_dir: Path, model: str, effort: str) -> list[dict]:
    """Original and variant prompts with the model's sampled accuracy."""
    problems = {p["problem_id"]: p for p in read_jsonl(work / "problems.jsonl")}
    prompts = [{"problem_id": pid, "type": "original", "variant": 0, "text": p["problem"]} for pid, p in problems.items()]
    prompts += [{k: v[k] for k in ("problem_id", "type", "variant", "text")} for v in read_jsonl(work / "variants.jsonl")]
    samples_file = samples_dir / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', model)}__{effort}.jsonl"
    samples = {(s["problem_id"], s["type"], s["variant"]): s for s in read_jsonl(samples_file)}
    if not samples:
        raise SystemExit(f"no samples in {samples_file}; run datagen/sample.py --work-dir {work} --samples-dir {samples_dir} first")
    for prompt in prompts:
        s = samples.get((prompt["problem_id"], prompt["type"], prompt["variant"]))
        prompt["accuracy"] = float(np.mean(s["correct"])) if s else None
        prompt["mean_tokens"] = float(np.mean(s["n_tokens"])) if s else None
        prompt["answer"] = problems[prompt["problem_id"]]["answer"]
    return [p for p in prompts if p["accuracy"] is not None]


def lopo_grid(states, failed, groups) -> np.ndarray:
    """(layers, positions) leave-one-problem-out AUC for failed vs solved."""
    scores = np.zeros(states.shape[:3])
    for g in np.unique(groups):
        test, train = groups == g, groups != g
        if not (failed[train].any() and (~failed[train]).any()):
            raise SystemExit("each training split needs both failed and solved prompts")
        direction = states[train & failed].mean(0) - states[train & ~failed].mean(0)
        centre = states[train].mean(0)
        scores[test] = np.einsum("nlpd,lpd->nlp", states[test] - centre, direction)
    return np.array([[auc(scores[:, l, p], failed) for p in range(states.shape[2])] for l in range(states.shape[1])])


def generate(model, tokenizer, prompt_text, n, max_new_tokens, seed):
    import torch

    chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt_text + SOLVE_SUFFIX}],
                                         tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(model.get_input_embeddings().weight.device)
    torch.manual_seed(seed)
    out = model.generate(**encoded, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                         num_return_sequences=n, max_new_tokens=max_new_tokens,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    texts = tokenizer.batch_decode(out[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    lengths = [int((row != (tokenizer.pad_token_id or tokenizer.eos_token_id)).sum()) for row in out[:, encoded["input_ids"].shape[1]:]]
    return texts, lengths


def causal(args, prompts, failed, directions, layer, position, names, out_dir) -> None:
    """Ablate the direction on failed prompts and add it on solved prompts."""
    import torch

    from diffmean_probe.features import _decoder_layers, load_model, release

    tokenizer, model = load_model(args.model)
    blocks = _decoder_layers(model)
    raw = torch.tensor(directions[layer, position])
    unit = raw / raw.norm()
    # hidden_states[k] is the output of block k-1 (index 0 is the embeddings).
    block_index = max(layer - 1, 0)

    def make_hook(mode):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            r = unit.to(hidden.device, hidden.dtype)
            if mode == "ablate":
                hidden = hidden - (hidden @ r).unsqueeze(-1) * r
            else:
                hidden = hidden + args.alpha * raw.to(hidden.device, hidden.dtype)
            return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden
        return hook

    rows = []
    selected = [(i, "ablate") for i in np.flatnonzero(failed)] + [(i, "add") for i in np.flatnonzero(~failed)]
    try:
        for i, mode in selected[: args.max_prompts]:
            prompt = prompts[i]
            handles = ([b.register_forward_hook(make_hook("ablate")) for b in blocks] if mode == "ablate"
                       else [blocks[block_index].register_forward_hook(make_hook("add"))])
            try:
                texts, lengths = generate(model, tokenizer, prompt["text"], args.samples, args.max_new_tokens, args.seed)
            finally:
                for handle in handles:
                    handle.remove()
            correct = [extract_answer(t) == prompt["answer"] for t in texts]
            rows.append({"problem_id": prompt["problem_id"], "type": prompt["type"], "variant": prompt["variant"],
                         "intervention": mode, "baseline_accuracy": round(prompt["accuracy"], 3),
                         "intervened_accuracy": round(float(np.mean(correct)), 3),
                         "baseline_mean_tokens": round(prompt["mean_tokens"] or 0), "intervened_mean_tokens": round(float(np.mean(lengths)))})
            print(f"  {mode:6s} {prompt['problem_id']} {prompt['type']}#{prompt['variant']}: accuracy "
                  f"{prompt['accuracy']:.2f} -> {np.mean(correct):.2f}, tokens {prompt['mean_tokens']:.0f} -> {np.mean(lengths):.0f}", flush=True)
    finally:
        del model
        release()
    with (out_dir / f"causal_layer{layer}_{names[position].split(':')[0]}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for mode in ("ablate", "add"):
        part = [r for r in rows if r["intervention"] == mode]
        if part:
            print(f"{mode}: mean accuracy {np.mean([r['baseline_accuracy'] for r in part]):.2f} -> "
                  f"{np.mean([r['intervened_accuracy'] for r in part]):.2f} over {len(part)} prompts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--work-dir", type=Path, default=HERE / "work")
    parser.add_argument("--samples-dir", type=Path, default=OUTPUTS / "samples")
    parser.add_argument("--effort", default="default")
    parser.add_argument("--failed-max", type=float, default=0.25)
    parser.add_argument("--solved-min", type=float, default=0.75)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--causal", action="store_true", help="Run ablate/add interventions.")
    parser.add_argument("--layer", type=int, help="Intervention layer (default: best cell).")
    parser.add_argument("--position", type=int, help="Intervention position index (default: best cell).")
    parser.add_argument("--alpha", type=float, default=1.0, help="Scale of the added direction.")
    parser.add_argument("--samples", type=int, default=4, help="Samples per prompt under intervention.")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-prompts", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    effort = normalize_effort(args.model, args.effort)
    prompts = load_prompts(args.work_dir, args.samples_dir, args.model, effort)
    keep = [p for p in prompts if p["accuracy"] <= args.failed_max or p["accuracy"] >= args.solved_min]
    counts = defaultdict(lambda: [0, 0])
    for p in keep:
        counts[p["problem_id"]][p["accuracy"] <= args.failed_max] += 1
    print(f"{len(prompts)} sampled prompts, {len(keep)} kept | per problem [solved, failed]: {dict(counts)}")

    out_dir = args.out_dir or OUTPUTS / "results" / f"prompt_level__{safe_model_id(args.model)}__{effort}"
    out_dir.mkdir(parents=True, exist_ok=True)
    act_path = out_dir / "activations.npz"
    texts = [p["text"] for p in keep]
    if act_path.is_file() and json.loads((out_dir / "prompts.json").read_text()) != texts:
        act_path.unlink()  # labels changed which prompts are kept
    if not act_path.is_file():
        extract(args.model, effort, texts, act_path)
        (out_dir / "prompts.json").write_text(json.dumps(texts))
    data = np.load(act_path)
    states, names = data["states"], [str(x) for x in data["names"]]
    failed = np.array([p["accuracy"] <= args.failed_max for p in keep])
    groups = np.array([p["problem_id"] for p in keep])

    grid = lopo_grid(states, failed, groups)
    order = np.dstack(np.unravel_index(np.argsort(-grid, axis=None), grid.shape))[0]
    print("top candidates (leave-one-problem-out AUC; 1.0 = failed prompts always project higher):")
    for layer, p in order[:10]:
        print(f"  layer {layer:2d}  {names[p]:28s}  AUC {grid[layer, p]:.2f}")
    directions = states[failed].mean(0) - states[~failed].mean(0)
    np.savez(out_dir / "directions.npz", directions=directions, auc=grid, names=np.asarray(names))
    with (out_dir / "auc.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["layer"] + names)
        writer.writerows([[layer] + [round(float(v), 3) for v in grid[layer]] for layer in range(grid.shape[0])])
    with (out_dir / "prompts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["problem_id", "type", "variant", "accuracy", "mean_tokens"])
        writer.writeheader()
        writer.writerows([{k: p[k] for k in writer.fieldnames} for p in keep])
    print(f"wrote {out_dir}/auc.csv, prompts.csv, directions.npz")

    if args.causal:
        layer = args.layer if args.layer is not None else int(order[0][0])
        position = args.position if args.position is not None else int(order[0][1])
        print(f"interventions at layer {layer}, {names[position]} (AUC {grid[layer, position]:.2f})")
        causal(args, keep, failed, directions, layer, position, names, out_dir)


if __name__ == "__main__":
    main()
