#!/usr/bin/env python3
"""Problem-level robustness direction, following Arditi et al. (refusal direction).

Contrast: problems the model is NOT robust on vs problems it is robust on
(labels in ``problems_deepseek.jsonl``). For every layer and every candidate
position, the candidate direction is the difference of means

    r[layer, position] = mean(h | not robust) - mean(h | robust)

Positions: each chat-template token after the problem (where the model
"decides" how to respond, as in the refusal paper), the last problem token,
and the mean over the problem's tokens.

Each candidate is scored by leave-one-problem-out AUC: the direction is
recomputed without the held-out problem, the held-out problem is projected
onto it, and the AUC asks whether not-robust problems project higher. With
~300 candidates and 12 problems the best AUC is inflated by chance, so a label
permutation test (the best AUC over all candidates under shuffled labels)
gives the reference to read it against.

Run on a GPU from the repo root::

    python solutions/diffmean-probe/research/direction_pilot/problem_level.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

Writes to ``data/diffmean-probe/direction_pilot/results/``: ``activations.npz``, ``auc.csv``, ``directions.npz`` (every candidate,
fitted on all problems) and ``auc.png`` if matplotlib is available. Re-running
skips extraction when ``activations.npz`` exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "solutions" / "diffmean-probe"))
OUTPUTS = ROOT / "data" / "diffmean-probe" / "direction_pilot"

from diffmean_probe.common import format_problem, normalize_effort, problem_token_mask, safe_model_id  # noqa: E402

HERE = Path(__file__).resolve().parent


def extract(model_id: str, effort: str, problems: list[str], path: Path) -> None:
    """Save (n, layers, positions, d) states and the position names."""
    import torch

    from diffmean_probe.features import load_model, release

    tokenizer, model = load_model(model_id)
    states, names = [], None
    try:
        for i, text in enumerate(problems):
            prompt = format_problem(tokenizer, text, model_id, effort)
            encoded = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
            offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")[0].tolist()]
            start = prompt.find(text)
            end = start + len(text)
            mask = np.asarray(problem_token_mask(offsets, start, end))
            post = [j for j, (s, _) in enumerate(offsets) if s >= end]  # template tokens after the problem
            device = model.get_input_embeddings().weight.device
            with torch.no_grad():
                output = model(**{k: v.to(device) for k, v in encoded.items()}, output_hidden_states=True,
                               use_cache=False, logits_to_keep=1)
            h = torch.stack([s[0] for s in output.hidden_states]).float().cpu().numpy()  # (L+1, T, d)
            last_problem = int(np.flatnonzero(mask)[-1])
            columns = [h[:, mask].mean(axis=1), h[:, last_problem]] + [h[:, j] for j in post]
            ids = encoded["input_ids"][0].tolist()
            these = ["problem_mean", "problem_last"] + [f"post{k}:{tokenizer.decode([ids[j]])!r}" for k, j in enumerate(post)]
            if names is None:
                names = these
            elif len(these) != len(names):
                raise RuntimeError("problems have different numbers of template tokens after the text")
            states.append(np.stack(columns, axis=1).astype(np.float32))  # (L+1, P, d)
            print(f"  problem {i}: {int(mask.sum())} problem tokens, {len(post)} template tokens after", flush=True)
    finally:
        del model
        release()
    np.savez(path, states=np.stack(states), names=np.asarray(names))


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    pos, neg = scores[positive], scores[~positive]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def loo_auc_grid(states: np.ndarray, not_robust: np.ndarray) -> np.ndarray:
    """(layers, positions) leave-one-problem-out AUC of the difference-of-means direction."""
    n = len(states)
    scores = np.zeros((n,) + states.shape[1:3])
    for i in range(n):
        keep = np.arange(n) != i
        pos, neg = states[keep & not_robust], states[keep & ~not_robust]
        direction = pos.mean(0) - neg.mean(0)                     # (L, P, d)
        midpoint = 0.5 * (pos.mean(0) + neg.mean(0))
        scores[i] = np.einsum("lpd,lpd->lp", states[i] - midpoint, direction)
    grid = np.zeros(states.shape[1:3])
    for layer in range(grid.shape[0]):
        for p in range(grid.shape[1]):
            grid[layer, p] = auc(scores[:, layer, p], not_robust)
    return grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--problems", type=Path, default=HERE / "problems_deepseek.jsonl")
    parser.add_argument("--effort", default="default")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.problems.read_text().splitlines() if line.strip()]
    effort = normalize_effort(args.model, args.effort)
    out_dir = args.out_dir or OUTPUTS / "results" / f"problem_level__{safe_model_id(args.model)}__{effort}"
    out_dir.mkdir(parents=True, exist_ok=True)
    act_path = out_dir / "activations.npz"
    if not act_path.is_file():
        print(f"extracting {len(rows)} problems with {args.model} ({effort})", flush=True)
        extract(args.model, effort, [r["problem"] for r in rows], act_path)

    data = np.load(act_path)
    states, names = data["states"], [str(x) for x in data["names"]]
    not_robust = np.asarray([not r["robust"] for r in rows])
    print(f"{len(rows)} problems: {int(not_robust.sum())} not robust, {int((~not_robust).sum())} robust | "
          f"{states.shape[1]} layers x {states.shape[2]} positions = {states.shape[1] * states.shape[2]} candidates")

    grid = loo_auc_grid(states, not_robust)
    rng = np.random.default_rng(args.seed)
    null_best = np.array([loo_auc_grid(states, rng.permutation(not_robust)).max() for _ in range(args.permutations)])
    best = float(grid.max())
    p_value = float((1 + (null_best >= best).sum()) / (1 + len(null_best)))

    order = np.dstack(np.unravel_index(np.argsort(-grid, axis=None), grid.shape))[0]
    print("\ntop candidates (leave-one-problem-out AUC; 1.0 = not-robust problems always project higher):")
    for layer, p in order[:10]:
        print(f"  layer {layer:2d}  {names[p]:28s}  AUC {grid[layer, p]:.2f}")
    print(f"\nbest AUC {best:.2f} | best AUC under shuffled labels: median {np.median(null_best):.2f}, "
          f"95th percentile {np.percentile(null_best, 95):.2f} | permutation p = {p_value:.3f}")
    print("per position, best layer:", {names[p]: (int(grid[:, p].argmax()), round(float(grid[:, p].max()), 2))
                                        for p in range(len(names))})

    with (out_dir / "auc.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["layer"] + names)
        for layer in range(grid.shape[0]):
            writer.writerow([layer] + [round(float(v), 3) for v in grid[layer]])
    np.savez(out_dir / "directions.npz",
             directions=states[not_robust].mean(0) - states[~not_robust].mean(0),
             auc=grid, names=np.asarray(names), null_best=null_best)
    print(f"wrote {out_dir / 'auc.csv'} and directions.npz")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(names)), 6))
    image = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1, origin="lower")
    ax.set(xticks=range(len(names)), xlabel="position", ylabel="layer",
           title=f"Leave-one-problem-out AUC (best {best:.2f}, shuffled-label 95th pct "
                 f"{np.percentile(null_best, 95):.2f})")
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    fig.colorbar(image, ax=ax, label="AUC")
    fig.tight_layout()
    fig.savefig(out_dir / "auc.png", dpi=130)
    print(f"wrote {out_dir / 'auc.png'}")


if __name__ == "__main__":
    main()
