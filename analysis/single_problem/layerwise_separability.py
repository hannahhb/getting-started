#!/usr/bin/env python3
"""Layer-wise, token-level separability of one problem's perturbed vs original prompts.

For one problem and its hand-written variants (``variants_*.jsonl``), runs the
model once per prompt and keeps the residual-stream state of every problem
token at every layer. Then, per layer, asks whether a linear direction
separates perturbed tokens (class 1) from original tokens (class 0):

* ``all``: every problem token of the original vs every token of a variant.
  Early layers separate these easily because the words differ.
* ``shared``: only tokens present in both the original and the variant
  (aligned with difflib on token ids). The token is identical, so separation
  means the perturbation changed its *context*.

Evaluation holds out one variant at a time (plus one block of original
tokens), so the classifier never sees the variant it is tested on:

* ``logreg_bacc``: balanced accuracy of a standardized logistic regression;
* ``shuffled_bacc``: the same with training labels shuffled (chance control);
* ``diffmean_auc``: AUC of the held-out tokens' projection onto the
  difference of class means (no fitting beyond two means).

``last_token_cos`` is the mean cosine distance between each variant's last
prompt token and the original's, per layer: how far the perturbation has
moved the state the model answers from.

Run on a GPU from the repo root::

    python analysis/single_problem/layerwise_separability.py \\
        --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \\
        --variants analysis/single_problem/variants_17017.jsonl

Writes ``activations.npz`` and ``results.csv`` (plus ``results.png`` if
matplotlib is available) to ``--out-dir``; re-running skips extraction when
``activations.npz`` exists.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "solutions" / "diffmean-probe"))

from diffmean_probe.common import format_problem, normalize_effort, problem_token_mask, safe_model_id  # noqa: E402


def extract(model_id: str, effort: str, rows: list[dict], path: Path) -> None:
    """Save each prompt's problem-token states at every layer, plus token ids."""
    import torch

    from diffmean_probe.features import load_model, release

    tokenizer, model = load_model(model_id)
    arrays = {}
    try:
        for i, row in enumerate(rows):
            prompt = format_problem(tokenizer, row["text"], model_id, effort)
            encoded = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
            offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")[0].tolist()]
            start = prompt.find(row["text"])
            mask = np.asarray(problem_token_mask(offsets, start, start + len(row["text"])))
            device = model.get_input_embeddings().weight.device
            with torch.no_grad():
                output = model(**{k: v.to(device) for k, v in encoded.items()}, output_hidden_states=True,
                               use_cache=False, logits_to_keep=1)
            states = torch.stack([h[0] for h in output.hidden_states]).float().cpu().numpy()  # (L+1, T, d)
            arrays[f"h{i}"] = states[:, mask].astype(np.float16)
            arrays[f"last{i}"] = states[:, -1].astype(np.float32)
            arrays[f"ids{i}"] = encoded["input_ids"][0].numpy()[mask]
            print(f"  prompt {i} ({row['type']} {row['variant']}): {int(mask.sum())} problem tokens", flush=True)
    finally:
        release(model)
    np.savez(path, n=len(rows), **arrays)


def shared_positions(ids_a: np.ndarray, ids_b: np.ndarray) -> tuple[list[int], list[int]]:
    """Aligned positions of identical tokens in two id sequences."""
    matcher = difflib.SequenceMatcher(None, ids_a.tolist(), ids_b.tolist(), autojunk=False)
    a, b = [], []
    for block in matcher.get_matching_blocks():
        a += range(block.a, block.a + block.size)
        b += range(block.b, block.b + block.size)
    return a, b


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([np.mean(pred[y == c] == c) for c in (0, 1) if np.any(y == c)]))


def analyse_layer(original: np.ndarray, variants: list[np.ndarray], blocks: list[np.ndarray], rng) -> dict:
    """Leave-one-variant-out separability for one layer and token selection."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    baccs, shuffled, aucs, recalls = [], [], [], []
    for k, held in enumerate(variants):
        if len(held) == 0:
            recalls.append(np.nan)
            continue
        test_block = blocks[k % len(blocks)]
        train_orig = np.delete(original, test_block, axis=0) if len(test_block) else original
        train_var = np.concatenate([v for j, v in enumerate(variants) if j != k and len(v)])
        x_train = np.concatenate([train_orig, train_var]).astype(np.float32)
        y_train = np.r_[np.zeros(len(train_orig)), np.ones(len(train_var))]
        x_test = np.concatenate([original[test_block], held]).astype(np.float32)
        y_test = np.r_[np.zeros(len(test_block)), np.ones(len(held))]

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        pred = clf.fit(x_train, y_train).predict(x_test)
        baccs.append(balanced_accuracy(y_test, pred))
        recalls.append(float(np.mean(pred[y_test == 1] == 1)))
        control = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        shuffled.append(balanced_accuracy(y_test, control.fit(x_train, rng.permutation(y_train)).predict(x_test)))

        direction = train_var.mean(0) - train_orig.mean(0)
        if len(test_block):
            aucs.append(float(roc_auc_score(y_test, x_test @ direction)))
    return {
        "logreg_bacc": float(np.mean(baccs)) if baccs else np.nan,
        "shuffled_bacc": float(np.mean(shuffled)) if shuffled else np.nan,
        "diffmean_auc": float(np.mean(aucs)) if aucs else np.nan,
        "recalls": recalls,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--effort", default="default")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel layers (CPU cores).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.variants.read_text().splitlines() if line.strip()]
    if rows[0]["type"] != "original":
        raise SystemExit("the first line of --variants must be the original problem")
    effort = normalize_effort(args.model, args.effort)
    out_dir = args.out_dir or args.variants.parent / "results" / f"{args.variants.stem}__{safe_model_id(args.model)}__{effort}"
    out_dir.mkdir(parents=True, exist_ok=True)
    act_path = out_dir / "activations.npz"
    if not act_path.is_file():
        print(f"extracting {len(rows)} prompts with {args.model} ({effort})", flush=True)
        extract(args.model, effort, rows, act_path)

    data = np.load(act_path)
    n = int(data["n"])
    hidden = [data[f"h{i}"] for i in range(n)]
    last = [data[f"last{i}"] for i in range(n)]
    ids = [data[f"ids{i}"] for i in range(n)]
    n_layers = hidden[0].shape[0]
    labels = [f"{r['type']}#{r['variant']}" for r in rows[1:]]
    alignments = [shared_positions(ids[0], ids[i]) for i in range(1, n)]
    def layer_rows(layer: int) -> list[dict]:
        rng = np.random.default_rng(args.seed + layer)
        rows_out = []
        cos = [
            1.0 - float(last[0][layer] @ last[i][layer] / (np.linalg.norm(last[0][layer]) * np.linalg.norm(last[i][layer]) + 1e-8))
            for i in range(1, n)
        ]
        for selection in ("all", "shared"):
            if selection == "all":
                original = hidden[0][layer].astype(np.float32)
                variants = [hidden[i][layer].astype(np.float32) for i in range(1, n)]
            else:
                used = sorted({p for a, _ in alignments for p in a})
                original = hidden[0][layer][used].astype(np.float32)
                variants = [hidden[i][layer][b].astype(np.float32) for i, (_, b) in zip(range(1, n), alignments)]
            blocks = np.array_split(np.arange(len(original)), min(len(variants), len(original)))
            stats = analyse_layer(original, variants, blocks, rng)
            rows_out.append({
                "layer": layer, "tokens": selection,
                "logreg_bacc": round(stats["logreg_bacc"], 3),
                "shuffled_bacc": round(stats["shuffled_bacc"], 3),
                "diffmean_auc": round(stats["diffmean_auc"], 3),
                "last_token_cos": round(float(np.mean(cos)), 4),
                **{f"recall_{name}": round(r, 2) for name, r in zip(labels, stats["recalls"])},
                **{f"cos_{name}": round(c, 4) for name, c in zip(labels, cos)},
            })
        return rows_out

    from joblib import Parallel, delayed

    per_layer = Parallel(n_jobs=args.jobs)(delayed(layer_rows)(layer) for layer in range(n_layers))
    results = [row for rows_out in per_layer for row in rows_out]
    for row_all, row_shared in zip(results[0::2], results[1::2]):
        layer = row_all["layer"]
        print(f"layer {layer:2d} | all: bacc {row_all['logreg_bacc']:.2f} (shuffled {row_all['shuffled_bacc']:.2f}) "
              f"auc {row_all['diffmean_auc']:.2f} | shared: bacc {row_shared['logreg_bacc']:.2f} "
              f"(shuffled {row_shared['shuffled_bacc']:.2f}) auc {row_shared['diffmean_auc']:.2f} "
              f"| last-token cos dist {row_all['last_token_cos']:.4f}", flush=True)

    with (out_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"shared tokens per variant: {dict(zip(labels, [len(b) for _, b in alignments]))}")
    print(f"wrote {out_dir / 'results.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for selection, style in (("all", "-"), ("shared", "--")):
        part = [r for r in results if r["tokens"] == selection]
        axes[0].plot([r["layer"] for r in part], [r["logreg_bacc"] for r in part], style, label=f"{selection} tokens")
        axes[0].plot([r["layer"] for r in part], [r["shuffled_bacc"] for r in part], style, alpha=0.4,
                     label=f"{selection}, shuffled labels")
    axes[0].set(xlabel="layer", ylabel="held-out balanced accuracy", title="Perturbed vs original tokens", ylim=(0.3, 1.02))
    axes[0].legend(fontsize=8)
    part = [r for r in results if r["tokens"] == "all"]
    for name in labels:
        axes[1].plot([r["layer"] for r in part], [r[f"cos_{name}"] for r in part], label=name, linewidth=1)
    axes[1].set(xlabel="layer", ylabel="cosine distance to original", title="Last prompt token")
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "results.png", dpi=130)
    print(f"wrote {out_dir / 'results.png'}")


if __name__ == "__main__":
    main()
