#!/usr/bin/env python3
"""Score the response-length method on the evaluation set (``eval_set/``).

The evaluation set holds every labelled train-main-v2 row (the competition
models) and the val-sample rows for the two runnable aliases
(``gpt-oss-120b:low`` -> openai/gpt-oss-120b low, ``qwen3-8b:low`` ->
DeepSeek-R1-0528-Qwen3-8B). Larger and closed models are left out.

1. Sample the originals for each model/effort into ``data/length-probe/eval_set/samples``
   (no gold answers are needed; only response lengths are used)::

       # small models, on a GPU with vLLM
       python datagen/sample.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \\
           --work-dir solutions/length-probe/research/eval_set \\
           --samples-dir data/length-probe/eval_set/samples --base-samples 4 --max-tokens 8192
       # gpt-oss, through OpenRouter
       python datagen/sample.py --backend openrouter --model openai/gpt-oss-120b --effort low \\
           --work-dir solutions/length-probe/research/eval_set \\
           --samples-dir data/length-probe/eval_set/samples --base-samples 4 --max-tokens 8192

2. Run this script. Per model/effort, the log mean response length is
   normalised with that model/effort's own median and spread over every
   sampled problem (no labels used), so one threshold fits all models. That
   threshold is fitted by logistic regression, leaving out one problem at a
   time; a case is predicted robust when its response is short.

Reports accuracy against always-True per source, per model/effort, and
overall, plus the length AUC where both classes are present.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "datagen")]
SAMPLES = ROOT / "data" / "length-probe" / "eval_set" / "samples"

from blackbox_signals import auc  # noqa: E402
from common import read_jsonl  # noqa: E402


def normalised_lengths(samples_dir: Path) -> dict[tuple[str, str], dict[str, float]]:
    """{(model, effort): {problem_id: z}} with label-free per-model/effort scaling."""
    out = {}
    for path in sorted(samples_dir.glob("*.jsonl")):
        if path.name.endswith(".traces.jsonl"):
            continue
        rows = [s for s in read_jsonl(path) if s["type"] == "original" and s["n_tokens"]]
        if not rows:
            continue
        logs = {s["problem_id"]: math.log(mean(s["n_tokens"]) + 1) for s in rows}
        values = np.array(list(logs.values()))
        centre = float(np.median(values))
        spread = float(np.subtract(*np.percentile(values, [75, 25]))) or 1.0
        out[(rows[0]["model_id"], rows[0]["reasoning_effort"])] = {p: (v - centre) / spread for p, v in logs.items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=HERE / "eval_set")
    parser.add_argument("--samples-dir", type=Path, default=SAMPLES)
    parser.add_argument("--source", help="Score only this source (e.g. train-main-v2).")
    args = parser.parse_args()
    from sklearn.linear_model import LogisticRegression

    z = normalised_lengths(args.samples_dir)
    cases = [c for c in read_jsonl(args.work_dir / "cases.jsonl")
             if (c["model_id"], c["reasoning_effort"]) in z and c["problem_id"] in z[(c["model_id"], c["reasoning_effort"])]
             and (args.source is None or c["source"] == args.source)]
    if not cases:
        raise SystemExit("no sampled cases yet: sample the originals into data/length-probe/eval_set/samples first")
    x = np.array([[z[(c["model_id"], c["reasoning_effort"])][c["problem_id"]]] for c in cases])
    y = np.array([bool(c["is_robust"]) for c in cases])
    groups = np.array([c["problem_id"] for c in cases])

    pred = np.zeros(len(y), dtype=bool)
    for g in np.unique(groups):
        test = groups == g
        train_y = y[~test]
        if train_y.all() or not train_y.any():
            pred[test] = train_y.mean() >= 0.5
            continue
        pred[test] = LogisticRegression().fit(x[~test], train_y).predict(x[test])

    def line(name, mask):
        yy, pp, xx = y[mask], pred[mask], x[mask, 0]
        both = yy.any() and not yy.all()
        return (f"{name:46s} n={mask.sum():3d} robust={yy.sum():3d} | accuracy {np.mean(pp == yy):.2f} "
                f"vs always-True {yy.mean():.2f}" + (f" | length AUC {auc(-xx, yy):.2f}" if both else ""))

    print(f"sampled model/efforts: {sorted(z)}\n")
    print(line("ALL", np.ones(len(y), dtype=bool)))
    sources = np.array([c["source"] for c in cases])
    for s in sorted(set(sources)):
        print(line(f"source {s}", sources == s))
    keys = defaultdict(list)
    for i, c in enumerate(cases):
        keys[(c["source"], c["model_id"].split("/")[-1], c["reasoning_effort"])].append(i)
    print()
    for key in sorted(keys):
        mask = np.zeros(len(y), dtype=bool)
        mask[keys[key]] = True
        print(line(" ".join(key), mask))


if __name__ == "__main__":
    main()
