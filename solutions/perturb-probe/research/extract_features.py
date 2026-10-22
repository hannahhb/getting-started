#!/usr/bin/env python3
"""Run the forced-answer perturbation probe over the labelled train cases.

For each model (and each effort it has labels for) every labelled problem is
probed once: X plus 4 code-made perturbations, ``--n`` traces each with a
``--budget``-token thinking budget, forced answers, and answer log-probs.
Results are appended per problem to
``data/perturb-probe/features/<model>__<effort>.jsonl``, so a rerun resumes.

``--with-uncertainty`` also attaches the uncertainty-profiling baseline's
collector to X's traces and stores its 14 features (prefixed ``u_``), at no
extra generation cost.

Run on a GPU from the repo root::

    python solutions/perturb-probe/research/extract_features.py --models deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(ROOT / "solutions" / "uncertainty-profiling")]

from perturb_probe.probe import _eos_ids, load_model, probe_problem, release  # noqa: E402

OUT = ROOT / "data" / "perturb-probe" / "features"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def uncertainty(collector, raw, tokenizer, model) -> dict:
    from uncertainty_profile.config import FEATURE_NAMES, GenerationConfidenceConfig

    sequences, prompt_len = raw
    rows, _ = collector.compute_features(sequences=sequences, prompt_length=prompt_len,
                                         pad_token_id=tokenizer.pad_token_id,
                                         eos_token_ids=set(_eos_ids(tokenizer, model)),
                                         config=GenerationConfidenceConfig())
    rows = [dict(row) for row in rows]
    return {f"u_{name}": float(sum(r[name] for r in rows) / len(rows)) for name in FEATURE_NAMES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--cases", type=Path, default=HERE / "train_set" / "cases.jsonl")
    parser.add_argument("--problems", type=Path, default=HERE / "train_set" / "problems.jsonl")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--n", type=int, default=4, help="Traces per prompt.")
    parser.add_argument("--budget", type=int, default=4096, help="Thinking tokens before the answer is forced.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-uncertainty", action="store_true")
    args = parser.parse_args()

    problems = {p["problem_id"]: p["problem"] for p in read_jsonl(args.problems)}
    cases = read_jsonl(args.cases)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for model_id in args.models:
        efforts = sorted({c["reasoning_effort"] for c in cases if c["model_id"] == model_id})
        if not efforts:
            print(f"{model_id}: no labelled cases, skipping")
            continue
        tokenizer, model = load_model(model_id)
        try:
            for effort in efforts:
                out = args.out_dir / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', model_id)}__{effort}.jsonl"
                done = {r["problem_id"] for r in read_jsonl(out)}
                todo = sorted({c["problem_id"] for c in cases if c["model_id"] == model_id
                               and c["reasoning_effort"] == effort} - done)
                print(f"{model_id} {effort}: {len(todo)} problems to probe ({len(done)} done) -> {out}", flush=True)
                for k, pid in enumerate(todo, 1):
                    start = time.time()
                    collector = None
                    if args.with_uncertainty:
                        from uncertainty_profile.extraction import GenerationConfidenceCollector
                        collector = GenerationConfidenceCollector()
                    features = probe_problem(model, tokenizer, model_id, effort, problems[pid], n=args.n,
                                             budget=args.budget, seed=args.seed, x_processor=collector)
                    raw = features.pop("x_raw")
                    if collector is not None:
                        features.update(uncertainty(collector, raw, tokenizer, model))
                    row = {"model_id": model_id, "reasoning_effort": effort, "problem_id": pid,
                           "n": args.n, "budget": args.budget, "seconds": round(time.time() - start, 1), **features}
                    with out.open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    print(f"  {k}/{len(todo)} {pid}: reference {features['reference']}, agree {features.get('agree_mean')}, "
                          f"dlogp {features.get('dlogp')}, {row['seconds']}s", flush=True)
        finally:
            del model
            release()


if __name__ == "__main__":
    main()
