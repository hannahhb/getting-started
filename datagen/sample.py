#!/usr/bin/env python3
"""Sample a target model on every original problem and variant, and grade each answer.

Originals get ``--base-samples`` samples and each variant gets
``--variant-samples``, at temperature 1.0 with the model's own chat template.
Truncated generations count as wrong. Results are appended per prompt to
``<work-dir>/samples/<model>__<effort>.jsonl``, so an interrupted run resumes.

Run in a vLLM environment, once per model (and per effort for gpt-oss)::

    python datagen/sample.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
    python datagen/sample.py --model openai/gpt-oss-120b --effort low
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DEFAULT_WORK_DIR, SOLVE_SUFFIX, append_jsonl, extract_answer, read_jsonl  # noqa: E402


def samples_path(work_dir: Path, model: str, effort: str) -> Path:
    return work_dir / "samples" / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', model)}__{effort}.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", default="default", choices=["default", "low", "medium", "high"])
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--base-samples", type=int, default=8)
    parser.add_argument("--variant-samples", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--chunk-size", type=int, default=64, help="Prompts per vLLM call (resume granularity).")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problems = {row["problem_id"]: row for row in read_jsonl(args.work_dir / "problems.jsonl")}
    jobs = [(pid, "original", 0, row["problem"], args.base_samples) for pid, row in problems.items()]
    jobs += [
        (row["problem_id"], row["type"], row["variant"], row["text"], args.variant_samples)
        for row in read_jsonl(args.work_dir / "variants.jsonl")
        if row["problem_id"] in problems
    ]
    out = samples_path(args.work_dir, args.model, args.effort)
    done = {(row["problem_id"], row["type"], row["variant"]) for row in read_jsonl(out)}
    jobs = [job for job in jobs if job[:3] not in done]
    print(f"{args.model} ({args.effort}): {len(jobs)} prompts to sample, {len(done)} already done")
    if not jobs:
        return

    from vllm import LLM, SamplingParams

    llm = LLM(args.model, tensor_parallel_size=args.tensor_parallel_size, seed=args.seed,
              max_model_len=args.max_tokens + 2048, gpu_memory_utilization=args.gpu_memory_utilization)
    tokenizer = llm.get_tokenizer()
    template_kwargs = {"reasoning_effort": args.effort} if args.effort != "default" else {}

    for start in range(0, len(jobs), args.chunk_size):
        chunk = jobs[start : start + args.chunk_size]
        prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": text + SOLVE_SUFFIX}],
                                          tokenize=False, add_generation_prompt=True, **template_kwargs)
            for _, _, _, text, _ in chunk
        ]
        params = [
            SamplingParams(n=n, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=args.max_tokens,
                           seed=args.seed)
            for *_, n in chunk
        ]
        rows = []
        for (pid, kind, variant, _, _), output in zip(chunk, llm.generate(prompts, params)):
            answers = [
                extract_answer(c.text) if c.finish_reason == "stop" else None for c in output.outputs
            ]
            gold = problems[pid]["answer"]
            rows.append({
                "model_id": args.model, "reasoning_effort": args.effort,
                "problem_id": pid, "type": kind, "variant": variant,
                "answers": answers,
                "correct": [answer == gold for answer in answers],
                "n_tokens": [len(c.token_ids) for c in output.outputs],
            })
        append_jsonl(out, rows)
        print(f"  {min(start + args.chunk_size, len(jobs))}/{len(jobs)} prompts")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
