#!/usr/bin/env python3
"""Turn graded samples into training rows in the ``train-main-v2`` schema.

For each (model, effort, problem)::

    base_accuracy = mean correct over original samples
    accuracy[t]   = mean correct over all variants and samples of type t
    max_drop      = max over types t of (base_accuracy - accuracy[t])
    is_robust     = True if max_drop <= 0.10, False if >= 0.25, else None

Problems the model does not solve reliably on the original
(``base_accuracy < --min-base-accuracy``) are dropped, following the
organizers' rule of dropping rows where "the original model is not robustly
correct". Output: ``<work-dir>/labels.jsonl``, which
``extract_features.py --extra-rows`` appends to the organizer rows.

Run in the project environment: ``uv run datagen/label.py``.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DEFAULT_WORK_DIR, label_for, read_jsonl, write_jsonl  # noqa: E402


def mean(values: list[bool]) -> float:
    return sum(values) / len(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--min-base-accuracy", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problems = {row["problem_id"]: row for row in read_jsonl(args.work_dir / "problems.jsonl")}
    groups: dict[tuple[str, str, str], dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted((args.work_dir / "samples").glob("*.jsonl")):
        if path.name.endswith(".traces.jsonl"):  # full responses, written by sample.py
            continue
        for row in read_jsonl(path):
            key = (row["model_id"], row["reasoning_effort"], row["problem_id"])
            groups[key][row["type"]].extend(row["correct"])

    rows, dropped = [], Counter()
    for (model_id, effort, pid), by_type in sorted(groups.items()):
        base = by_type.get("original")
        variant_types = {t: v for t, v in by_type.items() if t != "original" and v}
        if not base or not variant_types:
            dropped["incomplete"] += 1
            continue
        base_accuracy = mean(base)
        if base_accuracy < args.min_base_accuracy:
            dropped["base_accuracy"] += 1
            continue
        accuracy = {t: mean(v) for t, v in sorted(variant_types.items())}
        max_drop = round(max(base_accuracy - a for a in accuracy.values()), 4)
        rows.append({
            "model_id": model_id,
            "reasoning_effort": effort,
            "problem": problems[pid]["problem"],
            "is_robust": label_for(max_drop),
            "max_drop": max_drop,
            "problem_id": pid,
            "source": problems[pid]["source"],
            "base_accuracy": base_accuracy,
            "type_accuracy": accuracy,
        })

    path = args.work_dir / "labels.jsonl"
    write_jsonl(path, rows)
    counts = Counter((row["model_id"], row["is_robust"]) for row in rows)
    for model_id in sorted({row["model_id"] for row in rows}):
        print(f"{model_id}: robust {counts[(model_id, True)]}, not robust {counts[(model_id, False)]}, "
              f"unlabelled {counts[(model_id, None)]}")
    print(f"dropped: {dict(dropped)}; wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
