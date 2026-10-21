#!/usr/bin/env python3
"""Assemble perturbed variants of each problem into ``<work-dir>/variants.jsonl``.

* ``typos``: rule-based and deterministic (``--typo-variants`` per problem).
* ``rephrase``, ``rename``, ``domain``, ``distract``, ``expert``: written by
  Claude, one per problem and type, following the organizers' definitions
  (arXiv 2607.13899, Appendix B). Each written variant records in ``check``
  why its answer is unchanged; variants whose answer could not be verified
  were left out. They are stored in the repository at
  ``datagen/written_variants.jsonl`` so the data is reproducible.

Run in the project environment: ``uv run datagen/perturb.py``.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DEFAULT_WORK_DIR, read_jsonl, write_jsonl  # noqa: E402

WRITTEN_VARIANTS = Path(__file__).resolve().parent / "written_variants.jsonl"


def add_typos(text: str, rng: random.Random, rate: float = 0.04) -> str:
    """Swap, drop, or double one letter in a few words, never inside LaTeX or numbers."""
    parts = re.split(r"(\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\])", text, flags=re.DOTALL)
    for index in range(0, len(parts), 2):  # even indices are outside math
        words = re.split(r"(\s+)", parts[index])
        for w, word in enumerate(words):
            if len(word) > 3 and word.isalpha() and rng.random() < rate:
                i = rng.randrange(1, len(word) - 1)
                edit = rng.choice(("swap", "drop", "double"))
                if edit == "swap":
                    word = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
                elif edit == "drop":
                    word = word[:i] + word[i + 1 :]
                else:
                    word = word[:i] + word[i] + word[i:]
                words[w] = word
        parts[index] = "".join(words)
    return "".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--written-variants", type=Path, default=WRITTEN_VARIANTS)
    parser.add_argument("--typo-variants", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problems = list(read_jsonl(args.work_dir / "problems.jsonl"))
    if not problems:
        raise SystemExit(f"no problems in {args.work_dir}; run datagen/build_problems.py first")
    known = {problem["problem_id"] for problem in problems}

    variants = []
    for problem in problems:
        rng = random.Random(f"{args.seed}:{problem['problem_id']}")
        for k in range(args.typo_variants):
            variants.append({"problem_id": problem["problem_id"], "type": "typos", "variant": k,
                             "text": add_typos(problem["problem"], rng)})
    written = [
        {key: row[key] for key in ("problem_id", "type", "variant", "text")}
        for row in read_jsonl(args.written_variants)
        if row["problem_id"] in known
    ]
    variants += written

    path = args.work_dir / "variants.jsonl"
    write_jsonl(path, variants)
    print(f"wrote {len(variants)} variants ({len(written)} written, "
          f"{len(variants) - len(written)} typos) for {len(problems)} problems to {path}")


if __name__ == "__main__":
    main()
