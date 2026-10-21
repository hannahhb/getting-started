#!/usr/bin/env python3
"""Collect integer-answer olympiad problems at pinned revisions.

Sources:

* AIME 2022-24 and AMC 2022-23 (AI-MO validation sets), all kept.
* Omni-MATH-2 (``martheballon/Omni-MATH-2``), the manually revised Omni-MATH.
  Problems tagged non-standard (proof, image, estimation, should delete) and
  answers that are not a single integer are skipped. The problems used are
  pinned by id in ``datagen/omni2_ids.json``, so the set matches the written
  variants in ``datagen/written_variants.jsonl``. Without that file, Omni-MATH-2
  problems at ``--min-difficulty`` or above are sampled with ``--seed`` to fill
  up to ``--n-problems``.

Problems that also appear in ``aimo-interp/train-main-v2`` are dropped, as
are the problems listed in ``datagen/excluded_problems.json`` (wrong stated
answers or ill-posed statements found while writing variants).
Output: ``<work-dir>/problems.jsonl``.

Run in the project environment: ``uv run datagen/build_problems.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "datagen"), str(ROOT / "scripts")]

from common import DEFAULT_WORK_DIR, normalize, parse_integer, problem_id, write_jsonl  # noqa: E402

SOURCES = {
    "aime": ("AI-MO/aimo-validation-aime", "13f9e12f613e720c2a2b2f345dd04b998a29494d", "data/train-00000-of-00001.parquet"),
    "amc": ("AI-MO/aimo-validation-amc", "69d78a4a2c840e82d69af6bc742bda09005f6316", "data/train-00000-of-00001.parquet"),
    "omni2": ("martheballon/Omni-MATH-2", "693b21ad38ec8c8e0354d3a2f286ca83473821b0", "Omni-Math-2.jsonl"),
}
TRAIN_MAIN = ("aimo-interp/train-main-v2", "fa8af758c6519fc97a9d0bd9017feb6f12232b52", "data/train-00000-of-00001.parquet")
OMNI_IDS = Path(__file__).resolve().parent / "omni2_ids.json"
EXCLUDED = Path(__file__).resolve().parent / "excluded_problems.json"


def load_file(dataset: str, revision: str, filename: str) -> list[dict]:
    import pandas
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(dataset, filename, repo_type="dataset", revision=revision)
    if filename.endswith(".parquet"):
        return json.loads(pandas.read_parquet(path).to_json(orient="records"))
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--omni-ids", type=Path, default=OMNI_IDS, help="Pinned Omni-MATH-2 ids; ignored if missing.")
    parser.add_argument("--n-problems", type=int, default=300, help="Total problems when sampling Omni-MATH-2.")
    parser.add_argument("--min-difficulty", type=float, default=4.0, help="Omni-MATH-2 difficulty floor when sampling.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excluded = {normalize(row["problem"]) for row in load_file(*TRAIN_MAIN)}

    candidates: dict[str, list[dict]] = {}
    for source, (dataset, revision, filename) in SOURCES.items():
        kept = []
        for row in load_file(dataset, revision, filename):
            answer = parse_integer(str(row.get("answer", "")))
            text = str(row.get("problem", "")).strip()
            if answer is None or not text or normalize(text) in excluded or row.get("tags"):
                continue
            kept.append({
                "problem_id": problem_id(text),
                "source": source,
                "dataset": dataset,
                "revision": revision,
                "source_id": row.get("id"),
                "difficulty": row.get("difficulty"),
                "problem": text,
                "answer": answer,
            })
        candidates[source] = kept
        print(f"{source}: {len(kept)} usable integer-answer problems")

    chosen = candidates["aime"] + candidates["amc"]
    omni = candidates["omni2"]
    if args.omni_ids.is_file():
        by_id = {row["source_id"]: row for row in omni}
        pinned = json.loads(args.omni_ids.read_text())["ids"]
        missing = [i for i in pinned if i not in by_id]
        if missing:
            raise SystemExit(f"pinned Omni-MATH-2 ids not usable at this revision: {missing}")
        chosen += [by_id[i] for i in pinned]
        print(f"omni2: using {len(pinned)} pinned ids from {args.omni_ids}")
    else:
        pool = sorted(
            (row for row in omni if float(row["difficulty"] or 0) >= args.min_difficulty),
            key=lambda row: row["source_id"],
        )
        random.Random(args.seed).shuffle(pool)
        chosen += pool[: max(args.n_problems - len(chosen), 0)]
    dropped = {row["problem_id"] for row in json.loads(EXCLUDED.read_text())["problems"]}
    unique = [row for row in {row["problem_id"]: row for row in chosen}.values() if row["problem_id"] not in dropped]
    print(f"excluded {len(dropped)} problems listed in {EXCLUDED.name}")

    path = args.work_dir / "problems.jsonl"
    write_jsonl(path, unique)
    print(f"wrote {len(unique)} problems to {path}")


if __name__ == "__main__":
    main()
