#!/usr/bin/env python3
"""Re-grade saved samples from their traces after a change to answer parsing.

For every ``samples/<model>__<effort>.jsonl`` with a ``.traces.jsonl`` beside
it, recomputes ``answers`` and ``correct`` from the saved responses with the
current ``extract_answer`` (truncated responses still count as wrong). Prompts
without a saved trace are left as they are. The previous file is kept as
``.jsonl.bak``.

Run in the project environment: ``python datagen/regrade.py``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DEFAULT_WORK_DIR, extract_answer, read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    args = parser.parse_args()
    gold = {row["problem_id"]: row["answer"] for row in read_jsonl(args.work_dir / "problems.jsonl")}

    for path in sorted((args.work_dir / "samples").glob("*.jsonl")):
        traces_file = path.with_suffix(".traces.jsonl")
        if path.name.endswith(".traces.jsonl") or not traces_file.is_file():
            continue
        traces = {(t["problem_id"], t["type"], t["variant"]): t for t in read_jsonl(traces_file)}
        rows, changed = [], 0
        for row in read_jsonl(path):
            trace = traces.get((row["problem_id"], row["type"], row["variant"]))
            if trace is not None:
                finish = trace.get("finish_reasons") or ["stop"] * len(trace["texts"])
                answers = [extract_answer(text) if f == "stop" else None for text, f in zip(trace["texts"], finish)]
                correct = [answer == gold[row["problem_id"]] for answer in answers]
                changed += sum(a != b for a, b in zip(correct, row["correct"]))
                row = {**row, "answers": answers, "correct": correct}
            rows.append(row)
        shutil.copy(path, path.with_suffix(".jsonl.bak"))
        write_jsonl(path, rows)
        print(f"{path.name}: {changed} samples changed grade")


if __name__ == "__main__":
    main()
