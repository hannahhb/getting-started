#!/usr/bin/env python3
"""Cache every-layer hidden states of every training problem, read by every model.

Run on a GPU machine. Each model encodes *all* training rows, not just its
own, so the shared probe can learn each model's direction from every model's
labels. One ``<safe-model-id>.npz`` per model is written to ``--output-dir``
with float32 ``last`` and ``mean`` arrays of shape
``(n_rows, n_hidden_states, hidden)`` plus row metadata (``row_model_id``,
the ``max_drop`` target, and ``label``). Rows are in the same order in every
file. Models already present are skipped, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SOLUTION_ROOT = Path(__file__).resolve().parents[1]
ROOT = SOLUTION_ROOT.parents[1]
for import_path in (ROOT / "scripts", SOLUTION_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from diffmean_probe.common import normalize_effort, safe_model_id  # noqa: E402

DEFAULT_DATASET = "aimo-interp/train-main-v2"
DEFAULT_REVISION = "fa8af758c6519fc97a9d0bd9017feb6f12232b52"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "diffmean-probe" / "features-shared"


def load_rows(dataset: str, revision: str, split: str) -> list[dict]:
    """Fetch rows that have a ``max_drop`` target.

    ``label`` is 1 (robust), 0 (not robust), or -1 for rows in the band the
    organizers leave unlabelled (0.10 < max_drop < 0.25). Those still train
    the regression; only labelled rows are scored.
    """
    from import_hf_dataset import fetch_all_rows

    rows = []
    for row in fetch_all_rows(dataset, revision, "default", split):
        max_drop = row.get("max_drop")
        if max_drop is None:
            continue
        label = row.get("is_robust")
        if isinstance(label, str):
            label = {"true": True, "false": False}.get(label.lower())
        model_id = str(row["model_id"])
        rows.append(
            {
                "model_id": model_id,
                "reasoning_effort": normalize_effort(model_id, row.get("reasoning_effort")),
                "problem": str(row["problem"]),
                "max_drop": float(max_drop),
                "label": int(label) if type(label) is bool else -1,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="*", help="Only extract these model ids.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def encode_effort(encoder_id: str, row: dict) -> str:
    """Effort used when ``encoder_id`` reads ``row``'s problem.

    A model's own rows keep their effort; other models' rows use the
    encoder's default, which is what it sees at evaluation time.
    """
    if row["model_id"] == encoder_id:
        return row["reasoning_effort"]
    return normalize_effort(encoder_id, "default")


def main() -> None:
    args = parse_args()
    from diffmean_probe.features import encode_problem, load_model, release

    rows = load_rows(args.dataset, args.revision, args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source.json").write_text(
        json.dumps({"dataset": args.dataset, "revision": args.revision, "split": args.split}, indent=2)
    )
    model_ids = sorted({row["model_id"] for row in rows})
    if args.models:
        model_ids = [model_id for model_id in model_ids if model_id in set(args.models)]

    for model_id in model_ids:
        path = args.output_dir / f"{safe_model_id(model_id)}.npz"
        if path.is_file() and not args.overwrite:
            print(f"skip {model_id}: {path} exists")
            continue
        efforts = [encode_effort(model_id, row) for row in rows]
        keys = sorted({(row["problem"], effort) for row, effort in zip(rows, efforts)})
        print(f"{model_id}: encoding {len(keys)} prompts for {len(rows)} rows")
        tokenizer, model = load_model(model_id)
        try:
            cache = {
                (problem, effort): encode_problem(tokenizer, model, problem, model_id, effort)
                for problem, effort in keys
            }
        finally:
            release(model)
        encoded = [cache[(row["problem"], effort)] for row, effort in zip(rows, efforts)]
        np.savez(
            path,
            model_id=np.asarray(model_id),
            last=np.stack([item["last"] for item in encoded]).astype(np.float32),
            mean=np.stack([item["mean"] for item in encoded]).astype(np.float32),
            row_model_id=np.asarray([row["model_id"] for row in rows]),
            problem=np.asarray([row["problem"] for row in rows]),
            reasoning_effort=np.asarray([row["reasoning_effort"] for row in rows]),
            encode_effort=np.asarray(efforts),
            max_drop=np.asarray([row["max_drop"] for row in rows], dtype=np.float64),
            label=np.asarray([row["label"] for row in rows], dtype=np.int8),
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
