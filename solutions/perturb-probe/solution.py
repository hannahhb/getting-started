"""Codabench entry point for the forced-answer perturbation probe.

For each problem: sample reasoning on X and on 4 code-made perturbations
(typos, variable rename, irrelevant sentence, reworded instruction), force an
answer after a fixed thinking budget, and compare the forced answers
(perturb_probe/). No training is needed: the default decision is a fixed rule,

    robust  <=>  X yields a forced answer r, and every perturbation keeps r
                 in at least AGREE_MIN of its samples.

If X yields no forced answer at all, the problem is predicted not robust (the
model could not commit within the budget). If ``artifacts/model.json`` exists
(``research/cross_validate.py --export``), its fitted classifier is used
instead. Falls back to robust if the model cannot be loaded or the time budget
is nearly used up.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from perturb_probe.probe import load_model, probe_problem, release, resolve

SAMPLES = 4          # traces per prompt (X and each perturbation)
BUDGET = 4096        # thinking tokens before the answer is forced
AGREE_MIN = 0.5      # each perturbation must keep the reference answer this often
ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "model.json"
TOTAL_SECONDS = 3600
RESERVE_SECONDS = 300  # leave room for remaining batches and scoring
_START = time.time()


def _load_artifact() -> dict | None:
    if not ARTIFACT.is_file():
        return None
    artifact = json.loads(ARTIFACT.read_text())
    if any(name.startswith("u_") for name in artifact["features"]):
        return None  # uncertainty features are not computed at test time
    return artifact


def _rule(features: dict) -> bool:
    if features.get("reference") is None:
        return False
    return bool(features["agree_min"] >= AGREE_MIN)


def _fitted(artifact: dict, model_id: str, effort: str, rows: list[dict]) -> list[bool]:
    names = artifact["features"]
    raw = np.array([[float(r.get(n, math.nan)) for n in names] for r in rows])
    stats = artifact["stats"].get(f"{model_id}||{effort}")
    if stats is None:  # unseen model/effort: normalise over this batch (no labels needed)
        med = np.nan_to_num(np.nanmedian(raw, axis=0))
        filled = np.where(np.isnan(raw), med, raw)
        mu, sd = filled.mean(axis=0), filled.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
    else:
        med, mu, sd = (np.array(stats[k]) for k in ("median", "mean", "sd"))
        filled = np.where(np.isnan(raw), med, raw)
    scores = ((filled - mu) / sd) @ np.array(artifact["coef"]) + artifact["intercept"]
    return [bool(s >= 0.0) for s in scores]


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    if not problems:
        return []
    artifact = _load_artifact()
    samples = artifact["n"] if artifact else SAMPLES
    budget = artifact["budget"] if artifact else BUDGET
    checkpoint, effort = resolve(model_id, reasoning_effort)
    try:
        tokenizer, model = load_model(checkpoint)
    except Exception:
        return [True for _ in problems]

    rows: list[dict | None] = []
    try:
        for text in problems:
            if time.time() - _START > TOTAL_SECONDS - RESERVE_SECONDS:
                rows.append(None)
                continue
            try:
                features = probe_problem(model, tokenizer, checkpoint, effort, text, n=samples, budget=budget)
                features.pop("x_raw", None)
                rows.append(features)
            except Exception:
                rows.append(None)
    finally:
        del model
        release()

    usable = [r for r in rows if r is not None]
    decided = iter(_fitted(artifact, checkpoint, effort, usable) if artifact else [_rule(r) for r in usable])
    return [True if r is None else next(decided) for r in rows]
