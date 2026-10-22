"""Codabench entry point for the forced-answer perturbation probe.

For each problem: sample reasoning on X and on 4 code-made perturbations,
force an answer after a fixed thinking budget, and compare forced answers and
answer log-probabilities between X and the perturbations (perturb_probe/).
A logistic model exported by ``research/cross_validate.py --export`` maps the
features to robust / not robust.

Falls back to predicting robust when there is no artifact, the artifact needs
features this runtime does not compute, the model cannot be loaded, or the
evaluation time budget is nearly used up.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from perturb_probe.probe import load_model, probe_problem, release, resolve

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "model.json"
TOTAL_SECONDS = 3600
RESERVE_SECONDS = 300  # leave room for remaining batches and scoring
_START = time.time()


def _load_artifact() -> dict | None:
    if not ARTIFACT.is_file():
        return None
    artifact = json.loads(ARTIFACT.read_text())
    if any(name.startswith("u_") for name in artifact["features"]):
        return None  # uncertainty features are not computed at test time yet
    return artifact


def _scores(artifact: dict, model_id: str, effort: str, rows: list[dict]) -> np.ndarray:
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
    z = (filled - mu) / sd
    return z @ np.array(artifact["coef"]) + artifact["intercept"]


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    artifact = _load_artifact()
    if artifact is None or not problems:
        return [True for _ in problems]
    checkpoint, effort = resolve(model_id, reasoning_effort)
    try:
        tokenizer, model = load_model(checkpoint)
    except Exception:
        return [True for _ in problems]
    rows, predictions = [], []
    try:
        for text in problems:
            if time.time() - _START > TOTAL_SECONDS - RESERVE_SECONDS:
                rows.append(None)
                continue
            try:
                rows.append(probe_problem(model, tokenizer, checkpoint, effort, text,
                                          n=artifact["n"], budget=artifact["budget"]))
            except Exception:
                rows.append(None)
    finally:
        release(model)
    usable = [r for r in rows if r is not None]
    scores = iter(_scores(artifact, checkpoint, effort, usable)) if usable else iter(())
    for r in rows:
        predictions.append(True if r is None else bool(next(scores) >= 0.0))
    return predictions
