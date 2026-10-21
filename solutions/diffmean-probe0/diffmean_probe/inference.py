"""Evaluation-time prediction: max_drop prior, plus direction score when available."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifact import ARTIFACT_DIR, load_config, load_directions
from .common import normalize_effort
from .scoring import decide, predict_drop, prior_mean


def _scores(model_id: str, effort: str, problems: list[str], layers: list[int], direction, site: str) -> np.ndarray:
    # Imported lazily so prior-only runs never import transformers.
    from .features import encode_problems, load_model, release

    tokenizer, model = load_model(model_id)
    try:
        features = encode_problems(tokenizer, model, problems, model_id, effort, layers)
    finally:
        release(model)
    return direction.score(features[site])


def predict_robustness(
    model_id: str,
    reasoning_effort: str,
    problems: list[str],
    artifact_dir: Path = ARTIFACT_DIR,
) -> list[bool]:
    """Return one native ``bool`` per problem, in input order."""
    if not problems:
        return []
    config = load_config(artifact_dir)
    if config is None:
        return [False] * len(problems)

    effort = normalize_effort(model_id, reasoning_effort)
    prior = prior_mean(config["priors"], model_id, effort)
    beta, threshold = float(config["beta"]), float(config["threshold"])
    entry = load_directions(model_id, artifact_dir) if beta != 0.0 else None
    if entry is None:
        return [bool(value) for value in decide(np.full(len(problems), prior), threshold)]

    try:
        scores = _scores(model_id, effort, problems, entry.layers, entry.direction, config["site"])
    except Exception:
        # A failed forward pass (e.g. OOM on a shared worker) degrades to the
        # prior instead of invalidating the whole batch.
        scores = np.zeros(len(problems))
    drop = predict_drop(prior, scores, beta, entry.direction.target_scale)
    return [bool(value) for value in decide(drop, threshold)]
