"""Evaluation-time prediction: max_drop prior, plus hidden-state and behaviour scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifact import ARTIFACT_DIR, ModelDirections, load_config, load_directions
from .common import normalize_effort
from .scoring import decide, predict_drop, prior_mean


def _scores(
    model_id: str,
    effort: str,
    problems: list[str],
    entry: ModelDirections,
    site: str,
    hidden: bool,
    behaviour: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Hidden-state and behaviour scores; a score that is not requested is zeros."""
    # Imported lazily so prior-only runs never import transformers.
    from .features import behaviour_features, encode_problems, load_model, release

    zeros = np.zeros(len(problems))
    tokenizer, model = load_model(model_id)
    try:
        hidden_scores = zeros
        if hidden:
            features = encode_problems(tokenizer, model, problems, model_id, effort, entry.layers)
            hidden_scores = entry.direction.score(features[site])
        behaviour_scores = zeros
        if behaviour:
            values = np.stack(
                [behaviour_features(tokenizer, model, text, model_id, effort) for text in problems]
            )
            behaviour_scores = entry.behaviour.score(values)
    finally:
        del model
        release()
    return hidden_scores, behaviour_scores


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
    gamma = float(config.get("gamma", 0.0))
    entry = load_directions(model_id, artifact_dir) if beta != 0.0 or gamma != 0.0 else None
    if entry is None:
        return [bool(value) for value in decide(np.full(len(problems), prior), threshold)]

    use_behaviour = gamma != 0.0 and entry.behaviour is not None
    try:
        hidden_scores, behaviour_scores = _scores(
            model_id, effort, problems, entry, config["site"], beta != 0.0, use_behaviour
        )
    except Exception:
        # A failed forward pass (e.g. OOM on a shared worker) degrades to the
        # prior instead of invalidating the whole batch.
        hidden_scores = behaviour_scores = np.zeros(len(problems))
    drop = predict_drop(prior, hidden_scores, beta, entry.direction.target_scale)
    if use_behaviour:
        drop = predict_drop(drop, behaviour_scores, gamma, entry.behaviour.direction.target_scale)
    return [bool(value) for value in decide(drop, threshold)]
