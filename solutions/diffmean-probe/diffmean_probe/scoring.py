"""Covariance directions, hierarchical max_drop priors, and the decision rule.

The target is ``max_drop``, the largest accuracy drop across perturbations
(robust when <= 0.10, not robust when >= 0.25, unlabelled in between). Per
model and layer, the direction is the covariance between hidden state and
target, ``sum_i (h_i - mean_h) * (y_i - mean_y)``. This is the regression
analogue of diff-in-means (one-component PLS); it can use the unlabelled-band
rows that a binary diff-in-means must discard. Projections are standardized
on the training rows, so ``z`` has zero mean and unit variance per layer;
the layer scores are averaged and the average is rescaled to unit variance,
so ``beta`` keeps its meaning as a correlation.

The predicted drop shrinks toward a hierarchical prior::

    drop_hat = prior_mean(model, effort) + beta * target_std(model) * mean_layer_z
    robust = drop_hat < threshold

``beta`` acts as the out-of-sample correlation. In-sample it would be close to
1, since d >> n, so cross-validation picks it. ``beta = 0`` is the prior alone.
The threshold is also picked by CV for plain accuracy, the leaderboard metric.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

# Pseudo-counts pulling each prior mean toward its parent level.
PRIOR_STRENGTH = 2.0
# Minimum rows (with non-constant target) needed to fit a direction.
MIN_ROWS = 4


@dataclass(frozen=True)
class Direction:
    """Unit directions (k, d) with per-layer projection mean/std, plus target std.

    ``combined_scale`` is the training std of the layer-averaged score.
    """

    unit: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    target_scale: float
    combined_scale: float = 1.0

    def layer_mean(self, features: np.ndarray) -> np.ndarray:
        projections = np.einsum("nkd,kd->nk", features.astype(np.float64), self.unit)
        return ((projections - self.center) / self.scale).mean(axis=1)

    def score(self, features: np.ndarray) -> np.ndarray:
        """Unit-variance layer-averaged projection; ``features`` is (n, k, d) -> (n,)."""
        return self.layer_mean(features) / self.combined_scale


def fit_direction(features: np.ndarray, targets: np.ndarray) -> Direction | None:
    """Fit covariance directions on (n, k, d) features; ``None`` if underdetermined."""
    targets = np.asarray(targets, dtype=np.float64)
    target_scale = float(targets.std())
    if len(targets) < MIN_ROWS or target_scale == 0.0:
        return None
    features = features.astype(np.float64)
    centered = features - features.mean(axis=0)
    covariance = np.einsum("nkd,n->kd", centered, targets - targets.mean())
    norms = np.linalg.norm(covariance, axis=1, keepdims=True)
    if np.any(norms == 0):
        return None
    unit = covariance / norms
    projections = np.einsum("nkd,kd->nk", features, unit)
    scale = projections.std(axis=0)
    direction = Direction(
        unit=unit,
        center=projections.mean(axis=0),
        scale=np.where(scale > 0, scale, 1.0),
        target_scale=target_scale,
    )
    combined = float(direction.layer_mean(features).std())
    return replace(direction, combined_scale=combined if combined > 0 else 1.0)


def _shrunk(values: Sequence[float], parent: float) -> float:
    return (float(np.sum(values)) + PRIOR_STRENGTH * parent) / (len(values) + PRIOR_STRENGTH)


def fit_priors(
    model_ids: Sequence[str], efforts: Sequence[str], targets: Sequence[float]
) -> dict[str, object]:
    """Fit global -> model -> (model, effort) shrunken means of ``max_drop``."""
    targets = [float(value) for value in targets]
    global_mean = float(np.mean(targets))
    models: dict[str, dict[str, object]] = {}
    for model_id in sorted(set(model_ids)):
        rows = [i for i, m in enumerate(model_ids) if m == model_id]
        model_mean = _shrunk([targets[i] for i in rows], global_mean)
        by_effort = {}
        for effort in sorted({efforts[i] for i in rows}):
            cell = [targets[i] for i in rows if efforts[i] == effort]
            by_effort[effort] = _shrunk(cell, model_mean)
        models[model_id] = {"mean": model_mean, "efforts": by_effort}
    return {"mean": global_mean, "models": models}


def prior_mean(priors: Mapping[str, object], model_id: str, effort: str) -> float:
    """Look up the most specific prior available for ``(model_id, effort)``."""
    model = priors.get("models", {}).get(model_id)  # type: ignore[union-attr]
    if model is None:
        return float(priors["mean"])  # type: ignore[arg-type]
    return float(model["efforts"].get(effort, model["mean"]))


def predict_drop(
    prior: float | np.ndarray, score: float | np.ndarray, beta: float, target_scale: float
) -> np.ndarray:
    return np.asarray(prior + beta * target_scale * np.asarray(score), dtype=np.float64)


def decide(drop: float | np.ndarray, threshold: float) -> np.ndarray:
    """Robust when the predicted drop is below ``threshold``."""
    return np.asarray(drop, dtype=np.float64) < float(threshold)


def layer_window(n_hidden_states: int, depth_fraction: float, width: int = 3) -> list[int]:
    """Indices of ``width`` adjacent transformer layers centered at a relative depth.

    ``n_hidden_states`` counts the embedding output, which is never selected.
    """
    n_layers = n_hidden_states - 1
    center = int(round(depth_fraction * n_layers))
    half = width // 2
    low = min(max(center - half, 1), max(n_layers - width + 1, 1))
    return list(range(low, min(low + width, n_layers + 1)))


def accuracy(predictions: Iterable[bool], labels: Iterable[bool]) -> float:
    pairs = list(zip(predictions, labels, strict=True))
    return sum(bool(p) == bool(y) for p, y in pairs) / len(pairs) if pairs else float("nan")
