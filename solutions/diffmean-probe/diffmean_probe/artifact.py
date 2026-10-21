"""Artifact layout: one JSON config plus an optional ``.npz`` per model.

``artifacts/config.json`` holds the selected site, beta, threshold, max_drop
priors, and the CV summary. ``artifacts/<safe-model-id>.npz`` holds that
model's covariance directions. A model without an ``.npz`` is scored by its prior alone, so a
prior-only artifact is a valid submission that never loads a language model.
Plain JSON and ``.npz`` avoid pickle and are independent of library versions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import SITES, safe_model_id
from .scoring import Direction

SCHEMA_VERSION = 2
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
CONFIG_NAME = "config.json"


@dataclass(frozen=True)
class ModelDirections:
    layers: list[int]
    direction: Direction


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported schema_version {config.get('schema_version')!r}")
    if config.get("site") not in SITES:
        raise RuntimeError(f"unsupported site {config.get('site')!r}")
    for key in ("beta", "threshold"):
        if not isinstance(config.get(key), (int, float)):
            raise RuntimeError(f"config is missing a numeric {key}")
    priors = config.get("priors")
    if not isinstance(priors, dict) or "mean" not in priors:
        raise RuntimeError("config is missing priors")
    return config


def save_config(config: dict[str, Any], directory: Path = ARTIFACT_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CONFIG_NAME
    path.write_text(json.dumps(validate_config(config), indent=2, sort_keys=True) + "\n")
    return path


def load_config(directory: Path = ARTIFACT_DIR) -> dict[str, Any] | None:
    path = directory / CONFIG_NAME
    if not path.is_file():
        return None
    return validate_config(json.loads(path.read_text(encoding="utf-8")))


def save_directions(model_id: str, entry: ModelDirections, directory: Path = ARTIFACT_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{safe_model_id(model_id)}.npz"
    np.savez_compressed(
        path,
        model_id=np.asarray(model_id),
        layers=np.asarray(entry.layers, dtype=np.int64),
        unit=entry.direction.unit.astype(np.float32),
        center=entry.direction.center.astype(np.float64),
        scale=entry.direction.scale.astype(np.float64),
        target_scale=np.asarray(entry.direction.target_scale, dtype=np.float64),
    )
    return path


def load_directions(model_id: str, directory: Path = ARTIFACT_DIR) -> ModelDirections | None:
    path = directory / f"{safe_model_id(model_id)}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["model_id"]) != model_id:
            raise RuntimeError(f"{path.name} was fitted for {data['model_id']!s}, not {model_id}")
        unit = data["unit"].astype(np.float64)
        layers = [int(layer) for layer in data["layers"]]
        if unit.ndim != 2 or unit.shape[0] != len(layers):
            raise RuntimeError(f"{path.name} has inconsistent direction shapes")
        return ModelDirections(
            layers=layers,
            direction=Direction(
                unit=unit,
                center=data["center"],
                scale=data["scale"],
                target_scale=float(data["target_scale"]),
            ),
        )
