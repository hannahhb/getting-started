#!/usr/bin/env python3
"""Select site, depth, beta, and threshold by grouped CV, then export the artifact.

The regression target is ``max_drop``. All rows train the directions and
priors, including the unlabelled band (0.10 < max_drop < 0.25), but only
labelled rows are scored.

Rows are grouped by problem text across models, so a problem never appears in
both the training and held-out side of a split. Selection maximises plain
accuracy, the leaderboard metric. ``--nested`` additionally reports an honest
outer-loop estimate of the whole selection procedure, which matters with only
~80 labelled rows: the best inner-CV score is optimistically biased.

``--priors-only`` skips features and exports a prior-only artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

SOLUTION_ROOT = Path(__file__).resolve().parents[1]
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

from diffmean_probe.artifact import (  # noqa: E402
    ARTIFACT_DIR,
    SCHEMA_VERSION,
    ModelDirections,
    save_config,
    save_directions,
)
from diffmean_probe.common import SITES  # noqa: E402
from diffmean_probe.scoring import (  # noqa: E402
    accuracy,
    decide,
    fit_direction,
    fit_priors,
    layer_window,
    predict_drop,
    prior_mean,
)

DEPTHS = tuple(round(0.1 * step, 1) for step in range(1, 10))
BETAS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7)
# Labels are robust at max_drop <= 0.10 and not robust at >= 0.25.
THRESHOLDS = (0.10, 0.15, 0.175, 0.20, 0.25, 0.35)
Config = tuple[str, float, float, float]  # site, depth, beta, threshold


@dataclass
class Dataset:
    model_ids: list[str]
    efforts: list[str]
    problems: list[str]
    targets: np.ndarray  # max_drop, float
    labels: np.ndarray  # 1 robust, 0 not robust, -1 unlabelled band
    # Per model: global row indices and {site: (n_model_rows, n_hidden_states, d)}.
    rows_by_model: dict[str, np.ndarray]
    features: dict[str, dict[str, np.ndarray]]


def load_feature_dataset(features_dir: Path) -> Dataset:
    model_ids, efforts, problems, targets, labels = [], [], [], [], []
    rows_by_model, features = {}, {}
    for path in sorted(features_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            model_id = str(data["model_id"])
            start = len(model_ids)
            count = len(data["max_drop"])
            model_ids += [model_id] * count
            efforts += [str(value) for value in data["reasoning_effort"]]
            problems += [str(value) for value in data["problem"]]
            targets += [float(value) for value in data["max_drop"]]
            labels += [int(value) for value in data["label"]]
            rows_by_model[model_id] = np.arange(start, start + count)
            features[model_id] = {site: data[site].astype(np.float32) for site in SITES}
    if not model_ids:
        raise SystemExit(f"no feature files found in {features_dir}")
    return Dataset(
        model_ids, efforts, problems, np.asarray(targets), np.asarray(labels, dtype=np.int8),
        rows_by_model, features,
    )


def load_label_dataset() -> Dataset:
    """Targets only, from the default dataset used by ``extract_features.py``."""
    sys.path[:0] = [str(SOLUTION_ROOT / "scripts"), str(SOLUTION_ROOT.parents[1] / "scripts")]
    import extract_features

    rows = extract_features.load_rows(
        extract_features.DEFAULT_DATASET, extract_features.DEFAULT_REVISION, extract_features.DEFAULT_SPLIT
    )
    return Dataset(
        model_ids=[row["model_id"] for row in rows],
        efforts=[row["reasoning_effort"] for row in rows],
        problems=[row["problem"] for row in rows],
        targets=np.asarray([row["max_drop"] for row in rows], dtype=np.float64),
        labels=np.asarray([row["label"] for row in rows], dtype=np.int8),
        rows_by_model={},
        features={},
    )


def configs(has_features: bool) -> list[Config]:
    if not has_features:
        return [("last", 0.5, 0.0, threshold) for threshold in THRESHOLDS]
    return [
        (site, depth, beta, threshold)
        for site in SITES
        for depth in DEPTHS
        for beta in BETAS
        for threshold in THRESHOLDS
    ]


def fold_scores(
    data: Dataset, train: np.ndarray, test: np.ndarray
) -> dict[tuple[str, float], tuple[np.ndarray, np.ndarray]]:
    """Held-out ``(z, target_scale)`` per (site, depth); zeros where no direction fits."""
    train_set = set(train.tolist())
    position = {row: i for i, row in enumerate(test)}
    scores = {}
    for site in SITES if data.features else ():
        for depth in DEPTHS:
            z, target_scale = np.zeros(len(test)), np.zeros(len(test))
            for model_id, rows in data.rows_by_model.items():
                local_train = [i for i, row in enumerate(rows) if row in train_set]
                local_test = [i for i, row in enumerate(rows) if row in position]
                if not local_test:
                    continue
                site_features = data.features[model_id][site]
                layers = layer_window(site_features.shape[1], depth)
                direction = fit_direction(
                    site_features[local_train][:, layers], data.targets[rows[local_train]]
                )
                if direction is None:
                    continue
                held_out = direction.score(site_features[local_test][:, layers])
                for i, value in zip(local_test, held_out):
                    z[position[rows[i]]] = value
                    target_scale[position[rows[i]]] = direction.target_scale
            scores[(site, depth)] = (z, target_scale)
    return scores


def fold_predictions(data: Dataset, train: np.ndarray, test: np.ndarray, grid: list[Config]) -> dict:
    priors = fit_priors(
        [data.model_ids[i] for i in train], [data.efforts[i] for i in train], data.targets[train]
    )
    prior = np.asarray([prior_mean(priors, data.model_ids[i], data.efforts[i]) for i in test])
    scores = fold_scores(data, train, test)
    zeros = (np.zeros(len(test)), np.zeros(len(test)))
    predictions = {}
    for site, depth, beta, threshold in grid:
        z, target_scale = scores.get((site, depth), zeros)
        drop = predict_drop(prior, z, beta, target_scale)
        predictions[(site, depth, beta, threshold)] = decide(drop, threshold)
    return predictions


def splitter_for(labels: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> StratifiedGroupKFold:
    labelled = labels[labels >= 0]
    n_splits = min(n_splits, len(set(groups)), int((labelled == 1).sum()), int((labelled == 0).sum()))
    if n_splits < 2:
        raise SystemExit("not enough labelled problems of each class for grouped CV")
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def evaluate_grid(data: Dataset, subset: np.ndarray, repeats: int, n_splits: int, seed: int) -> dict:
    """Mean grouped-CV accuracy (labelled rows only) per config on ``subset``."""
    grid = configs(bool(data.features))
    totals: dict[Config, list[float]] = {config: [] for config in grid}
    labels = data.labels[subset]
    scored = labels >= 0
    groups = np.asarray([data.problems[i] for i in subset])
    for repeat in range(repeats):
        splitter = splitter_for(labels, groups, n_splits, seed + repeat)
        predictions = {config: np.zeros(len(subset), dtype=bool) for config in grid}
        for train_local, test_local in splitter.split(np.zeros(len(subset)), labels, groups):
            fold = fold_predictions(data, subset[train_local], subset[test_local], grid)
            for config in grid:
                predictions[config][test_local] = fold[config]
        for config in grid:
            totals[config].append(accuracy(predictions[config][scored], labels[scored] == 1))
    robust_rate = float((labels[scored] == 1).mean())
    return {
        "grid": {config: float(np.mean(values)) for config, values in totals.items()},
        "grid_std": {config: float(np.std(values)) for config, values in totals.items()},
        "reference": {"always_true": robust_rate, "always_false": 1.0 - robust_rate},
    }


def select(grid: dict[Config, float]) -> Config:
    """Best accuracy; ties prefer smaller beta, mid-depth, then the label-band midpoint."""
    return max(
        grid,
        key=lambda c: (round(grid[c], 6), -c[2], -abs(c[1] - 0.5), -abs(c[3] - 0.175)),
    )


def nested_estimate(data: Dataset, repeats: int, n_splits: int, seed: int) -> float:
    """Outer grouped-CV accuracy of running :func:`select` on each outer training set."""
    everything = np.arange(len(data.labels))
    groups = np.asarray(data.problems)
    predictions = np.zeros(len(everything), dtype=bool)
    splitter = splitter_for(data.labels, groups, n_splits, seed + 1000)
    for train, test in splitter.split(everything, data.labels, groups):
        config = select(evaluate_grid(data, train, repeats, n_splits, seed)["grid"])
        predictions[test] = fold_predictions(data, train, test, [config])[config]
    scored = data.labels >= 0
    return accuracy(predictions[scored], data.labels[scored] == 1)


def export(data: Dataset, config: Config, summary: dict, output_dir: Path) -> None:
    site, depth, beta, threshold = config
    priors = fit_priors(data.model_ids, data.efforts, data.targets)
    for stale in output_dir.glob("*.npz"):
        stale.unlink()
    exported = []
    if beta != 0.0:
        for model_id, rows in data.rows_by_model.items():
            site_features = data.features[model_id][site]
            layers = layer_window(site_features.shape[1], depth)
            direction = fit_direction(site_features[:, layers], data.targets[rows])
            if direction is not None:
                save_directions(model_id, ModelDirections(layers=layers, direction=direction), output_dir)
                exported.append(model_id)
    save_config(
        {
            "schema_version": SCHEMA_VERSION,
            "target": "max_drop",
            "site": site,
            "depth_fraction": depth,
            "beta": beta,
            "threshold": threshold,
            "priors": priors,
            "models_with_directions": exported,
            "cv": summary,
        },
        output_dir,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--features-dir", type=Path)
    source.add_argument("--priors-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nested", action="store_true", help="Also report a nested-CV estimate.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    data = load_label_dataset() if args.priors_only else load_feature_dataset(args.features_dir)

    everything = np.arange(len(data.labels))
    result = evaluate_grid(data, everything, args.repeats, args.n_splits, args.seed)
    config = select(result["grid"])
    labelled = data.labels >= 0
    summary = {
        "n_rows": int(len(data.labels)),
        "n_labelled_rows": int(labelled.sum()),
        "n_problems": len(set(data.problems)),
        "robust_rate": float((data.labels[labelled] == 1).mean()),
        "repeats": args.repeats,
        "selected_accuracy": result["grid"][config],
        "selected_accuracy_std": result["grid_std"][config],
        "prior_only_accuracy": max(value for c, value in result["grid"].items() if c[2] == 0.0),
        **result["reference"],
    }
    if args.nested:
        summary["nested_accuracy"] = nested_estimate(data, max(args.repeats // 2, 1), args.n_splits, args.seed)
    export(data, config, summary, args.output_dir)
    site, depth, beta, threshold = config
    print(json.dumps(
        {"site": site, "depth_fraction": depth, "beta": beta, "threshold": threshold, **summary}, indent=2
    ))
    return summary


if __name__ == "__main__":
    main()
