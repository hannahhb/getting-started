#!/usr/bin/env python3
"""Train the shared probe, select it by grouped CV, and export it.

**Shared probe.** Robustness is mostly a property of the problem: a problem
that breaks one model usually breaks the others. So each model's direction is
fitted on *every* model's labelled problems, read through that model's own
hidden states. The target is the residual ``max_drop - prior(row)``, so the
direction learns which problems are fragile beyond each model's base rate. At
evaluation time, a model is still scored only from its own activations.

Feature files from ``extract_features.py`` hold every row read by every model.
Older files that hold only a model's own rows still load; each model then
trains on its own rows only (the per-model probe).

**Selection.** Rows are grouped by problem text across models, so no problem's
labels leak between the training and held-out sides. Site, depth, beta, and
threshold are picked for plain accuracy on labelled rows; all rows, including
the unlabelled band (0.10 < max_drop < 0.25), train directions and priors.

**Behaviour score.** When feature files include short-resample features
(``behaviour``), each model also gets a direction over those standardized
features, fitted the same way, and its score enters with weight ``gamma``::

    drop_hat = prior + beta * s_h * z_hidden + gamma * s_b * z_behaviour

**Export.** The selected configuration is always exported, and every model's
direction is saved, even when the selected beta is 0. ``--nested`` also
reports a repeated nested-CV estimate of the whole selection procedure, since
the best inner-CV score is optimistically biased.

``--priors-only`` skips features and exports a prior-only artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

SOLUTION_ROOT = Path(__file__).resolve().parents[1]
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

from diffmean_probe.artifact import (  # noqa: E402
    ARTIFACT_DIR,
    SCHEMA_VERSION,
    Behaviour,
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

# A small grid: with ~80 labelled rows, a wide search overfits the inner CV
# (648 configs gave inner 0.77 but nested 0.73 in a gpt-oss proxy run).
DEPTHS = (0.3, 0.5, 0.7)
BETAS = (0.0, 0.3, 0.5)
# Labels are robust at max_drop <= 0.10 and not robust at >= 0.25.
THRESHOLDS = (0.175, 0.25)
# Weight on the short-resample behaviour score (used when features have it).
GAMMAS = (0.0, 0.3, 0.5)
Config = tuple[str, float, float, float, float]  # site, depth, beta, gamma, threshold


@dataclass
class Dataset:
    model_ids: list[str]  # model whose robustness each row labels
    efforts: list[str]
    problems: list[str]
    targets: np.ndarray  # max_drop, float
    labels: np.ndarray  # 1 robust, 0 not robust, -1 unlabelled band
    # Per encoder model: {site: (n_rows, n_hidden_states, d)} and the rows it has read.
    features: dict[str, dict[str, np.ndarray]]
    available: dict[str, np.ndarray]
    # Per encoder model that has them: short-resample features (n_rows, n_features).
    behaviour: dict[str, np.ndarray] = field(default_factory=dict)


def _row_keys(data) -> list[tuple[str, str, str]]:
    if "row_model_id" in data.files:
        owners = [str(value) for value in data["row_model_id"]]
    else:
        owners = [str(data["model_id"])] * len(data["max_drop"])
    return [(m, str(p), str(e)) for m, p, e in zip(owners, data["problem"], data["reasoning_effort"])]


def load_feature_dataset(features_dir: Path) -> Dataset:
    """Load shared feature files, or older own-rows-only files."""
    paths = sorted(features_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no feature files found in {features_dir}")
    files = [np.load(path, allow_pickle=False) for path in paths]
    try:
        shared = all("row_model_id" in data.files for data in files)
        sources = files[:1] if shared else files
        table = [
            (key, float(target), int(label))
            for data in sources
            for key, target, label in zip(_row_keys(data), data["max_drop"], data["label"])
        ]
        index = {key: i for i, (key, _, _) in enumerate(table)}
        if len(index) != len(table):
            raise SystemExit("feature files contain duplicate (model, problem, effort) rows")

        features, available, behaviour = {}, {}, {}
        for data in files:
            encoder = str(data["model_id"])
            rows = [index[key] for key in _row_keys(data)]
            if shared and rows != list(range(len(table))):
                raise SystemExit(f"{encoder}: rows are not in the shared order; re-run extraction")
            available[encoder] = np.zeros(len(table), dtype=bool)
            available[encoder][rows] = True
            features[encoder] = {}
            for site in SITES:
                values = data[site].astype(np.float32)
                full = np.zeros((len(table),) + values.shape[1:], dtype=np.float32)
                full[rows] = values
                features[encoder][site] = full
            if "behaviour" in data.files:
                values = data["behaviour"].astype(np.float64)
                full = np.zeros((len(table), values.shape[1]))
                full[rows] = values
                behaviour[encoder] = full
    finally:
        for data in files:
            data.close()

    return Dataset(
        model_ids=[key[0] for key, _, _ in table],
        problems=[key[1] for key, _, _ in table],
        efforts=[key[2] for key, _, _ in table],
        targets=np.asarray([target for _, target, _ in table]),
        labels=np.asarray([label for _, _, label in table], dtype=np.int8),
        features=features,
        available=available,
        behaviour=behaviour,
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
        features={},
        available={},
    )


def configs(data: Dataset) -> list[Config]:
    has_features = bool(data.features)
    sites = SITES if has_features else ("last",)
    depths = DEPTHS if has_features else (0.5,)
    betas = BETAS if has_features else (0.0,)
    gammas = GAMMAS if data.behaviour else (0.0,)
    return [
        (s, d, b, g, t) for s in sites for d in depths for b in betas for g in gammas for t in THRESHOLDS
    ]


def row_priors(data: Dataset, priors: dict, rows: np.ndarray) -> np.ndarray:
    return np.asarray([prior_mean(priors, data.model_ids[i], data.efforts[i]) for i in rows])


def fit_encoder(data: Dataset, encoder: str, site: str, depth: float, train: np.ndarray, priors: dict):
    """Fit ``encoder``'s direction on the residuals of every training row it has read."""
    rows = train[data.available[encoder][train]]
    site_features = data.features[encoder][site]
    layers = layer_window(site_features.shape[1], depth)
    residuals = data.targets[rows] - row_priors(data, priors, rows)
    return layers, fit_direction(site_features[rows][:, layers], residuals)


def fit_behaviour(data: Dataset, encoder: str, train: np.ndarray, priors: dict) -> Behaviour | None:
    """Standardize ``encoder``'s behaviour features on its training rows, then fit a direction."""
    rows = train[data.available[encoder][train]]
    values = data.behaviour[encoder][rows]
    mean, std = values.mean(axis=0), values.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    residuals = data.targets[rows] - row_priors(data, priors, rows)
    direction = fit_direction(((values - mean) / std)[:, None, :], residuals)
    return None if direction is None else Behaviour(mean=mean, std=std, direction=direction)


def fold_scores(data: Dataset, train: np.ndarray, test: np.ndarray, priors: dict) -> dict:
    """Held-out ``(z, target_scale)`` per (site, depth); each row uses its own model's encoder."""
    owners = np.asarray([data.model_ids[i] for i in test])
    scores = {}
    for site in SITES if data.features else ():
        for depth in DEPTHS:
            z, target_scale = np.zeros(len(test)), np.zeros(len(test))
            for encoder in data.features:
                local = np.flatnonzero((owners == encoder) & data.available[encoder][test])
                if not len(local):
                    continue
                layers, direction = fit_encoder(data, encoder, site, depth, train, priors)
                if direction is None:
                    continue
                z[local] = direction.score(data.features[encoder][site][test[local]][:, layers])
                target_scale[local] = direction.target_scale
            scores[(site, depth)] = (z, target_scale)
    if data.behaviour:
        z, target_scale = np.zeros(len(test)), np.zeros(len(test))
        for encoder in data.behaviour:
            local = np.flatnonzero((owners == encoder) & data.available[encoder][test])
            if not len(local):
                continue
            behaviour = fit_behaviour(data, encoder, train, priors)
            if behaviour is None:
                continue
            z[local] = behaviour.score(data.behaviour[encoder][test[local]])
            target_scale[local] = behaviour.direction.target_scale
        scores["behaviour"] = (z, target_scale)
    return scores


def fold_predictions(data: Dataset, train: np.ndarray, test: np.ndarray, grid: list[Config]) -> dict:
    priors = fit_priors(
        [data.model_ids[i] for i in train], [data.efforts[i] for i in train], data.targets[train]
    )
    prior = row_priors(data, priors, test)
    scores = fold_scores(data, train, test, priors)
    zeros = (np.zeros(len(test)), np.zeros(len(test)))
    z_behaviour, scale_behaviour = scores.get("behaviour", zeros)
    predictions = {}
    for site, depth, beta, gamma, threshold in grid:
        z, target_scale = scores.get((site, depth), zeros)
        drop = predict_drop(prior, z, beta, target_scale)
        drop = predict_drop(drop, z_behaviour, gamma, scale_behaviour)
        predictions[(site, depth, beta, gamma, threshold)] = decide(drop, threshold)
    return predictions


def splitter_for(labels: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> StratifiedGroupKFold:
    labelled = labels[labels >= 0]
    n_splits = min(n_splits, len(set(groups)), int((labelled == 1).sum()), int((labelled == 0).sum()))
    if n_splits < 2:
        raise SystemExit("not enough labelled problems of each class for grouped CV")
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def evaluate_grid(data: Dataset, subset: np.ndarray, repeats: int, n_splits: int, seed: int) -> dict:
    """Mean grouped-CV accuracy (labelled rows only) per config on ``subset``."""
    grid = configs(data)
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
    """Best accuracy; ties prefer smaller beta and gamma, mid-depth, then the label-band midpoint."""
    return max(
        grid,
        key=lambda c: (round(grid[c], 6), -c[2] - c[3], -abs(c[1] - 0.5), -abs(c[4] - 0.175)),
    )


def nested_predictions(data: Dataset, repeats: int, n_splits: int, seed: int) -> np.ndarray:
    """Outer grouped-CV predictions from running :func:`select` on each outer training set."""
    everything = np.arange(len(data.labels))
    groups = np.asarray(data.problems)
    predictions = np.zeros(len(everything), dtype=bool)
    splitter = splitter_for(data.labels, groups, n_splits, seed)
    for train, test in splitter.split(everything, data.labels, groups):
        config = select(evaluate_grid(data, train, repeats, n_splits, seed)["grid"])
        predictions[test] = fold_predictions(data, train, test, [config])[config]
    return predictions


def sign_test_p(wins: int, losses: int) -> float:
    """One-sided exact binomial p-value for at least ``wins`` of ``wins + losses`` by chance."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n


def nested_report(data: Dataset, outer_repeats: int, inner_repeats: int, n_splits: int, seed: int) -> dict:
    """Repeated nested CV, compared row by row with always-True.

    Wins and losses are pooled over outer repeats, so the sign test treats
    repeats as independent and overstates significance; read it as a sanity
    check, not a guarantee.
    """
    scored = data.labels >= 0
    truth = data.labels[scored] == 1
    accuracies, wins, losses = [], 0, 0
    for repeat in range(outer_repeats):
        predicted = nested_predictions(data, inner_repeats, n_splits, seed + 1000 + repeat)[scored]
        accuracies.append(accuracy(predicted, truth))
        correct = predicted == truth
        wins += int((correct & ~truth).sum())  # always-True is wrong exactly where truth is False
        losses += int((~correct & truth).sum())
    return {
        "nested_accuracy": float(np.mean(accuracies)),
        "nested_accuracy_runs": [float(value) for value in accuracies],
        "nested_vs_always_true_wins": wins,
        "nested_vs_always_true_losses": losses,
        "nested_vs_always_true_p": sign_test_p(wins, losses),
    }


def export(data: Dataset, config: Config, summary: dict, output_dir: Path) -> list[str]:
    site, depth, beta, gamma, threshold = config
    everything = np.arange(len(data.labels))
    priors = fit_priors(data.model_ids, data.efforts, data.targets)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.npz"):
        stale.unlink()
    exported = []
    if data.features:
        for encoder in data.features:
            layers, direction = fit_encoder(data, encoder, site, depth, everything, priors)
            behaviour = fit_behaviour(data, encoder, everything, priors) if encoder in data.behaviour else None
            if direction is not None:
                entry = ModelDirections(layers=layers, direction=direction, behaviour=behaviour)
                save_directions(encoder, entry, output_dir)
                exported.append(encoder)
    save_config(
        {
            "schema_version": SCHEMA_VERSION,
            "target": "max_drop",
            "site": site,
            "depth_fraction": depth,
            "beta": beta,
            "gamma": gamma,
            "threshold": threshold,
            "priors": priors,
            "models_with_directions": exported,
            "cv": summary,
        },
        output_dir,
    )
    return exported


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--features-dir", type=Path)
    source.add_argument("--priors-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--repeats", type=int, default=10, help="Inner CV repeats.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nested", action="store_true", help="Also report a nested-CV estimate.")
    parser.add_argument("--outer-repeats", type=int, default=3, help="Nested CV repeats.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    data = load_label_dataset() if args.priors_only else load_feature_dataset(args.features_dir)

    everything = np.arange(len(data.labels))
    result = evaluate_grid(data, everything, args.repeats, args.n_splits, args.seed)
    config = select(result["grid"])
    prior_grid = {c: v for c, v in result["grid"].items() if c[2] == 0.0 and c[3] == 0.0}
    labelled = data.labels >= 0
    summary = {
        "n_rows": int(len(data.labels)),
        "n_labelled_rows": int(labelled.sum()),
        "n_problems": len(set(data.problems)),
        "robust_rate": float((data.labels[labelled] == 1).mean()),
        "shared_features": bool(data.available) and all(mask.all() for mask in data.available.values()),
        "behaviour_models": sorted(data.behaviour),
        "repeats": args.repeats,
        "selected": list(config),
        "selected_accuracy": result["grid"][config],
        "selected_accuracy_std": result["grid_std"][config],
        "prior_only_accuracy": max(prior_grid.values()),
        **result["reference"],
    }

    if args.nested:
        summary.update(nested_report(
            data, args.outer_repeats, max(args.repeats // 2, 1), args.n_splits, args.seed
        ))

    exported = export(data, config, summary, args.output_dir)
    site, depth, beta, gamma, threshold = config
    print(json.dumps(
        {"site": site, "depth_fraction": depth, "beta": beta, "gamma": gamma, "threshold": threshold,
         "models_with_directions": exported, **summary},
        indent=2,
    ))
    return summary


if __name__ == "__main__":
    main()
