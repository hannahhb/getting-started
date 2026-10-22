#!/usr/bin/env python3
"""5-fold cross-validation of the forced-answer perturbation probe on the train cases.

Joins ``data/perturb-probe/features/*.jsonl`` with the labelled cases and fits
a logistic regression on a fixed feature set:

* ``perturbation``: agreement and log-prob drop between X and its perturbations
* ``perturbation+length``: plus whether reasoning finished within budget, and its length
* ``perturbation+length+uncertainty``: plus the uncertainty-profiling features (``u_*``,
  present only if extraction ran with ``--with-uncertainty``)

Features are z-scored within each model/effort over its own problems (no
labels used), so one classifier serves all models. Folds are grouped by
problem (a problem never sits on both sides) and stratified by label; the
5-fold split is repeated ``--repeats`` times.

``--export`` fits the chosen set on all rows and writes
``solutions/perturb-probe/artifacts/model.json`` for ``solution.py``.

    python solutions/perturb-probe/research/cross_validate.py
    python solutions/perturb-probe/research/cross_validate.py --export perturbation+length
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))



PERTURBATION = ["x_consistency", "agree_mean", "agree_min", "dlogp", "dlogp_max"]
LENGTH = ["finish_x", "tokens_x"]
UNCERTAINTY_PREFIX = "u_"
ARTIFACT = HERE.parent / "artifacts" / "model.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def auc(scores, labels) -> float:
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def load(features_dir: Path, cases_path: Path, source: str | None):
    feats = {}
    for path in sorted(features_dir.glob("*.jsonl")):
        for row in read_jsonl(path):
            feats[(row["model_id"], row["reasoning_effort"], row["problem_id"])] = row
    rows = []
    for case in read_jsonl(cases_path):
        if case["is_robust"] is None or (source and case["source"] != source):
            continue
        f = feats.get((case["model_id"], case["reasoning_effort"], case["problem_id"]))
        if f is not None:
            rows.append({**f, "label": bool(case["is_robust"]), "source": case["source"]})
    return rows


def matrix(rows, names, stats=None):
    """Per-model/effort z-scored features (label-free); returns X and the stats used."""
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["model_id"], r["reasoning_effort"])].append(i)
    raw = np.array([[float(r.get(n, np.nan)) for n in names] for r in rows], dtype=float)
    x = np.zeros_like(raw)
    stats = dict(stats or {})
    for key, idx in groups.items():
        block = raw[idx]
        if key not in stats:
            med = np.nanmedian(block, axis=0)
            med = np.where(np.isnan(med), 0.0, med)
            filled = np.where(np.isnan(block), med, block)
            sd = filled.std(axis=0)
            stats[key] = (med, np.where(sd > 0, sd, 1.0), filled.mean(axis=0))
        med, sd, mu = stats[key]
        filled = np.where(np.isnan(block), med, block)
        x[idx] = (filled - mu) / sd
    return x, stats


def cross_validate(rows, names, folds, repeats, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold

    x, _ = matrix(rows, names)
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["problem_id"] for r in rows])
    accs, aucs, probs = [], [], np.zeros(len(y))
    for rep in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=min(folds, len(set(groups))), shuffle=True, random_state=seed + rep)
        p = np.zeros(len(y))
        for train, test in splitter.split(x, y, groups):
            if y[train].all() or not y[train].any():
                p[test] = y[train].mean()
                continue
            clf = LogisticRegression(C=0.5, max_iter=1000).fit(x[train], y[train])
            p[test] = clf.predict_proba(x[test])[:, 1]
        accs.append(np.mean((p >= 0.5) == y))
        aucs.append(auc(p, y))
        probs += p / repeats
    return np.array(accs), np.array(aucs), probs, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", type=Path, default=ROOT / "data" / "perturb-probe" / "features")
    parser.add_argument("--cases", type=Path, default=HERE / "train_set" / "cases.jsonl")
    parser.add_argument("--source", default="train-main-v2", help="Only score this source ('' for all).")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export", help="Feature set to fit on all rows and write to artifacts/model.json.")
    args = parser.parse_args()

    rows = load(args.features_dir, args.cases, args.source or None)
    if not rows:
        raise SystemExit("no rows: run research/extract_features.py first")
    u_names = sorted(k for k in rows[0] if k.startswith(UNCERTAINTY_PREFIX))
    sets = {"perturbation": PERTURBATION, "perturbation+length": PERTURBATION + LENGTH}
    if u_names:
        sets["perturbation+length+uncertainty"] = PERTURBATION + LENGTH + u_names
        sets["uncertainty only (baseline features, our traces)"] = u_names
    y_all = np.array([r["label"] for r in rows])
    models = sorted({(r["model_id"].split("/")[-1], r["reasoning_effort"]) for r in rows})
    print(f"{len(rows)} labelled cases from {len({r['problem_id'] for r in rows})} problems, {models}")
    print(f"always-True accuracy {y_all.mean():.3f}\n")

    for name, names in sets.items():
        accs, aucs, probs, y = cross_validate(rows, names, args.folds, args.repeats, args.seed)
        print(f"{name:52s} accuracy {accs.mean():.3f} ± {accs.std():.3f} | AUC {np.nanmean(aucs):.3f} ({args.folds}-fold x {args.repeats})")
        by = defaultdict(list)
        for i, r in enumerate(rows):
            by[(r["model_id"].split("/")[-1], r["reasoning_effort"])].append(i)
        print("    per model: " + "; ".join(
            f"{m} {e}: {np.mean((probs[idx] >= 0.5) == y[idx]):.2f} (vs {y[idx].mean():.2f}, n={len(idx)})"
            for (m, e), idx in sorted(by.items())))

    if args.export:
        from sklearn.linear_model import LogisticRegression

        names = sets[args.export]
        x, stats = matrix(rows, names)
        clf = LogisticRegression(C=0.5, max_iter=1000).fit(x, y_all)
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps({
            "feature_set": args.export, "features": names,
            "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
            "stats": {f"{m}||{e}": {"median": s[0].tolist(), "sd": s[1].tolist(), "mean": s[2].tolist()}
                      for (m, e), s in stats.items()},
            "n": int(rows[0]["n"]), "budget": int(rows[0]["budget"]),
            "trained_on": len(rows),
        }, indent=2))
        print(f"\nwrote {ARTIFACT}")


if __name__ == "__main__":
    main()
