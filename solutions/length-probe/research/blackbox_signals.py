#!/usr/bin/env python3
"""Score black-box, test-time-computable signals against generated robustness labels.

Inputs: ``datagen`` samples and traces for one model/effort (or several).
Every feature uses only what a submission could compute at test time: the
model's responses to the ORIGINAL problem, plus (``typos_agree``) its answers
on a rule-based typo variant. No gold answer is used in any feature.

Labels follow ``datagen/label.py`` (max over perturbation types of the
accuracy drop; robust <= 0.10, not robust >= 0.25; originals solved < 0.5
dropped). When scoring ``typos_agree`` the typos type is left out of the label,
so the feature is not compared with itself.

Features (per problem, from the original's samples):
    len_mean / len_max   response tokens (log)
    trunc_frac           share of responses that hit the token cap
    hesitation           "wait", "alternatively", "let me double-check"... per 1k words
    churn                distinct candidate answers stated in the text
    consistency          share of samples giving the most common answer
    n_distinct           number of distinct final answers (incl. none)
    typos_agree          share of typo-variant answers equal to the original's majority answer

Prints per-feature AUC (oriented so > 0.5 means the expected direction:
robust problems are shorter, calmer, more consistent) and a small logistic
model scored by leave-one-problem-out across all inputs.

    python solutions/length-probe/research/blackbox_signals.py data/labelgen/samples/openai_gpt-oss-120b__low.jsonl \\
        data/labelgen/samples/openai_gpt-oss-120b__medium.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "datagen"))

from common import label_for, read_jsonl  # noqa: E402

HESITATION = re.compile(
    r"\b(wait|hmm+|alternatively|actually|let me (?:double[- ]check|re-?check|verify|reconsider)|"
    r"but (?:wait|hold on)|on second thought|i made (?:a|an) (?:mistake|error)|that'?s (?:not right|wrong))\b",
    re.IGNORECASE,
)
CANDIDATE = re.compile(r"(?:answer|result)\s*(?:is|=|:)\s*\$?\\?(?:boxed\{)?\s*(-?\d[\d,]*)", re.IGNORECASE)
# Expected sign: +1 if larger values mean robust, -1 if larger means NOT robust.
SIGNS = {"len_mean": -1, "len_max": -1, "trunc_frac": -1, "hesitation": -1, "churn": -1,
         "consistency": +1, "n_distinct": -1, "typos_agree": +1}


def auc(scores, positive) -> float:
    pos = [s for s, p in zip(scores, positive) if p]
    neg = [s for s, p in zip(scores, positive) if not p]
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def problem_rows(samples_path: Path, min_base: float) -> list[dict]:
    traces_path = samples_path.with_suffix(".traces.jsonl")
    samples = list(read_jsonl(samples_path))
    traces = {(t["problem_id"], t["type"], t["variant"]): t for t in read_jsonl(traces_path)}
    correct, answers, tokens = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list)), {}
    for s in samples:
        correct[s["problem_id"]][s["type"]] += s["correct"]
        answers[s["problem_id"]][s["type"]] += s["answers"]
        if s["type"] == "original":
            tokens[s["problem_id"]] = s["n_tokens"]
    model = samples[0]["model_id"] if samples else "?"
    effort = samples[0]["reasoning_effort"] if samples else "?"

    rows = []
    for pid, by_type in correct.items():
        if "original" not in by_type or len(by_type) < 2:
            continue
        base = mean(by_type["original"])
        if base < min_base:
            continue
        drops = {t: base - mean(v) for t, v in by_type.items() if t != "original" and v}
        trace = traces.get((pid, "original", 0), {})
        texts = [(r or "") + "\n" + (c or "") for r, c in zip(trace.get("reasoning") or [""] * len(trace.get("texts", [])),
                                                               trace.get("texts", []))]
        words = [max(len(t.split()), 1) for t in texts] or [1]
        orig_answers = answers[pid]["original"]
        majority, majority_count = Counter(orig_answers).most_common(1)[0]
        typo_answers = answers[pid].get("typos", [])
        rows.append({
            "model": model, "effort": effort, "problem_id": pid,
            "label": label_for(max(drops.values())),
            "label_without_typos": label_for(max([d for t, d in drops.items() if t != "typos"], default=0.0)),
            "len_mean": math.log(mean(tokens.get(pid, [1])) + 1),
            "len_max": math.log(max(tokens.get(pid, [1])) + 1),
            "trunc_frac": mean(f == "length" for f in trace.get("finish_reasons", ["stop"])),
            "hesitation": mean(1000 * len(HESITATION.findall(t)) / w for t, w in zip(texts, words)) if texts else 0.0,
            "churn": mean(len(set(CANDIDATE.findall(t))) for t in texts) if texts else 0.0,
            "consistency": majority_count / len(orig_answers),
            "n_distinct": len(set(orig_answers)),
            "typos_agree": mean(a == majority for a in typo_answers) if typo_answers and majority is not None else np.nan,
        })
    return rows


def loo_logistic(rows, features, label_key) -> tuple[float, float, float]:
    """Leave-one-problem-out logistic regression: (AUC, accuracy, majority-class accuracy)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = [r for r in rows if r[label_key] is not None and all(np.isfinite(r[f]) for f in features)]
    groups = np.array([r["problem_id"] for r in data])
    efforts = sorted({r["effort"] for r in data})
    x = np.array([[r[f] for f in features] + [r["effort"] == e for e in efforts[1:]] for r in data], dtype=float)
    y = np.array([bool(r[label_key]) for r in data])
    prob = np.zeros(len(y))
    for g in np.unique(groups):
        test = groups == g
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))
        clf.fit(x[~test], y[~test])
        prob[test] = clf.predict_proba(x[test])[:, 1]
    return auc(prob, y), float(np.mean((prob >= 0.5) == y)), float(max(y.mean(), 1 - y.mean()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", type=Path, nargs="+", help="samples/<model>__<effort>.jsonl files (traces beside them)")
    parser.add_argument("--min-base-accuracy", type=float, default=0.5)
    args = parser.parse_args()

    rows = [r for path in args.samples for r in problem_rows(path, args.min_base_accuracy)]
    for (model, effort) in sorted({(r["model"], r["effort"]) for r in rows}):
        part = [r for r in rows if (r["model"], r["effort"]) == (model, effort) and r["label"] is not None]
        y = [bool(r["label"]) for r in part]
        print(f"\n== {model} {effort}: {len(part)} labelled problems, {sum(y)} robust ({mean(y):.0%})")
        for feature, sign in SIGNS.items():
            key = "label_without_typos" if feature == "typos_agree" else "label"
            usable = [r for r in part if r[key] is not None and np.isfinite(r[feature])]
            value = auc([sign * r[feature] for r in usable], [bool(r[key]) for r in usable])
            print(f"   {feature:12s} AUC {value:.2f}  (n={len(usable)})")

    print("\n== combined, leave-one-problem-out across all inputs (effort as an extra input)")
    for name, features, key in [
        ("length only", ["len_mean"], "label"),
        ("length + consistency", ["len_mean", "consistency"], "label"),
        ("length + consistency + hesitation", ["len_mean", "consistency", "hesitation"], "label"),
        ("length + consistency + typos_agree (label without typos)", ["len_mean", "consistency", "typos_agree"], "label_without_typos"),
    ]:
        a, acc, base = loo_logistic(rows, features, key)
        print(f"   {name:58s} AUC {a:.2f} | accuracy {acc:.2f} vs majority-class {base:.2f}")


if __name__ == "__main__":
    main()
