"""Unit tests for the diff-in-means probe: scoring, training, and inference."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOLUTION_ROOT = ROOT / "solutions" / "diffmean-probe"
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

from diffmean_probe import inference  # noqa: E402
from diffmean_probe.artifact import load_config, load_directions  # noqa: E402
from diffmean_probe.common import normalize_effort, problem_token_mask  # noqa: E402
from diffmean_probe.scoring import (  # noqa: E402
    fit_direction,
    fit_priors,
    layer_window,
    predict_drop,
    prior_mean,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TRAINING = load_module("diffmean_training_test", SOLUTION_ROOT / "scripts" / "train_diffmean.py")
SOLUTION = load_module("diffmean_solution_test", SOLUTION_ROOT / "solution.py")


def write_synthetic_features(directory: Path, signal: float, seed: int = 0) -> None:
    """Two models sharing 24 problems; layer 2 carries max_drop when ``signal > 0``."""
    rng = np.random.default_rng(seed)
    problems = [f"problem {i}" for i in range(24)]
    for model_id, dim in (("org/a", 8), ("org/b", 6)):
        max_drop = np.round(rng.choice([0.0, 0.05, 0.15, 0.3, 0.6], size=len(problems)), 2)
        max_drop[:3], max_drop[3:6] = 0.0, 0.5
        label = np.where(max_drop <= 0.10, 1, np.where(max_drop >= 0.25, 0, -1)).astype(np.int8)
        last = rng.normal(size=(len(problems), 5, dim)).astype(np.float32)
        last[:, 2, 0] += signal * (max_drop - max_drop.mean()) / max_drop.std()
        np.savez(
            directory / f"{model_id.replace('/', '_')}.npz",
            model_id=np.asarray(model_id),
            last=last,
            mean=rng.normal(size=last.shape).astype(np.float32),
            problem=np.asarray(problems),
            reasoning_effort=np.asarray(["default"] * len(problems)),
            max_drop=max_drop,
            label=label,
        )


def write_shared_features(directory: Path, signal: float, seed: int = 0) -> None:
    """Three models labelling the same 20 problems; every model reads every row.

    Fragility is a property of the problem (shared across models), and each
    encoder carries it on layers 1-3 of its own hidden space.
    """
    rng = np.random.default_rng(seed)
    models = (("org/a", 8), ("org/b", 6), ("org/c", 5))
    problems = [f"problem {i}" for i in range(20)]
    fragile = np.arange(len(problems)) % 3 == 0
    rows = [(m, p, k) for m, _ in models for k, p in enumerate(problems)]
    max_drop = np.asarray([0.5 if fragile[k] else 0.0 for _, _, k in rows]) + rng.normal(0, 0.02, len(rows))
    max_drop = np.clip(max_drop, 0.0, 1.0).round(2)
    label = np.where(max_drop <= 0.10, 1, np.where(max_drop >= 0.25, 0, -1)).astype(np.int8)
    for model_id, dim in models:
        last = rng.normal(size=(len(rows), 5, dim)).astype(np.float32)
        last[:, 1:4, 0] += signal * np.asarray([1.0 if fragile[k] else -1.0 for _, _, k in rows])[:, None]
        np.savez(
            directory / f"{model_id.replace('/', '_')}.npz",
            model_id=np.asarray(model_id),
            last=last,
            mean=rng.normal(size=last.shape).astype(np.float32),
            row_model_id=np.asarray([m for m, _, _ in rows]),
            problem=np.asarray([p for _, p, _ in rows]),
            reasoning_effort=np.asarray(["default"] * len(rows)),
            max_drop=max_drop,
            label=label,
        )


class ScoringTest(unittest.TestCase):
    def test_direction_tracks_target(self) -> None:
        rng = np.random.default_rng(0)
        targets = rng.random(30)
        features = rng.normal(size=(30, 2, 4))
        features[:, :, 1] += 5.0 * targets[:, None]
        direction = fit_direction(features, targets)
        self.assertGreater(np.corrcoef(direction.score(features), targets)[0, 1], 0.9)
        self.assertAlmostEqual(direction.target_scale, float(targets.std()))
        self.assertAlmostEqual(float(direction.score(features).std()), 1.0)

    def test_direction_requires_varying_target(self) -> None:
        self.assertIsNone(fit_direction(np.ones((6, 1, 3)), np.zeros(6)))
        self.assertIsNone(fit_direction(np.ones((3, 1, 3)), np.array([0.0, 0.5, 1.0])))

    def test_priors_fall_back_to_parent_levels(self) -> None:
        priors = fit_priors(["m", "m", "m"], ["low", "low", "high"], [0.0, 0.0, 0.6])
        self.assertLess(prior_mean(priors, "m", "low"), prior_mean(priors, "m", "high"))
        self.assertEqual(prior_mean(priors, "m", "medium"), priors["models"]["m"]["mean"])
        self.assertEqual(prior_mean(priors, "unknown", "low"), priors["mean"])

    def test_predicted_drop_moves_with_score(self) -> None:
        drop = predict_drop(0.2, np.array([-1.0, 0.0, 1.0]), beta=0.5, target_scale=0.2)
        np.testing.assert_allclose(drop, [0.1, 0.2, 0.3])

    def test_layer_window_skips_embeddings_and_stays_in_range(self) -> None:
        self.assertEqual(layer_window(37, 0.5), [17, 18, 19])
        self.assertEqual(layer_window(37, 0.0), [1, 2, 3])
        self.assertEqual(layer_window(37, 1.0), [34, 35, 36])

    def test_effort_and_token_mask(self) -> None:
        self.assertEqual(normalize_effort("openai/gpt-oss-120b", "default"), "medium")
        self.assertEqual(normalize_effort("Qwen/Qwen3.5-4B", None), "default")
        self.assertEqual(problem_token_mask([(0, 3), (3, 8), (8, 9)], 4, 8), [False, True, False])


class TrainingTest(unittest.TestCase):
    def run_training(self, writer, signal: float, *extra: str):
        workdir = tempfile.TemporaryDirectory()
        features_dir, output_dir = Path(workdir.name) / "features", Path(workdir.name) / "artifacts"
        features_dir.mkdir()
        writer(features_dir, signal)
        args = ["--features-dir", str(features_dir), "--output-dir", str(output_dir),
                "--repeats", "2", "--outer-repeats", "1", *extra]
        with mock.patch("builtins.print"), mock.patch("sys.stderr"):
            summary = TRAINING.main(args)
        return summary, output_dir, workdir

    def test_own_rows_signal_is_selected_and_exported(self) -> None:
        summary, output_dir, workdir = self.run_training(write_synthetic_features, 4.0)
        with workdir:
            config = load_config(output_dir)
            self.assertFalse(summary["shared_features"])
            self.assertEqual(config["site"], "last")
            self.assertEqual(config["target"], "max_drop")
            self.assertGreater(config["beta"], 0.0)
            self.assertGreater(summary["selected_accuracy"], summary["prior_only_accuracy"])
            self.assertIn(2, load_directions("org/a", output_dir).layers)

    def test_shared_probe_learns_from_other_models(self) -> None:
        summary, output_dir, workdir = self.run_training(write_shared_features, 3.0, "--nested")
        with workdir:
            self.assertTrue(summary["shared_features"])
            self.assertGreater(summary["nested_accuracy"], summary["always_true"])
            config = load_config(output_dir)
            self.assertGreater(config["beta"], 0.0)
            self.assertEqual(sorted(config["models_with_directions"]), ["org/a", "org/b", "org/c"])

    def test_sign_test(self) -> None:
        self.assertEqual(TRAINING.sign_test_p(0, 0), 1.0)
        self.assertAlmostEqual(TRAINING.sign_test_p(3, 0), 0.125)
        self.assertAlmostEqual(TRAINING.sign_test_p(1, 1), 0.75)


class InferenceTest(unittest.TestCase):
    def test_prior_only_artifact_never_loads_a_model(self) -> None:
        with mock.patch.object(inference, "_scores", side_effect=AssertionError("model loaded")):
            predictions = SOLUTION.are_robust("openai/gpt-oss-120b", "default", ["a", "b"])
        self.assertEqual(predictions, [True, True])
        self.assertTrue(all(type(value) is bool for value in predictions))

    def test_directions_shift_predictions_and_failures_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            features_dir, output_dir = Path(workdir) / "features", Path(workdir) / "artifacts"
            features_dir.mkdir()
            write_synthetic_features(features_dir, signal=4.0)
            with mock.patch("builtins.print"):
                TRAINING.main(["--features-dir", str(features_dir), "--output-dir", str(output_dir),
                               "--repeats", "1"])
            with mock.patch.object(inference, "_scores", return_value=np.array([-50.0, 50.0, 0.0])):
                predictions = inference.predict_robustness("org/a", "default", ["x", "y", "z"], output_dir)
            self.assertEqual(predictions[:2], [True, False])
            with mock.patch.object(inference, "_scores", side_effect=RuntimeError("oom")):
                fallback = inference.predict_robustness("org/a", "default", ["x"], output_dir)
            self.assertEqual(len(fallback), 1)
            self.assertIs(type(fallback[0]), bool)
            self.assertEqual(inference.predict_robustness("org/a", "default", [], output_dir), [])

    def test_missing_config_predicts_false(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            self.assertEqual(inference.predict_robustness("m", "default", ["p"], Path(workdir)), [False])


if __name__ == "__main__":
    unittest.main()
