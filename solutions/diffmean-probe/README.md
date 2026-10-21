# Mean-Difference Probe on max_drop (Phase 0)

A low-variance representation probe for the competition models, trained as a
**shared probe**: each model's direction learns from every model's labels.
It is the first step of a larger plan; later phases add perturbation-steering
sensitivity features on top of it.

## Shared probe

Robustness is mostly a property of the problem. In `train-main-v2`, of the 15
problems labelled for two or more models, 14 are robust for all of them or
for none. So every model reads all 94 training rows (36 problems), and each
model's direction is fitted on all of them, with the target
`max_drop - prior(row's model, effort)`. At evaluation time a model is still
scored only from its own activations. This gives each direction ~94 rows
instead of 10–45. Feature files that hold only a model's own rows still load
and train the older per-model probe.

## What changes relative to the baselines

1. **Regress `max_drop`, not the boolean.** In `train-main-v2`, labels follow
   `max_drop`: robust at <= 0.10, not robust at >= 0.25, and `None` in
   between. Regressing the continuous target keeps the 14 unlabelled-band rows
   for fitting (94 rows instead of 80). Only the 80 labelled rows are scored.
2. **Accuracy-optimal decision.** The leaderboard scores plain accuracy, but
   the uncertainty baseline tunes its threshold for *balanced* accuracy. Here
   the threshold on the predicted drop is picked by CV for plain accuracy.
3. **Hierarchical priors.** Shrunken means of `max_drop` for global → model →
   `(model, reasoning_effort)`. gpt-oss `"default"` maps to `"medium"`.
4. **Covariance direction instead of logistic probes.** With ~10–45 rows per
   model, the direction `sum_i (h_i - mean_h)(y_i - mean_y)` (the regression
   form of diff-in-means, i.e. one-component PLS) is far more stable than a
   fitted linear probe. Projections are standardized per layer, so scores are
   comparable across models:

   ```text
   drop_hat = prior(model, effort) + beta * std(residual | model) * z(3 layers at depth f)
   robust   = drop_hat < threshold
   ```

   `site` (last prompt token or problem-span mean), `f` in {0.3, 0.5, 0.7},
   `beta` in {0, 0.3, 0.5}, and `threshold` in {0.175, 0.25} are shared
   across models and chosen by grouped CV over all models. The grid is kept
   small because a 648-setting search overfit the inner CV. `beta` acts as the out-of-sample correlation, and `beta = 0` (prior
   only) is part of the grid.
5. **Honest validation.** Rows are grouped by problem text *across models*
   (the 5 models share problems), and `--nested` reports an outer-loop estimate
   of the full selection procedure.

## Data rules

A 19 Sep 2026 clarification retired the earlier ban on extra labelled data.
Test-time perturbation generation is allowed but limited by the 3,600 s
budget, and any leakage of evaluation data means disqualification. Award
candidates must make their score reproducible from the technical report alone,
so every training source must be documented. This solution currently trains
only on the organizers' `aimo-interp/train-main-v2@fa8af75`.

## Checked-in artifact

`artifacts/config.json` is currently **prior-only** (`beta = 0`), built from
`aimo-interp/train-main-v2@fa8af75` (94 rows, 80 labelled, 36 problems, 73.75%
of labelled rows robust). Every prior mean is below the selected threshold, so
this artifact is the same as always-True on that data. It is a valid fallback;
the representation part needs the GPU steps below. Training overwrites
`artifacts/`, and its `cv` block records the CV summary.

## Reproduce on a GPU (1–2 A100 80GB)

```bash
# 1. Every model reads every training problem (36 prompts per model, minutes each).
uv run solutions/diffmean-probe/scripts/extract_features.py \
  --models Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B \
           allenai/Olmo-3-7B-Think deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

# 2. Optional: gpt-oss-120b. On A100, MXFP4 may dequantize to bf16 (~240 GB);
#    device_map="auto" then offloads to CPU, which is slow but fine for 60 prompts.
uv run solutions/diffmean-probe/scripts/extract_features.py --models openai/gpt-oss-120b

# 3. Select by grouped CV and export (add --nested for the nested estimate).
uv run solutions/diffmean-probe/scripts/train_diffmean.py \
  --features-dir data/diffmean-probe/features-shared
```

Features now go to `data/diffmean-probe/features-shared/`, so the older
own-rows files in `features/` are not overwritten.

## Export and nested CV

The training script always exports the selected configuration and saves every
model's direction. `--nested` also prints a repeated nested-CV estimate of the
whole selection procedure, compared row by row with always-True.

Results so far:

| Setup | Nested accuracy | vs always-True (0.7375) |
|---|---|---|
| Per-model probe, own rows only | 0.70 | 0 wins, 8 losses |
| Shared labels, gpt-oss features for all models (preview, 64 of 80 rows covered) | 0.741 | 3 wins, 2 losses, p = 0.5 |
| Shared probe, each model's own features | 0.704 | 2 wins, 10 losses |

Models whose directions cannot be fitted (fewer than 4 rows, or a constant
target) fall back to their prior at inference time, and so does a failed
forward pass.

## Inference cost

When `beta > 0`, each problem needs one forward pass over the prompt, with no
generation. That is well under the 3,600 s budget for every model, gpt-oss-120b
included.
