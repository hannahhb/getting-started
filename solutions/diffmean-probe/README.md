# Mean-Difference Probe on max_drop (Phase 0)

A low-variance representation probe for the competition models. It is the
first step of a larger plan; later phases add perturbation-steering
sensitivity features on top of it.

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
   drop_hat = prior(model, effort) + beta * std(max_drop | model) * mean_z(3 layers at depth f)
   robust   = drop_hat < threshold
   ```

   `site` (last prompt token or problem-span mean), `f`, `beta`, and
   `threshold` are shared across models and chosen by grouped CV over all
   models. `beta` acts as the out-of-sample correlation, and `beta = 0` (prior
   only) is part of the grid.
5. **Honest validation.** Rows are grouped by problem text *across models*
   (the 5 models share problems), and `--nested` reports an outer-loop estimate
   of the full selection procedure.

## Data rules

The competition paper (arXiv 2607.13899, §2.2) forbids extra *labelled* data
for the perturbation types in the final validation set (rephrase, rename,
domain, distract, typos, expert perturbations, expert no-solution). Unlabelled
data, and labels for other perturbation types, are allowed. This solution
trains only on the organizers' `train-main-v2`.

## Checked-in artifact

`artifacts/config.json` is currently **prior-only** (`beta = 0`), built from
`aimo-interp/train-main-v2@fa8af75` (94 rows, 80 labelled, 36 problems, 73.75%
of labelled rows robust). Every prior mean is below the selected threshold, so
this artifact is the same as always-True on that data. It is a valid fallback;
the representation part needs the GPU steps below.

## Reproduce on a GPU (1–2 A100 80GB)

```bash
# 1. Cache every-layer features for the four small models (bf16, minutes each).
uv run solutions/diffmean-probe/scripts/extract_features.py \
  --models Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B \
           allenai/Olmo-3-7B-Think deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

# 2. Optional: gpt-oss-120b. On A100, MXFP4 may dequantize to bf16 (~240 GB);
#    device_map="auto" then offloads to CPU, which is slow but fine for 45 prompts.
uv run solutions/diffmean-probe/scripts/extract_features.py --models openai/gpt-oss-120b

# 3. Select by grouped CV, report the nested estimate, and export.
uv run solutions/diffmean-probe/scripts/train_diffmean.py \
  --features-dir data/diffmean-probe/features --nested
```

Compare `nested_accuracy` against `always_true` in the printed summary. Ship
the representation artifact only if it beats the prior out of sample. Models
whose directions cannot be fitted (fewer than 4 rows, or a constant target)
fall back to their prior at inference time, and so does a failed forward pass.

## Inference cost

When `beta > 0`, each problem needs one forward pass over the prompt, with no
generation. That is well under the 3,600 s budget for every model, gpt-oss-120b
included.
