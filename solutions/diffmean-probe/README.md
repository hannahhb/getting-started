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

## Behaviour score (short resampling)

Besides the prompt's hidden states, each model samples 8 continuations of 64
tokens from the prompt (temperature 1, fixed seed) and four numbers summarize
how much they disagree: dispersion of the continuations' mid-layer states,
the same over the first 16 tokens, mean next-token entropy, and mean NLL of the
sampled tokens. Training fits a direction over these (standardized), the same
way as for hidden states, and weights it with `gamma` in {0, 0.3, 0.5}:

```text
drop_hat = prior + beta * s_h * z_hidden + gamma * s_b * z_behaviour
```

At evaluation time this costs one short generation per problem.

## More labelled data (`datagen/`)

Since the 19 Sep clarification, self-generated labels are allowed. The
pipeline in `datagen/` builds them in the `train-main-v2` schema:

1. `build_problems.py`: 296 integer-answer problems at pinned revisions: all
   of AIME 2022-24 and AMC 2022-23 (AI-MO validation sets), plus 123 problems
   from Omni-MATH-2 (`martheballon/Omni-MATH-2`, the manually revised
   Omni-MATH), pinned by id in `datagen/omni2_ids.json`. Omni-MATH-2 problems
   tagged non-standard and answers that are not a single integer are skipped,
   and problems in `train-main-v2` are excluded.
2. `perturb.py`: one rule-based `typos` variant per problem, plus one variant
   per problem of rephrase, rename, domain, distract, and **expert** (an
   answer-preserving edit that needs real understanding of the problem, e.g. an
   equivalent but unfamiliar restatement of a condition). The written variants
   were produced by Claude and are stored in `datagen/written_variants.jsonl`,
   each with a `check` note on why its answer is unchanged. Variants whose
   answer could not be verified were left out. Some source answers look wrong
   even in Omni-MATH-2 (e.g. the a+b+ab blackboard problem, listed as 101 but
   actually 100); `label.py` drops problems a model cannot solve unperturbed,
   which removes most such cases.
3. `sample.py`: each target model answers every original 8 times and every
   variant 4 times, at temperature 1.0 with up to 16k tokens; truncated
   answers count as wrong.
   For a shorter run, `datagen/sample_vllm.slurm` samples one model on the
   first 100 problems with 4/2 samples and 8k tokens
   (`sbatch datagen/sample_vllm.slurm <model-id>`). `--backend openrouter`
   samples through OpenRouter instead (key in `OPENROUTER_API_KEY`), which is
   how gpt-oss-120b is sampled; it records the serving provider per sample.
4. `label.py`: `max_drop = max over types (base accuracy - type accuracy)`,
   labelled with the 0.10/0.25 thresholds. Problems with base accuracy below
   0.5 are dropped, following the organizers' rule of dropping rows where the
   original is not robustly correct.

Step 3 needs vLLM, which pins its own torch, so run it in a separate
environment (e.g. `python -m venv $PROJ/venvs/vllm && $PROJ/venvs/vllm/bin/pip install vllm`).
Steps 1, 2, and 4 run with `uv run`. Each script's `--help` lists its options.

## Reproduce on a GPU (1–2 A100)

```bash
# 1. Generate labels (optional; skip to use train-main-v2 only).
uv run datagen/build_problems.py
uv run datagen/perturb.py
for m in Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B allenai/Olmo-3-7B-Think deepseek-ai/DeepSeek-R1-0528-Qwen3-8B; do
  $PROJ/venvs/vllm/bin/python datagen/sample.py --model "$m"
done
uv run datagen/label.py

# 2. Every model reads every row: hidden states plus behaviour features.
uv run solutions/diffmean-probe/scripts/extract_features.py \
  --extra-rows data/labelgen/labels.jsonl \
  --output-dir data/diffmean-probe/features-extended \
  --models Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B \
           allenai/Olmo-3-7B-Think deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
# gpt-oss-120b offloads to CPU on A100s; --skip-behaviour avoids slow generation there.
uv run solutions/diffmean-probe/scripts/extract_features.py \
  --extra-rows data/labelgen/labels.jsonl \
  --output-dir data/diffmean-probe/features-extended \
  --models openai/gpt-oss-120b --skip-behaviour

# 3. Select by grouped CV and export (add --nested for the nested estimate).
uv run solutions/diffmean-probe/scripts/train_diffmean.py \
  --features-dir data/diffmean-probe/features-extended
```

Use a new `--output-dir` whenever the rows change: training refuses feature
files whose rows are not in the same order.

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
