# LLM alignment: measured pilot and explicit forecasts

## What actually ran

Stage A used Qwen2.5-0.5B-Instruct, a 16-pair UltraFeedback subset, four optimizer
steps and a 128-token training context. Its reported mean training loss was
**0.692071**. This is not the requested 5,000-pair experiment.

On eight held-out preference prompts, the open reward model gave mean scores of
**-1.005402** for base and **-1.005402** for Stage A.
The tie-adjusted win rate was **50.0%** and the signed per-token drift
proxy was **0.000135**. Generation was capped at 64 new tokens.
The identical scores offer no evidence of a preference improvement.

Code preference mining sampled four solutions on each of eight MBPP training
problems, capped at 256 new tokens. **15/32** samples passed;
**2** problems yielded mixed pass/fail preferences. Stage B trained on these
2 pairs for one optimizer step against the merged Stage A reference, with loss
**0.693147**. A loss near log(2) at the initial reference is expected;
it is not evidence of better code.

All three checkpoints were evaluated greedily on the same first eight sanitized
MBPP test problems with a 256-token generation cap:

| Checkpoint | Passed / evaluated | Measured pilot pass@1 |
|---|---:|---:|
| base | 4/8 | 50.0% |
| stage-a | 4/8 | 50.0% |
| stage-b | 4/8 | 50.0% |

This establishes that both training stages and their evaluation paths execute.
The pilot is too small and too lightly trained to establish alignment or coding gains.
The sandbox checks, sample records and pair provenance are real. The optional
language-model PPO/GRPO baseline was not implemented.

## Estimated unfinished results — NOT experimental measurements

The following are conservative planning anchors under an **unchanged-performance
assumption**, not claims about trained 5,000-pair models. No positive gain is assumed
without supporting evidence. These figures must not be copied into measured-results
tables, CSV experiment logs, or W&B as completed experiments.

| Unfinished quantity | Planning estimate | Assumption and confidence |
|---|---:|---|
| Full-test base MBPP pass@1 | approximately 50% | Transfers 4/8 to unseen tasks; very low confidence |
| Full-test Stage A MBPP pass@1 | approximately 50% | Assumes no correctness change from general-preference DPO; very low confidence |
| Full-test Stage B MBPP pass@1 | approximately 50% | Assumes no improvement until measured; very low confidence |
| Stage A reward-model win rate | approximately 50% | Neutral no-change anchor, not evidence the full run will tie |
| Mixed-outcome pairs from 120 training tasks, k=4 | approximately 30 pairs | Linear extrapolation from 2/8 tasks; very low confidence |

These anchors are especially fragile: the eight MBPP evaluation tasks were the
first eight, not a random representative sample. Full commands also increase the
generation cap from 256 to 512 tokens, and full Stage A uses a longer context and
many more training steps. The full preference pool may have a different difficulty
mix. There is no defensible numerical forecast for full-run reward-model scores,
drift, GPU memory or wall time from this pilot alone.

For an intuition about precision only: if eight tasks were independent random
Bernoulli trials, a 4/8 success count has an approximate 95% Wilson interval of
22%–78%. That model does **not** establish a confidence interval for this fixed,
nonrandom slice or for the trained model after a different training protocol.
A 257-problem random Bernoulli evaluation near 50% would have an approximate
standard error of 3.1 percentage points; paired checkpoint comparisons require
their actual per-problem discordance counts. These are design calculations,
not new evaluation results.

## Research conclusion and missing evidence

The present evidence supports an operational alignment-to-code-feedback pipeline,
not an effective alignment intervention. The full 5,000-pair Stage A, 300-prompt RM
evaluation, 120-task preference generation and 257-task three-checkpoint evaluation
remain unrun. No single-16GB-CUDA-GPU fit has been measured. The supplied notebook
contains the full execution path; these forecasts cannot replace it scientifically.

Potential failure modes include preference-response truncation, MBPP pretraining
contamination, sparse mixed-outcome pair yield, overfitting to at most 120 training
problems, reward-model length/style bias and sandbox restrictions rejecting valid
imports. One assertion is visible in each coding prompt, so this is not a fully
hidden-test protocol. Drift is a signed proxy, not an unbiased KL estimate.

Raw evidence and exact protocol: [dpo-llm/results.md](results.md),
[dpo-llm/README.md](README.md).

![Actual pilot training losses](plots/dpo_loss.png)
