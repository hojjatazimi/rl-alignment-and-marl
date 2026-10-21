"""Generate an evidence-based portfolio report; forecasts never enter measured logs."""
import csv
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def metrics(relative):
    path = ROOT / relative
    if not (path / 'complete.json').exists():
        raise RuntimeError(f'Incomplete run: {path}')
    with (path / 'metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    result = {}
    for row in rows:
        result.setdefault(row['metric'], []).append(float(row['value']))
    return result


def summary(values):
    return f'{st.mean(values):.3f} ± {st.stdev(values):.3f}'


def main():
    algorithms = ['ippo', 'shared-local', 'independent-central', 'mappo']
    required = [ROOT/f'marl-ppo/runs/{a}-seed{s}/complete.json' for a in algorithms for s in range(5)]
    required += [ROOT/f'dpo-llm/runs/{name}/complete.json' for name in [
        'cpu-pilot', 'cpu-pilot-stage-b', 'cpu-pilot-code-prefs', 'cpu-pilot-rm',
        'cpu-pilot-mbpp-base', 'cpu-pilot-mbpp-stage-a', 'cpu-pilot-mbpp-stage-b']]
    required += [ROOT/f'marl-ppo/runs/{name}/complete.json' for name in ['cartpole-seed0','lunarlander-seed0']]
    if not all(p.exists() for p in required):
        print('Combined initial-run analysis skipped: bundled initial run logs are absent. Project-specific reports remain available.')
        return
    runs = {a: [metrics(f'marl-ppo/runs/{a}-seed{s}') for s in range(5)] for a in algorithms}
    last = lambda a, k: [r[k][-1] for r in runs[a]]
    rows = []
    for a in algorithms:
        rows.append(f"| {a} | {summary(last(a, 'eval_return'))} | {summary(last(a, 'coverage'))} | {summary(last(a, 'collisions'))} |")
    contrasts = [
        ('Centralized critic, independent networks', 'independent-central', 'ippo'),
        ('Centralized critic, shared networks', 'mappo', 'shared-local'),
        ('Parameter sharing, decentralized critic', 'shared-local', 'ippo'),
        ('Parameter sharing, centralized critic', 'mappo', 'independent-central'),
        ('MAPPO versus IPPO (both factors change)', 'mappo', 'ippo'),
    ]
    contrast_rows = []
    for label, a, b in contrasts:
        delta = [x-y for x, y in zip(last(a, 'eval_return'), last(b, 'eval_return'))]
        mean = st.mean(delta)
        half = 2.7764451051977987 * st.stdev(delta) / math.sqrt(5)
        contrast_rows.append(f'| {label} | {mean:+.3f} | [{mean-half:+.3f}, {mean+half:+.3f}] |')
    cart = metrics('marl-ppo/runs/cartpole-seed0')
    lunar = metrics('marl-ppo/runs/lunarlander-seed0')
    a = metrics('dpo-llm/runs/cpu-pilot')
    b = metrics('dpo-llm/runs/cpu-pilot-stage-b')
    prefs = metrics('dpo-llm/runs/cpu-pilot-code-prefs')
    rm = metrics('dpo-llm/runs/cpu-pilot-rm')
    mbpp = {name: metrics(f'dpo-llm/runs/cpu-pilot-mbpp-{name}') for name in ['base', 'stage-a', 'stage-b']}
    count = len(mbpp['base']['passed'])
    assert count == 8
    assert all(len(r['passed']) == count for r in mbpp.values())
    pairs = int(prefs['pairs'][-1])
    pair_forecast = 120 * pairs / 8
    measured_mbpp = '\n'.join(f"| {name} | {int(sum(r['passed']))}/{count} | {r['pass_at_1'][-1]:.1%} |" for name, r in mbpp.items())
    marl = f'''# Multi-agent PPO: measured findings

The implementation solves CartPole in the recorded seed: final evaluation return
**{cart['eval_return'][-1]:.2f} ± {cart['eval_return_std'][-1]:.2f}** over 20 fixed episodes.
LunarLander-v3 reaches **{lunar['eval_return'][-1]:.2f} ± {lunar['eval_return_std'][-1]:.2f}**;
this is a negative sanity-check result, not a solved environment. Both use 200,000 steps.

## Equal-budget comparison

All four configurations ran five training seeds at 200,000 joint environment
transitions per seed, or 600,000 agent transitions. Values below are mean ± sample
standard deviation across training seeds. Final evaluation uses 20 fixed episodes.
Higher return is better; coverage is a fraction; collisions count unordered pairs per step.

| Configuration | Final return | Coverage | Collision pairs/step |
|---|---:|---:|---:|
{chr(10).join(rows)}

## Factorial interpretation

| Matched-seed return contrast | Mean difference | Approximate 95% t interval |
|---|---:|---:|
{chr(10).join(contrast_rows)}

These exploratory intervals use five paired training-seed differences and a
Student-t reference with four degrees of freedom. They assume approximately normal,
independent seed differences, do not correct for multiple comparisons, and do not
establish equivalence. They are estimates derived from real runs, not additional runs.

The observed MAPPO–IPPO difference is small relative to seed variation. Centralizing
the critic changes the mean less once parameters are already shared. The experiment
provides no compelling evidence of a consistently better configuration at this budget.
Parameter sharing changes both actor and critic parameterization; it is not an isolated
actor-sharing intervention.

Coverage remains low for every configuration. The curves therefore support modest
reward improvement, not successful landmark coverage. Collision avoidance and coarse
positioning can improve the composite reward while leaving landmarks uncovered; this
is a plausible interpretation, not a causal diagnosis from the present logs.

## Reproducibility and limits

The PPO core includes GAE with separate termination/truncation treatment, clipping,
advantage normalization, entropy/value terms, minibatch epochs and gradient clipping.
Evaluation curves are unsmoothed. Hardware was an Apple M1 Pro, 10 CPU cores, 16 GiB
RAM, Python 3.11.7 and PyTorch 2.6.0, with one PyTorch thread per process. Concurrent
jobs make recorded elapsed times unsuitable for clean throughput comparisons.

Five seeds, one task, a short budget and fixed monitoring seeds limit generalization.
Longer training, independent evaluation seeds and stronger coverage diagnostics are
needed before claiming a coordination advantage. Training loss alone is not a
performance measure. No hyperparameter sweep or best-checkpoint selection is claimed.

Reproduce numeric tables and plots with `make report`. Raw scalar logs and protocol:
[marl-ppo/results.md](marl-ppo/results.md), [marl-ppo/README.md](marl-ppo/README.md).

![Mean ± standard deviation over five seeds](marl-ppo/plots/marl_returns.png)
'''
    dpo = f'''# LLM alignment: measured pilot and explicit forecasts

## What actually ran

Stage A used Qwen2.5-0.5B-Instruct, a 16-pair UltraFeedback subset, four optimizer
steps and a 128-token training context. Its reported mean training loss was
**{a['train_loss'][-1]:.6f}**. This is not the requested 5,000-pair experiment.

On eight held-out preference prompts, the open reward model gave mean scores of
**{rm['mean_rm_base'][-1]:.6f}** for base and **{rm['mean_rm_dpo'][-1]:.6f}** for Stage A.
The tie-adjusted win rate was **{rm['mean_win'][-1]:.1%}** and the signed per-token drift
proxy was **{rm['mean_drift'][-1]:.6f}**. Generation was capped at 64 new tokens.
The identical scores offer no evidence of a preference improvement.

Code preference mining sampled four solutions on each of eight MBPP training
problems, capped at 256 new tokens. **{int(sum(prefs['passed']))}/{len(prefs['passed'])}** samples passed;
**{pairs}** problems yielded mixed pass/fail preferences. Stage B trained on these
{pairs} pairs for one optimizer step against the merged Stage A reference, with loss
**{b['train_loss'][-1]:.6f}**. A loss near log(2) at the initial reference is expected;
it is not evidence of better code.

All three checkpoints were evaluated greedily on the same first eight sanitized
MBPP test problems with a 256-token generation cap:

| Checkpoint | Passed / evaluated | Measured pilot pass@1 |
|---|---:|---:|
{measured_mbpp}

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
| Mixed-outcome pairs from 120 training tasks, k=4 | approximately {pair_forecast:.0f} pairs | Linear extrapolation from {pairs}/8 tasks; very low confidence |

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

Raw evidence and exact protocol: [dpo-llm/results.md](dpo-llm/results.md),
[dpo-llm/README.md](dpo-llm/README.md).

![Actual pilot training losses](dpo-llm/plots/dpo_loss.png)
'''
    intro = '''# RL alignment and multi-agent PPO: research report

**Evidence status:** the multi-agent experiment is complete at the stated budget;
the LLM experiment has real CPU pilot results only. Estimated unfinished LLM
outcomes are labeled separately below. No forecast is represented as a completed run.

## Executive assessment

The strongest portfolio result is an auditable from-scratch PPO implementation,
a solved CartPole sanity check, and a balanced five-seed, two-factor cooperative
multi-agent experiment. The small differences between methods are a valid negative
result. The DPO project demonstrates a working general-preference-to-code-feedback
pipeline but currently supports no claim of improved preference quality or correctness.

This report is generated from completed CSV logs by
`scripts/write_research_report.py`. Run `make report` to regenerate it. Seeds,
software versions, immutable Hugging Face asset revisions and hardware provenance
are stored alongside the source and logs. Forecasts are narrative planning assumptions;
they are never inserted into the measured results tables.

'''
    (ROOT/'RESEARCH_REPORT.md').write_text(intro+marl+'\n---\n\n'+dpo)
    # Adjust links for the project-specific copies.
    for project, text in [('marl-ppo', marl), ('dpo-llm', dpo)]:
        (ROOT/project/'DISCUSSION.md').write_text(text.replace(f']({project}/', ']('))
    print('Generated RESEARCH_REPORT.md and both DISCUSSION.md files from completed logs.')


if __name__ == '__main__':
    main()
