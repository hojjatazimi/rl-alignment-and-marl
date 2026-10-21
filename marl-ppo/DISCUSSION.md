# Multi-agent PPO: measured findings

The implementation solves CartPole in the recorded seed: final evaluation return
**500.00 ± 0.00** over 20 fixed episodes.
LunarLander-v3 reaches **-80.55 ± 22.46**;
this is a negative sanity-check result, not a solved environment. Both use 200,000 steps.

## Equal-budget comparison

All four configurations ran five training seeds at 200,000 joint environment
transitions per seed, or 600,000 agent transitions. Values below are mean ± sample
standard deviation across training seeds. Final evaluation uses 20 fixed episodes.
Higher return is better; coverage is a fraction; collisions count unordered pairs per step.

| Configuration | Final return | Coverage | Collision pairs/step |
|---|---:|---:|---:|
| ippo | -20.750 ± 1.135 | 0.048 ± 0.010 | 0.090 ± 0.039 |
| shared-local | -20.292 ± 0.587 | 0.060 ± 0.018 | 0.092 ± 0.045 |
| independent-central | -20.427 ± 0.638 | 0.057 ± 0.011 | 0.105 ± 0.050 |
| mappo | -20.259 ± 0.256 | 0.054 ± 0.013 | 0.094 ± 0.045 |

## Factorial interpretation

| Matched-seed return contrast | Mean difference | Approximate 95% t interval |
|---|---:|---:|
| Centralized critic, independent networks | +0.323 | [-1.506, +2.152] |
| Centralized critic, shared networks | +0.033 | [-0.635, +0.701] |
| Parameter sharing, decentralized critic | +0.458 | [-1.591, +2.508] |
| Parameter sharing, centralized critic | +0.168 | [-0.622, +0.959] |
| MAPPO versus IPPO (both factors change) | +0.491 | [-1.194, +2.177] |

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
[marl-ppo/results.md](results.md), [marl-ppo/README.md](README.md).

![Mean ± standard deviation over five seeds](plots/marl_returns.png)
