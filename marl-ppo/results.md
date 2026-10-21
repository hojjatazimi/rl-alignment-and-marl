# Results generated from completed CSV logs

No smoothing or best-checkpoint selection. MARL return is the sum of per-step mean agent rewards; bands and table deviations are across seeds (sample standard deviation). Evaluation uses 20 fixed environment seeds and deterministic actions.

| Single-agent run | Seed | Steps | Final eval return ± episode std | Wall seconds |
|---|---:|---:|---:|---:|
| CartPole-v1 | 0 | 200000 | 500.00 ± 0.00 | 39.4 |
| LunarLander-v3 | 0 | 200000 | -80.55 ± 22.46 | 58.8 |

| Algorithm | Completed seeds | Joint env steps/seed | Return ± std | Coverage ± std | Collision pairs/step ± std | Wall seconds ± std |
|---|---:|---:|---:|---:|---:|---:|
| independent-central | 5 | 200000 | -20.427 ± 0.638 | 0.057 ± 0.011 | 0.105 ± 0.050 | 96.278 ± 8.488 |
| ippo | 5 | 200000 | -20.750 ± 1.135 | 0.048 ± 0.010 | 0.090 ± 0.039 | 85.355 ± 6.282 |
| mappo | 5 | 200000 | -20.259 ± 0.256 | 0.054 ± 0.013 | 0.094 ± 0.045 | 85.755 ± 4.460 |
| shared-local | 5 | 200000 | -20.292 ± 0.587 | 0.060 ± 0.018 | 0.092 ± 0.045 | 86.956 ± 4.514 |

Wall time includes training and evaluation, excludes GIF rendering. Concurrent suite runs contend for CPU; do not infer algorithm speed from this grid.

![MARL curves](plots/marl_returns.png)

![Single-agent curves](plots/single_returns.png)

Source logs: [cartpole-seed0](runs/cartpole-seed0/metrics.csv), [independent-central-seed0](runs/independent-central-seed0/metrics.csv), [independent-central-seed1](runs/independent-central-seed1/metrics.csv), [independent-central-seed2](runs/independent-central-seed2/metrics.csv), [independent-central-seed3](runs/independent-central-seed3/metrics.csv), [independent-central-seed4](runs/independent-central-seed4/metrics.csv), [ippo-seed0](runs/ippo-seed0/metrics.csv), [ippo-seed1](runs/ippo-seed1/metrics.csv), [ippo-seed2](runs/ippo-seed2/metrics.csv), [ippo-seed3](runs/ippo-seed3/metrics.csv), [ippo-seed4](runs/ippo-seed4/metrics.csv), [lunarlander-seed0](runs/lunarlander-seed0/metrics.csv), [mappo-seed0](runs/mappo-seed0/metrics.csv), [mappo-seed1](runs/mappo-seed1/metrics.csv), [mappo-seed2](runs/mappo-seed2/metrics.csv), [mappo-seed3](runs/mappo-seed3/metrics.csv), [mappo-seed4](runs/mappo-seed4/metrics.csv), [shared-local-seed0](runs/shared-local-seed0/metrics.csv), [shared-local-seed1](runs/shared-local-seed1/metrics.csv), [shared-local-seed2](runs/shared-local-seed2/metrics.csv), [shared-local-seed3](runs/shared-local-seed3/metrics.csv), [shared-local-seed4](runs/shared-local-seed4/metrics.csv)
