# RL alignment and multi-agent PPO

**[Read the research report](RESEARCH_REPORT.md)** — measured results, ablation analysis, limitations, and separately labeled DPO planning estimates.

Two small research projects with real environments, real model training, fixed seeds,
CSV provenance, optional Weights & Biases, and reports generated from those logs.

- [marl-ppo](marl-ppo/README.md): PPO from scratch, CartPole/LunarLander sanity checks,
  IPPO/MAPPO and a complete 2×2 critic/sharing ablation.
- [dpo-llm](dpo-llm/README.md): Qwen LoRA DPO on UltraFeedback, reward-model evaluation,
  sandboxed MBPP preference mining and a second DPO stage.

Python **3.11**. From this directory:

```bash
# Install uv first if needed: https://docs.astral.sh/uv/
make setup
make test
make sanity                    # fresh paths; supplied logs are never overwritten
make marl WORKERS=1             # 5 seeds × 4 configurations × 200k joint transitions
# CUDA GPU (T4/L4), use the Colab notebook or:
make dpo
make code-prefs                 # Linux with libseccomp.so.2
make stage-b
make eval
make report
```

Use `WANDB=--wandb` with Make targets after `wandb login`. Every logged scalar is
written to long-form `metrics.csv` (`step,metric,value,wall_seconds`); the same
scalar values go to W&B when enabled. Each run writes `config.json` with software,
seed, hardware, and Git revision. Reports are reproducible from checked-in CSVs;
large checkpoints, adapters, and caches are ignored. Runs refuse existing output
directories. Choose fresh paths for reruns.

The initial host was an Apple M1 Pro (10 cores, 16 GiB RAM) macOS CPU, Python 3.11.7, PyTorch 2.6.0;
CUDA and MPS were unavailable to the process. GPU memory fit is an intended
configuration, not a measured claim. Consult each generated table for what
actually completed. The optional PPO/GRPO language-model baseline is not implemented.

Model and dataset licenses are separate from this repository's MIT code license.
See project READMEs for sources, experimental protocol, and limitations.
