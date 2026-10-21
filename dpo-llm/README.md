# DPO alignment with a code-correctness reward

[Analysis and interpretation](DISCUSSION.md) · [Combined research report](../RESEARCH_REPORT.md)

**Question:** can general preference alignment be followed by a verifiable coding
preference stage, and does that change held-out correctness? This project connects
language-model alignment to coding-agent feedback without requiring a large policy.

Stage A trains LoRA adapters on 5,000 seeded UltraFeedback preference pairs using
Qwen2.5-0.5B-Instruct. DPO minimizes
`−log σ(β[(log π(chosen)−log π_ref(chosen))−(log π(rejected)−log π_ref(rejected))])`.
The frozen reference is the initial policy. TRL disables the fresh adapter for
reference inference and precomputes reference log probabilities. LoRA covers all
attention and MLP projections. Beta, learning rate, epochs, rank, accumulation and
context lengths live in YAML files.

Stage B samples four solutions for each **sanitized MBPP training** problem,
executes the assertions and selects one uniformly chosen passing/failing pair
per problem with mixed outcomes. All-pass and all-fail problems are skipped;
no preferences are invented. A second adapter is trained against the **Stage A
policy as its reference**, implemented by merging Stage A before adding fresh LoRA.
Stage B inference reconstructs that merged base using `lineage.json`; keep Stage A
and Stage B directories together and run commands from the repository root.

## Exact commands

Use Python 3.11 and a CUDA T4/L4. From repository root:

```bash
make setup
# optional 4-bit dependency, Linux/CUDA only:
uv pip install --python .venv/bin/python -r requirements-cuda.txt
make dpo
make code-prefs                 # Linux libseccomp.so.2 must be installed
make stage-b
make eval                      # RM 300 prompts; MBPP 257 test problems/checkpoint
make report
```

The [Colab notebook](notebooks/end_to_end.ipynb) installs a Python 3.11 environment,
accepts a repository URL or source zip, and runs the same commands end to end.
`make dpo` uses `configs/stage_a.yaml` (5k pairs, β=.1, lr=5e−6, one epoch, r=16,
1024 total/512 prompt tokens, batch 1, accumulation 16). Stage B uses lr=2e−6,
accumulation 8 (capped at the number of available minibatches for small preference sets). Models use fp16 on CUDA, gradient checkpointing and eager attention.
The single 16GB target has **not been measured on this CPU host**; CUDA runs record
peak allocated/reserved VRAM. If memory is tight, Stage A supports `--four-bit`;
Stage B uses a float16 merge and forbids 4-bit merging. Decrease context length if
needed, and record that protocol change. Reward and policy models are loaded
sequentially during evaluation to limit peak memory.

Explicit pilot commands used locally:

```bash
.venv/bin/python dpo-llm/train_dpo.py --config dpo-llm/configs/cpu_pilot.yaml --allow-cpu --out dpo-llm/runs/cpu-pilot
.venv/bin/python dpo-llm/eval_rm.py --adapter dpo-llm/runs/cpu-pilot/adapter --label cpu-pilot --limit 8 --max-new-tokens 64 --out dpo-llm/runs/cpu-pilot-rm
make sandbox
.venv/bin/python dpo-llm/build_code_prefs.py --adapter dpo-llm/runs/cpu-pilot/adapter --limit 8 --max-new-tokens 256 --sandbox docker --out dpo-llm/runs/cpu-pilot-code-prefs
.venv/bin/python dpo-llm/train_dpo.py --config dpo-llm/configs/stage_b.yaml --prefs dpo-llm/runs/cpu-pilot-code-prefs/preferences.jsonl --init-adapter dpo-llm/runs/cpu-pilot/adapter --allow-cpu --out dpo-llm/runs/cpu-pilot-stage-b
for stage in base stage-a stage-b; do
  args=()
  if [ "$stage" = stage-a ]; then args=(--adapter dpo-llm/runs/cpu-pilot/adapter); fi
  if [ "$stage" = stage-b ]; then args=(--adapter dpo-llm/runs/cpu-pilot-stage-b/adapter); fi
  .venv/bin/python dpo-llm/eval_mbpp.py --label pilot "${args[@]}" --limit 8 --max-new-tokens 256 --sandbox docker --out "dpo-llm/runs/cpu-pilot-mbpp-$stage"
done
```

Append `--wandb` to any experiment command, or `WANDB=--wandb` to Make targets.
Run paths must be fresh. To repeat a full pipeline set `RUN_ROOT=dpo-llm/runs/repeat`.
On macOS use `make sandbox` then `SANDBOX=docker`; the Docker daemon must be running.

## Evaluation protocol

Eval A uses 300 deduplicated prompts from UltraFeedback `test_prefs` with a fixed
seed, the same greedy decoder for base and adapter, and OpenAssistant's DeBERTa
reward model. It reports raw mean reward scores, paired win rate (ties count .5),
and a signed mean token log-probability difference from base on the **DPO-generated
continuations**, including EOS. This drift proxy is not an unbiased KL estimate.
Reward-model inputs use `(question, answer)` pairs and 512-token truncation.
Prompts are left-truncated to 1,024 policy tokens, consistently for generation and drift scoring. Raw completions, token IDs, per-prompt scores and drift are saved to JSONL.

Eval B uses the sanitized **257-problem test split**, greedy decoding and identical
prompts/512-token budgets for base, Stage A and Stage B. Prompts show the task plus
one assertion to establish the function signature. All assertions are executed;
this is a visible-test protocol, not strict hidden-test generalization. No canonical
solution is included in the prompt. Stage B uses only the 120-problem train split;
validation/test are never used to mine pairs. Full and pilot evaluations must not
be compared when their limits or token budgets differ.

## Execution sandbox

A subprocess alone is not a security sandbox. The Linux worker preloads selected
standard-library modules, sets CPU/address-space/file-size limits, and installs a
**seccomp syscall allowlist before candidate execution**. File opens, networking,
process spawning and exec are blocked. Tests and code run in a fresh namespace.
The parent enforces a wall timeout and requires a fresh completion marker, so
`os._exit(0)` cannot produce an accidental pass. Imports requiring new filesystem
reads are rejected, which can count otherwise valid solutions as failures.

The Docker backend adds a read-only filesystem, no network, dropped capabilities,
non-root UID, 512MB memory, CPU/PID limits and the same seccomp worker. The parent
removes timed-out containers. A startup self-test requires a known correct program
to pass and a filesystem probe to fail. There is no unrestricted fallback.
This is a research runner for model samples, not a hardened public code-execution
service. Candidate code and tests share a Python process, so adversarial test-harness
manipulation is outside the correctness threat model. Resource/forbidden-syscall
failures count as failed solutions under this protocol; infrastructure failures
abort the run rather than becoming preferences.

Hardware used locally: Apple M1 Pro (10 cores, 16 GiB RAM), macOS arm64 CPU, Python 3.11.7, PyTorch 2.6.0; Docker Linux
for generated code. See per-run config files. No CUDA result is claimed.

<!-- RESULTS_START -->
# Results generated from CSV logs

| Run | Kind | Metric | Observed value | Samples / training steps |
|---|---|---|---:|---:|
| [cpu-pilot](runs/cpu-pilot/metrics.csv) | dpo (16 pairs, max_steps=4) | train_loss | 0.692071 | 4 |
| [cpu-pilot](runs/cpu-pilot/metrics.csv) | dpo (16 pairs, max_steps=4) | train_runtime | 6.831600 | 4 |
| [cpu-pilot-code-prefs](runs/cpu-pilot-code-prefs/metrics.csv) | code_prefs | pairs | 2.000000 | 32 |
| [cpu-pilot-code-prefs](runs/cpu-pilot-code-prefs/metrics.csv) | code_prefs | sample_pass_rate | 0.468750 | 32 |
| [cpu-pilot-mbpp-base](runs/cpu-pilot-mbpp-base/metrics.csv) | mbpp | pass_at_1 | 0.500000 | 8 |
| [cpu-pilot-mbpp-stage-a](runs/cpu-pilot-mbpp-stage-a/metrics.csv) | mbpp | pass_at_1 | 0.500000 | 8 |
| [cpu-pilot-mbpp-stage-b](runs/cpu-pilot-mbpp-stage-b/metrics.csv) | mbpp | pass_at_1 | 0.500000 | 8 |
| [cpu-pilot-rm](runs/cpu-pilot-rm/metrics.csv) | rm | mean_rm_base | -1.005402 | 8 |
| [cpu-pilot-rm](runs/cpu-pilot-rm/metrics.csv) | rm | mean_rm_dpo | -1.005402 | 8 |
| [cpu-pilot-rm](runs/cpu-pilot-rm/metrics.csv) | rm | mean_win | 0.500000 | 8 |
| [cpu-pilot-rm](runs/cpu-pilot-rm/metrics.csv) | rm | mean_drift | 0.000135 | 8 |
| [cpu-pilot-stage-b](runs/cpu-pilot-stage-b/metrics.csv) | dpo (2 pairs, max_steps=-1) | train_loss | 0.693147 | 1 |
| [cpu-pilot-stage-b](runs/cpu-pilot-stage-b/metrics.csv) | dpo (2 pairs, max_steps=-1) | train_runtime | 7.489400 | 1 |

![DPO loss](plots/dpo_loss.png)

CPU pilot runs are integration checks, not the requested 5,000-pair Stage A result. A training loss is not evidence of preference improvement. Full 300-prompt reward-model evaluation and 257-problem MBPP results appear only after those commands complete.

Drift is the mean per-token log probability difference on greedy DPO continuations, including EOS. It is a signed drift proxy, not an unbiased KL estimator. Reward-model ties count as half wins. Scores use 512-token truncation.
<!-- RESULTS_END -->

## Limitations

A small CPU pilot validates plumbing only. An initial debug run with too-large gradient accumulation is preserved in `validation-runs/initial-partial-accumulation`; its accumulation-scaled loss is excluded from research tables. The corrected run caps accumulation to the available minibatches and is logged under `runs/cpu-pilot-stage-b`. A 5k-pair, 1-epoch GPU run and full evals
are required before claiming alignment or coding gains. MBPP can be contaminated
in pretraining. Reward-model scores are imperfect preference proxies and may
reward length/style; truncation can bias scoring. Truncated long preference
responses can weaken the training signal. One pair per mixed problem gives at
most 120 Stage B pairs; overfitting and low pair yield are plausible. Seeds do not
ensure identical results across hardware/library revisions. The optional PPO/GRPO
language-model baseline and three-seed stability comparison are not implemented.

Sources: [DPO paper](https://arxiv.org/abs/2305.18290),
[TRL 0.19.1 DPO API](https://huggingface.co/docs/trl/v0.19.1/en/dpo_trainer),
[Qwen model](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
[UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized),
[MBPP schema and license](https://huggingface.co/datasets/google-research-datasets/mbpp),
[reward-model protocol](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2).
Known model and dataset revisions are pinned in `assets.lock.json`; custom model IDs may use `ID@commit`. `requirements-macos-observed.txt` records the complete local dependency snapshot.

Code is MIT; model weights/datasets retain their respective licenses.
