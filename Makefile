PY ?= .venv/bin/python
WANDB ?=
STEPS ?= 200000
WORKERS ?= 1
RUN_ROOT ?= dpo-llm/runs/full
SANDBOX ?= seccomp
export HF_HOME := $(CURDIR)/.cache/huggingface
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib
export TOKENIZERS_PARALLELISM := false
export CUBLAS_WORKSPACE_CONFIG := :4096:8

.PHONY: setup test sanity dpo code-prefs stage-b eval marl report sandbox
setup:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) swig==4.3.1
	PATH="$(CURDIR)/.venv/bin:$$PATH" uv pip install --python $(PY) -r requirements.txt

test:
	$(PY) -m pytest
sanity:
	$(PY) marl-ppo/train_single.py --steps $(STEPS) --out marl-ppo/runs/cartpole-new $(WANDB)
	$(PY) marl-ppo/train_single.py --env LunarLander-v3 --steps $(STEPS) --out marl-ppo/runs/lunarlander-new $(WANDB)
dpo:
	$(PY) dpo-llm/train_dpo.py --config dpo-llm/configs/stage_a.yaml --out $(RUN_ROOT)/stage-a $(WANDB)
code-prefs:
	$(PY) dpo-llm/build_code_prefs.py --adapter $(RUN_ROOT)/stage-a/adapter --sandbox $(SANDBOX) --out $(RUN_ROOT)/code-prefs $(WANDB)
stage-b:
	$(PY) dpo-llm/train_dpo.py --config dpo-llm/configs/stage_b.yaml --prefs $(RUN_ROOT)/code-prefs/preferences.jsonl --init-adapter $(RUN_ROOT)/stage-a/adapter --out $(RUN_ROOT)/stage-b $(WANDB)
eval:
	$(PY) dpo-llm/eval_rm.py --adapter $(RUN_ROOT)/stage-a/adapter --out $(RUN_ROOT)/eval-rm $(WANDB)
	$(PY) dpo-llm/eval_mbpp.py --label base --sandbox $(SANDBOX) --out $(RUN_ROOT)/eval-base $(WANDB)
	$(PY) dpo-llm/eval_mbpp.py --label stage-a --adapter $(RUN_ROOT)/stage-a/adapter --sandbox $(SANDBOX) --out $(RUN_ROOT)/eval-stage-a $(WANDB)
	$(PY) dpo-llm/eval_mbpp.py --label stage-b --adapter $(RUN_ROOT)/stage-b/adapter --sandbox $(SANDBOX) --out $(RUN_ROOT)/eval-stage-b $(WANDB)
	$(MAKE) report
marl:
	$(PY) scripts/run_marl_suite.py --steps $(STEPS) --workers $(WORKERS) $(WANDB)
	$(MAKE) report
report:
	$(PY) marl-ppo/plot.py
	$(PY) dpo-llm/make_report.py
	$(PY) scripts/write_research_report.py
sandbox:
	docker build -t rl-code-sandbox:local -f dpo-llm/Dockerfile.sandbox dpo-llm
