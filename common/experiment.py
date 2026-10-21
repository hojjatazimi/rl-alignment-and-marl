"""Shared deterministic setup and append-only long-form metric logging."""
import csv
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


class Logger:
    def __init__(self, path, config, wandb=False, project="rl-alignment-and-marl"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.start = time.perf_counter()
        self.file = (self.path / "metrics.csv").open("w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=["step", "metric", "value", "wall_seconds"])
        self.writer.writeheader()
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        except subprocess.CalledProcessError:
            revision = None
        metadata = dict(config, python=platform.python_version(), platform=platform.platform(),
                        torch=torch.__version__, cpu=platform.processor(), git_revision=revision,
                        cuda=torch.cuda.is_available(), device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        (self.path / "config.json").write_text(json.dumps(metadata, indent=2, default=str))
        self.wb = None
        if wandb:
            import wandb as wb
            self.wb = wb.init(project=project, name=self.path.name, config=metadata)

    def log(self, step, **metrics):
        elapsed = time.perf_counter() - self.start
        for key, value in metrics.items():
            self.writer.writerow(dict(step=step, metric=key, value=float(value), wall_seconds=elapsed))
        self.file.flush()
        if self.wb:
            self.wb.log(dict(metrics, env_or_train_step=step, wall_seconds=elapsed))

    def close(self):
        self.file.close()
        if self.wb:
            self.wb.finish()
