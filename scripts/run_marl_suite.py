"""Complete balanced 2x2 ablation grid; refuse to overwrite previous runs."""
import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=200000)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--seeds',type=int,default=5)
    p.add_argument('--out',default='marl-ppo/runs')
    p.add_argument('--wandb',action='store_true')
    a=p.parse_args()
    def run(job):
        algo,seed=job
        out=Path(a.out)/f'{algo}-seed{seed}'
        if out.exists():
            cfg=json.loads((out/'config.json').read_text())
            if (out/'complete.json').exists() and cfg['steps']==a.steps:
                return f'Existing completed {out}'
            raise RuntimeError(f'Incomplete or incompatible existing run: {out}; choose a fresh --out')
        cmd=[sys.executable,'marl-ppo/train_marl.py','--algo',algo,'--seed',str(seed),'--steps',str(a.steps),'--out',str(out),'--gif']
        if a.wandb: cmd.append('--wandb')
        console=Path(a.out)/f'{algo}-seed{seed}.console.txt'; console.parent.mkdir(parents=True,exist_ok=True)
        with console.open('w') as f: subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        return f'Completed {out}'
    jobs=[(algo,s) for algo in ['ippo','mappo','shared-local','independent-central'] for s in range(a.seeds)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
            print(future.result(),flush=True)

if __name__=='__main__': main()
