"""Generate tables only from completed real runs; plot observed evaluation points."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_run(path):
    config=json.loads((path/'config.json').read_text())
    with (path/'metrics.csv').open() as f: rows=list(csv.DictReader(f))
    metrics={}
    for r in rows:
        metrics.setdefault(r['metric'],[]).append((int(r['step']),float(r['value'])))
    return config,metrics,float(rows[-1]['wall_seconds'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='marl-ppo/runs');p.add_argument('--out',default='marl-ppo')
    a=p.parse_args();out=Path(a.out);(out/'plots').mkdir(exist_ok=True,parents=True)
    groups={};single=[]
    for path in sorted(Path(a.runs).glob('*')):
        if not (path/'complete.json').exists(): continue
        c,m,wall=read_run(path)
        if c['kind']=='single': single.append((path,c,m,wall)); continue
        groups.setdefault((c['algo'],c['steps']),[]).append((path,c,m,wall))
    lines=['# Results generated from completed CSV logs','',
           'No smoothing or best-checkpoint selection. MARL return is the sum of per-step mean agent rewards; '
           'bands and table deviations are across seeds (sample standard deviation). '
           'Evaluation uses 20 fixed environment seeds and deterministic actions.','',
           '| Single-agent run | Seed | Steps | Final eval return ± episode std | Wall seconds |',
           '|---|---:|---:|---:|---:|']
    for path,c,m,wall in single:
        lines.append(f"| {c['env']} | {c['seed']} | {m['eval_return'][-1][0]} | {m['eval_return'][-1][1]:.2f} ± {m['eval_return_std'][-1][1]:.2f} | {wall:.1f} |")
    lines+=['','| Algorithm | Completed seeds | Joint env steps/seed | Return ± std | Coverage ± std | Collision pairs/step ± std | Wall seconds ± std |',
            '|---|---:|---:|---:|---:|---:|---:|']
    fig,ax=plt.subplots(figsize=(8,5))
    for (algo,budget),runs in sorted(groups.items()):
        # Common evaluation grid only; do not extrapolate missing seed values.
        points=set.intersection(*(set(s for s,v in m['eval_return']) for _,_,m,_ in runs))
        x=np.array(sorted(points))
        y=np.array([[dict(m['eval_return'])[int(s)] for s in x] for _,_,m,_ in runs])
        ddof=1 if len(runs)>1 else 0
        mean,std=y.mean(0),y.std(0,ddof=ddof)
        ax.plot(x,mean,label=f'{algo} (n={len(runs)}, {budget:,} steps)')
        if len(runs)>1: ax.fill_between(x,mean-std,mean+std,alpha=.18)
        fmt=lambda vals:f'{np.mean(vals):.3f} ± {np.std(vals,ddof=ddof):.3f}'
        vals=[fmt([m[k][-1][1] for _,_,m,_ in runs]) for k in ['eval_return','coverage','collisions']]
        vals.append(fmt([w for *_,w in runs]))
        lines.append(f'| {algo} | {len(runs)} | {budget} | '+ ' | '.join(vals)+' |')
    ax.set(xlabel='Joint environment transitions (3 agent actions each)',ylabel='Evaluation episode return')
    if groups: ax.legend(fontsize=8)
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'plots'/'marl_returns.png',dpi=160);plt.close(fig)
    if single:
        fig,ax=plt.subplots(figsize=(8,4))
        for _,c,m,_ in single:
            x,y=zip(*m['eval_return']);ax.plot(x,y,label=f"{c['env']} seed {c['seed']}")
        ax.set(xlabel='Environment steps',ylabel='Evaluation return');ax.legend();fig.tight_layout()
        fig.savefig(out/'plots'/'single_returns.png',dpi=160);plt.close(fig)
    lines+=['','Wall time includes training and evaluation, excludes GIF rendering. Concurrent suite runs contend for CPU; do not infer algorithm speed from this grid.',
            '', '![MARL curves](plots/marl_returns.png)','', '![Single-agent curves](plots/single_returns.png)',
            '', 'Source logs: '+', '.join(f'[{path.name}](runs/{path.name}/metrics.csv)' for path in sorted(Path(a.runs).glob('*')) if (path/'complete.json').exists())]
    result='\n'.join(lines)+'\n';(out/'results.md').write_text(result)
    readme=out/'README.md'
    if readme.exists():
        content=readme.read_text(); start='<!-- RESULTS_START -->';end='<!-- RESULTS_END -->'
        if start in content: readme.write_text(content.split(start)[0]+start+'\n'+result+end+content.split(end)[1])
    print(result)

if __name__=='__main__':main()
