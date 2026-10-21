import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='dpo-llm/runs');p.add_argument('--out',default='dpo-llm')
    a=p.parse_args();out=Path(a.out);(out/'plots').mkdir(parents=True,exist_ok=True)
    lines=['# Results generated from CSV logs','',
           '| Run | Kind | Metric | Observed value | Samples / training steps |',
           '|---|---|---|---:|---:|']
    found=False;curves=[]
    for logfile in sorted(Path(a.runs).rglob('metrics.csv')):
        path=logfile.parent
        if not (path/'complete.json').exists():continue
        cfg=json.loads((path/'config.json').read_text())
        with (path/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
        if not rows:continue
        latest={r['metric']:r for r in rows}
        kind=cfg['kind']
        keys={'dpo':['train_loss','train_runtime'], 'rm':['mean_rm_base','mean_rm_dpo','mean_win','mean_drift'],
              'mbpp':['pass_at_1'],'code_prefs':['pairs','sample_pass_rate']}.get(kind,[])
        for k in keys:
            if k not in latest:continue
            r=latest[k];found=True
            label=kind
            if kind=='dpo': label+=f" ({cfg.get('actual_pairs',cfg.get('pairs','unknown'))} pairs, max_steps={cfg.get('max_steps')})"
            lines.append(f"| [{path.relative_to(Path(a.runs))}](runs/{path.relative_to(Path(a.runs))}/metrics.csv) | {label} | {k} | {float(r['value']):.6f} | {r['step']} |")
        points=[(int(r['step']),float(r['value'])) for r in rows if r['metric']=='loss']
        if points:curves.append((path.name,points))
    if not found:lines+=['','No completed result metrics found. Training/evaluations remain unmeasured.']
    if curves:
        fig,ax=plt.subplots(figsize=(7,4))
        for label,pts in curves:
            x,y=zip(*pts);ax.plot(x,y,marker='.',label=label)
        ax.set(xlabel='Optimizer steps',ylabel='DPO loss');ax.legend();fig.tight_layout()
        fig.savefig(out/'plots'/'dpo_loss.png',dpi=160);plt.close(fig)
        lines+=['','![DPO loss](plots/dpo_loss.png)']
    lines+=['','CPU pilot runs are integration checks, not the requested 5,000-pair Stage A result. '
            'A training loss is not evidence of preference improvement. '
            'Full 300-prompt reward-model evaluation and 257-problem MBPP results appear only after those commands complete.',
            '', 'Drift is the mean per-token log probability difference on greedy DPO continuations, '
            'including EOS. It is a signed drift proxy, not an unbiased KL estimator. '
            'Reward-model ties count as half wins. Scores use 512-token truncation.']
    result='\n'.join(lines)+'\n';(out/'results.md').write_text(result)
    readme=out/'README.md'
    if readme.exists():
        s=readme.read_text();start='<!-- RESULTS_START -->';end='<!-- RESULTS_END -->'
        if start in s:readme.write_text(s.split(start)[0]+start+'\n'+result+end+s.split(end)[1])
    print(result)

if __name__=='__main__':main()
