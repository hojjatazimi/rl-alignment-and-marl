import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from datasets import load_dataset
from common.experiment import Logger,seed_all
from common.assets import ASSETS
from llm import load,generate,mbpp_prompt,append_json
from sandbox import check_backend,execute,extract_code


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--adapter')
    p.add_argument('--label',choices=['base','stage-a','stage-b','pilot'],required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--limit',type=int,default=257)
    p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--sandbox',choices=['seccomp','docker'],default='seccomp')
    p.add_argument('--out',required=True)
    p.add_argument('--wandb',action='store_true')
    a=p.parse_args(); check_backend(a.sandbox); seed_all(a.seed)
    log=Logger(a.out,dict(vars(a),kind='mbpp',assets=ASSETS,split='sanitized/test'),wandb=a.wandb)
    ds=load_dataset('google-research-datasets/mbpp','sanitized',revision=ASSETS['google-research-datasets/mbpp']['revision'],split='test')
    model,tok=load(a.model,a.adapter); passed=0
    for i,row in enumerate(ds.select(range(min(a.limit,len(ds))))):
        text,_=generate(model,tok,mbpp_prompt(row),a.max_new_tokens)[0]
        result=execute(extract_code(text),row['test_list'],'\n'.join(row['test_imports']),a.sandbox)
        passed+=int(result['passed'])
        append_json(Path(a.out)/'predictions.jsonl',dict(task_id=row['task_id'],text=text,**result))
        log.log(i+1,passed=result['passed'],pass_at_1=passed/(i+1),timeout=result['status']=='timeout')
    print(f'{a.label} greedy MBPP pass@1: {passed}/{i+1} = {passed/(i+1):.4f}')
    log.close()
    (Path(a.out)/'complete.json').write_text(json.dumps(dict(samples=i+1,pass_at_1=passed/(i+1)),indent=2))

if __name__=='__main__': main()
