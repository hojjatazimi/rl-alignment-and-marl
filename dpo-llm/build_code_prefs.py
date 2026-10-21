import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
from datasets import load_dataset
from common.experiment import Logger,seed_all
from common.assets import ASSETS
from llm import load,generate,mbpp_prompt,append_json
from sandbox import check_backend,execute,extract_code


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--adapter',required=True)
    p.add_argument('--k',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--limit',type=int,default=120)
    p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--sandbox',choices=['seccomp','docker'],default='seccomp')
    p.add_argument('--out',required=True)
    p.add_argument('--wandb',action='store_true')
    a=p.parse_args(); check_backend(a.sandbox); seed_all(a.seed)
    log=Logger(a.out,dict(vars(a),kind='code_prefs',assets=ASSETS,split='sanitized/train'),wandb=a.wandb)
    ds=load_dataset('google-research-datasets/mbpp','sanitized',revision=ASSETS['google-research-datasets/mbpp']['revision'],split='train')
    model,tok=load(a.model,a.adapter)
    count=passes=0
    for i,row in enumerate(ds.select(range(min(a.limit,len(ds))))):
        messages=mbpp_prompt(row)
        outputs=generate(model,tok,messages,a.max_new_tokens,sample=True,k=a.k)
        judged=[]
        for j,(text,_) in enumerate(outputs):
            result=execute(extract_code(text),row['test_list'],'\n'.join(row['test_imports']),a.sandbox)
            judged.append((text,result['passed'])); passes+=int(result['passed'])
            append_json(Path(a.out)/'samples.jsonl',dict(task_id=row['task_id'],sample=j,text=text,**result))
            log.log(i*a.k+j+1,passed=result['passed'],timeout=result['status']=='timeout')
        yes=[x for x,passed in judged if passed]; no=[x for x,passed in judged if not passed]
        # One pair/problem, chosen uniformly from its pass/fail groups. Avoid weighting easy problems more.
        if yes and no:
            import random
            pair=dict(prompt=messages,chosen=[{'role':'assistant','content':random.choice(yes)}],
                      rejected=[{'role':'assistant','content':random.choice(no)}],task_id=row['task_id'])
            append_json(Path(a.out)/'preferences.jsonl',pair); count+=1
        log.log((i+1)*a.k,pairs=count,sample_pass_rate=passes/((i+1)*a.k))
    print(f'Code preferences: {count} mixed-outcome pairs; {passes}/{len(ds.select(range(min(a.limit,len(ds)))))*a.k} samples passed.')
    if count==0:
        log.close(); raise SystemExit('No mixed-outcome pairs. Inspect samples; increase k or revise generation. Stage B was not trained.')
    log.close()
    (Path(a.out)/'complete.json').write_text(json.dumps(dict(samples=(i+1)*a.k,pairs=count,sample_pass_rate=passes/((i+1)*a.k)),indent=2))

if __name__=='__main__': main()
