import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import gc
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from common.experiment import Logger,seed_all
from common.assets import ASSETS,hf
from llm import load,generate,continuation_logp,append_json,device


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--adapter',required=True)
    p.add_argument('--reward-model',default='OpenAssistant/reward-model-deberta-v3-large-v2')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--limit',type=int,default=300)
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--label',default='stage-a')
    p.add_argument('--out',required=True)
    p.add_argument('--wandb',action='store_true')
    a=p.parse_args(); seed_all(a.seed)
    log=Logger(a.out,dict(vars(a),kind='rm',assets=ASSETS,split='test_prefs'),wandb=a.wandb)
    ds=load_dataset('HuggingFaceH4/ultrafeedback_binarized',revision=ASSETS['HuggingFaceH4/ultrafeedback_binarized']['revision'],split='test_prefs')
    # Official train/test splits; additionally deduplicate evaluation prompts.
    seen=set(); rows=[]
    for row in ds.shuffle(seed=a.seed):
        messages=row['chosen'][:-1]
        key=str(messages)
        if key not in seen: rows.append(dict(prompt=messages)); seen.add(key)
        if len(rows)==a.limit: break
    model,tok=load(a.model)
    for row in rows:
        row['base'],_=generate(model,tok,row['prompt'],a.max_new_tokens)[0]
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    model,tok=load(a.model,a.adapter)
    for row in rows:
        row['dpo'],row['ids']=generate(model,tok,row['prompt'],a.max_new_tokens)[0]
        row['policy_logp']=continuation_logp(model,tok,row['prompt'],row['ids'])
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    model,tok=load(a.model)
    for row in rows:
        row['drift']=row['policy_logp']-continuation_logp(model,tok,row['prompt'],row['ids'])
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    rm_name,revision=hf(a.reward_model)
    rt=AutoTokenizer.from_pretrained(rm_name,**revision)
    rm=AutoModelForSequenceClassification.from_pretrained(rm_name,**revision).to(device()).eval()
    scores=[]
    for i,row in enumerate(rows):
        prompt='\n'.join(m['content'] for m in row['prompt'])
        batch=rt([prompt,prompt],[row['base'],row['dpo']],padding=True,truncation=True,max_length=512,return_tensors='pt').to(device())
        with torch.no_grad(): b,d=rm(**batch).logits.flatten().float().cpu().tolist()
        win=1. if d>b else (0.5 if d==b else 0.)
        metrics=dict(rm_base=b,rm_dpo=d,win=win,drift=row['drift'])
        log.log(i+1,**metrics); scores.append(metrics)
        append_json(Path(a.out)/'predictions.jsonl',dict(row,**metrics))
    means={key:sum(r[key] for r in scores)/len(scores) for key in scores[0]}
    log.log(len(rows),**{'mean_'+k:v for k,v in means.items()})
    print(f'Reward model evaluation ({len(rows)} prompts): {means}')
    log.close()
    (Path(a.out)/'complete.json').write_text(json.dumps(dict(samples=len(rows),**means),indent=2))

if __name__=='__main__': main()
