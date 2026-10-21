import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import TrainerCallback
from trl import DPOConfig,DPOTrainer
from common.experiment import Logger,seed_all
from common.assets import ASSETS
from llm import load


class CSVCallback(TrainerCallback):
    def __init__(self,logger): self.logger=logger
    def on_log(self,args,state,control,logs=None,**kwargs):
        numeric={k:v for k,v in (logs or {}).items() if isinstance(v,(float,int))}
        self.logger.log(state.global_step,**numeric)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='dpo-llm/configs/stage_a.yaml')
    p.add_argument('--out',required=True)
    p.add_argument('--prefs',help='Stage B JSONL; prompt/chosen/rejected conversational records')
    p.add_argument('--init-adapter',help='Stage A adapter: merge before creating Stage B LoRA and reference')
    p.add_argument('--four-bit',action='store_true')
    p.add_argument('--allow-cpu',action='store_true',help='Explicit CPU pilot/integration runs only')
    p.add_argument('--wandb',action='store_true')
    a=p.parse_args(); c=yaml.safe_load(Path(a.config).read_text()); seed_all(c['seed'])
    if not torch.cuda.is_available() and not a.allow_cpu:
        raise SystemExit('CUDA GPU required for the full run. Use the Colab notebook, or --allow-cpu for an explicitly configured pilot.')
    if a.init_adapter and a.four_bit:
        raise SystemExit('Merge Stage A in float16 first; 4-bit merging is not supported here. Stage B fits without --four-bit.')
    if a.prefs:
        ds=load_dataset('json',data_files=a.prefs,split='train')
        if not len(ds): raise ValueError('No mixed pass/fail preferences; do not invent pairs.')
        ds=ds.select_columns(['prompt','chosen','rejected'])
    else:
        ds=load_dataset('HuggingFaceH4/ultrafeedback_binarized',revision=ASSETS['HuggingFaceH4/ultrafeedback_binarized']['revision'],split='train_prefs').shuffle(seed=c['seed'])
        ds=ds.select(range(min(c['pairs'],len(ds))))
        def convert(row):
            chosen,rejected=row['chosen'],row['rejected']
            assert chosen[:-1]==rejected[:-1], 'Mismatched preference prompts'
            return dict(prompt=chosen[:-1],chosen=chosen[-1:],rejected=rejected[-1:])
        ds=ds.map(convert,remove_columns=ds.column_names)
    effective_accumulation=min(c['gradient_accumulation_steps'],max(1,(len(ds)+c['batch_size']-1)//c['batch_size']))
    log=Logger(a.out,dict(c,kind='dpo',assets=ASSETS,prefs=a.prefs,init_adapter=a.init_adapter,four_bit=a.four_bit,
                        actual_pairs=len(ds),effective_accumulation=effective_accumulation),wandb=a.wandb)
    manifest=[hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest() for r in ds]
    (Path(a.out)/'data_manifest.json').write_text(json.dumps(dict(fingerprint=ds._fingerprint,sha256=manifest),indent=2))
    model,tok=load(c['model'],train=True,four_bit=a.four_bit)
    if a.init_adapter:
        model=PeftModel.from_pretrained(model,a.init_adapter).merge_and_unload()
        if hasattr(model,'peft_config'): del model.peft_config
        # Disabling the new LoRA adapter now yields the frozen Stage A reference.
    if a.four_bit: model=prepare_model_for_kbit_training(model)
    model.config.use_cache=False
    peft=LoraConfig(r=c['lora_rank'],lora_alpha=c['lora_alpha'],lora_dropout=c['lora_dropout'],
                    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM')
    cfg=DPOConfig(output_dir=str(Path(a.out)/'checkpoints'),beta=c['beta'],learning_rate=c['learning_rate'],
                  num_train_epochs=c['epochs'],per_device_train_batch_size=c['batch_size'],
                  gradient_accumulation_steps=effective_accumulation,max_length=c['max_length'],
                  max_prompt_length=c['max_prompt_length'],logging_steps=c['logging_steps'],
                  max_steps=c.get('max_steps',-1),fp16=torch.cuda.is_available(),bf16=False,
                  gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
                  seed=c['seed'],data_seed=c['seed'],report_to=[],save_strategy='no',
                  precompute_ref_log_probs=True,dataset_num_proc=1,use_cpu=not torch.cuda.is_available())
    trainer=DPOTrainer(model=model,args=cfg,train_dataset=ds,processing_class=tok,
                       peft_config=peft,callbacks=[CSVCallback(log)])
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    result=trainer.train()
    if torch.cuda.is_available():
        log.log(trainer.state.global_step,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
    log.log(trainer.state.global_step,**{k:v for k,v in result.metrics.items() if isinstance(v,(float,int))})
    trainer.save_model(str(Path(a.out)/'adapter')); tok.save_pretrained(Path(a.out)/'adapter')
    # Stage B adapters depend on a merged Stage A base. Save that lineage explicitly.
    (Path(a.out)/'lineage.json').write_text(json.dumps(dict(base=c['model'],init_adapter=a.init_adapter),indent=2))
    print(json.dumps(result.metrics,indent=2)); log.close()
    (Path(a.out)/'complete.json').write_text(json.dumps(dict(steps=trainer.state.global_step,**result.metrics),indent=2))

if __name__=='__main__': main()
