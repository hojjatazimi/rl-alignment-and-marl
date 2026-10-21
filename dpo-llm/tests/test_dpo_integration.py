"""Offline tiny random Qwen verifies TRL/PEFT contracts; never research data."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from datasets import Dataset
from peft import LoraConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
from trl import DPOConfig,DPOTrainer
from common.experiment import Logger,seed_all
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_dpo import CSVCallback


def test_real_dpo_step_and_csv(tmp_path):
    seed_all(7)
    tokenizer=Tokenizer(WordLevel({'<pad>':0,'<unk>':1,'<eos>':2,'prompt':3,'good':4,'bad':5},unk_token='<unk>'))
    tokenizer.pre_tokenizer=Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=tokenizer,pad_token='<pad>',eos_token='<eos>',unk_token='<unk>')
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=6,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                                     num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64))
    ds=Dataset.from_list([dict(prompt='prompt',chosen='good',rejected='bad')]*2)
    log=Logger(tmp_path/'run',{'kind':'test'})
    args=DPOConfig(output_dir=str(tmp_path/'out'),max_steps=1,per_device_train_batch_size=1,logging_steps=1,
                   report_to=[],use_cpu=True,bf16=False,fp16=False,save_strategy='no',max_length=16,max_prompt_length=8)
    trainer=DPOTrainer(model=model,args=args,processing_class=tok,train_dataset=ds,
                       peft_config=LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'),
                       callbacks=[CSVCallback(log)])
    result=trainer.train();log.close()
    assert result.global_step==1
    assert 'loss' in (tmp_path/'run'/'metrics.csv').read_text()


def test_stage_b_lineage_reconstructs_merged_stage_a(tmp_path):
    import torch
    from peft import get_peft_model
    from llm import load
    seed_all(8)
    tokenizer=Tokenizer(WordLevel({'<pad>':0,'<unk>':1,'<eos>':2,'prompt':3,'good':4,'bad':5},unk_token='<unk>'))
    tokenizer.pre_tokenizer=Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=tokenizer,pad_token='<pad>',eos_token='<eos>',unk_token='<unk>')
    base=tmp_path/'base'
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=6,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                                     num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64))
    model.save_pretrained(base);tok.save_pretrained(base)
    cfg=LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')
    a=get_peft_model(model,cfg)
    with torch.no_grad():
        for name,param in a.named_parameters():
            if 'lora_B' in name:param.fill_(.2)
    ap=tmp_path/'stage-a'/'adapter';a.save_pretrained(ap)
    merged=a.merge_and_unload()
    if hasattr(merged,'peft_config'):del merged.peft_config
    b=get_peft_model(merged,cfg).eval()
    with torch.no_grad():
        for name,param in b.named_parameters():
            if 'lora_B' in name:param.fill_(.1)
    bp=tmp_path/'stage-b'/'adapter';b.save_pretrained(bp)
    (bp.parent/'lineage.json').write_text(json.dumps(dict(base=str(base),init_adapter=str(ap))))
    restored,_=load(str(base),str(bp))
    x=torch.tensor([[3,4,5]])
    with torch.no_grad():
        torch.testing.assert_close(restored(x).logits,b(x).logits,atol=1e-6,rtol=1e-5)
