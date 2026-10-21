"""Shared prompt, loading and scoring routines; no remote executable model code."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from common.assets import hf


def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load(model_id, adapter=None, train=False, four_bit=False):
    model_name,revision=hf(model_id)
    tok=AutoTokenizer.from_pretrained(model_name,**revision)
    tok.pad_token=tok.eos_token
    tok.truncation_side='left'
    tok.padding_side='right' if train else 'left'
    kwargs=dict(torch_dtype=torch.float16 if device()=='cuda' else torch.float32, attn_implementation='eager')
    if four_bit:
        if device()!='cuda': raise ValueError('bitsandbytes 4-bit requires CUDA')
        from transformers import BitsAndBytesConfig
        kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                      bnb_4bit_compute_dtype=torch.float16),device_map={'':0})
    model=AutoModelForCausalLM.from_pretrained(model_name,**revision,**kwargs)
    if not four_bit: model.to(device())
    if adapter:
        lineage=Path(adapter).parent/'lineage.json'
        if lineage.exists():
            parent=json.loads(lineage.read_text()).get('init_adapter')
            if parent:
                parent_path=Path(parent)
                if not parent_path.exists():
                    raise FileNotFoundError(f'Stage B requires its Stage A adapter at {parent}. Preserve the run directory layout.')
                model=PeftModel.from_pretrained(model,str(parent_path)).merge_and_unload()
                if hasattr(model,'peft_config'): del model.peft_config
        model=PeftModel.from_pretrained(model,adapter,is_trainable=train)
    if not train: model.eval()
    return model,tok


def render(tok,messages):
    return tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)


@torch.no_grad()
def generate(model,tok,messages,max_new_tokens=256,sample=False,k=1):
    x=tok(render(tok,messages),return_tensors='pt',truncation=True,max_length=1024).to(model.device)
    kwargs=dict(max_new_tokens=max_new_tokens,do_sample=sample,num_return_sequences=k,pad_token_id=tok.pad_token_id)
    if sample: kwargs.update(temperature=0.8,top_p=0.95)
    else: kwargs.update(temperature=None,top_p=None,top_k=None)
    output=model.generate(**x,**kwargs)[:,x.input_ids.shape[1]:]
    # Keep raw token IDs, including EOS, for a drift proxy on exactly the generated continuation.
    return [(tok.decode(ids,skip_special_tokens=True),ids.cpu().tolist()) for ids in output]


@torch.no_grad()
def continuation_logp(model,tok,messages,completion_ids):
    prompt=tok(render(tok,messages),return_tensors='pt',truncation=True,max_length=1024).input_ids.to(model.device)
    continuation=torch.tensor(completion_ids,device=model.device)[None]
    ids=torch.cat([prompt,continuation],-1)
    logits=model(ids,logits_to_keep=len(completion_ids)+1).logits[:,:-1].float()
    lp=logits.log_softmax(-1).gather(-1,continuation[:,:,None]).squeeze(-1)
    return float(lp.mean())


def mbpp_prompt(row):
    # Only one visible assertion establishes the required function name/signature.
    # All assertions remain in the evaluator. Identical protocol for every checkpoint.
    return [{'role':'user','content':f"Write Python code only, without explanation.\n{row['prompt']}\nExample test:\n{row['test_list'][0]}"}]


def append_json(path,row):
    with Path(path).open('a') as f: f.write(json.dumps(row)+'\n')
