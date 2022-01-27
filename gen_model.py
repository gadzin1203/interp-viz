from comet_ml import Experiment

import collections
import einops
import matplotlib.pyplot as plt
import math
import numpy as np
import torch as t
from torch import nn
import transformers
from IPython.core.display import HTML, display
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from typing import Tuple, List, Dict
from tqdm import tqdm
from functools import partial
import typing
from torch.multiprocessing import Process, Pool, set_start_method
import json

from ppo import PerTokenPPOTrainer
from minigpt_utils import get_minigpt, MiniGPT
from days.utils import *

DEVICES = [f"cuda:{i}" for i in range(8)]

tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
tokenizer._add_tokens(["[BEGIN]", "[END]"])
tokenizer.pad_token = "[END]"
tokenizer.eos_token = "[END]"

### Training

def mask(logits, allowed):
    logits[:,~allowed] = -t.inf

def generate(model, queries, allowed, txt_len=20, top_k=0, top_p=1.0):
    """Sample text from language model."""
    input_ids = queries
    for i in range(txt_len):
        # Get Logits
        outputs = model(input_ids)
        next_token_logits = outputs[0][:, -1, :] # start from next token
        mask(next_token_logits, allowed) # remove verboten vocab
        next_token_logits = transformers.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # Sample
        probs = t.nn.functional.softmax(next_token_logits, dim=-1) # b, vocab_size
        probs[t.isnan(probs)] = 0
        next_token = t.multinomial(probs, num_samples=1).squeeze(1)
        input_ids = t.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    return input_ids[:, -txt_len:]

def train(
    gen_model,
    ref_model,
    vocab,
    ppo_config,
    reward_fn,
    reward_fn_args,
    device,
    gen_len:int = 20,
    top_p:float = 1.0,
    num_batches:int = 2,
    logging:bool = True,
    comet_tag:str = "experiment",
):
    ppo_trainer = PPOTrainer(gen_model, ref_model, **ppo_config)

    query_tensor = (t.tensor([tokenizer.bos_token_id], dtype=t.long, device=device)
                    .unsqueeze(0)
                    .repeat(ppo_config['batch_size'], 1))

    allowed = t.isin(t.arange(tokenizer.vocab_size, device=device), vocab)
    
    if logging:
        experiment = Experiment(
            api_key="Khyn4qyMFwwh4ezfJpFQvydhQ",
            project_name="interp-viz",
            workspace="gadzin1203",
            auto_output_logging=False,
        )
        experiment.add_tag(comet_tag)
        experiment.log_parameters(ppo_trainer.ppo_params)
        experiment.log_parameters(reward_fn_args)
        experiment.log_parameters({'top_p': top_p, 'gen_len': gen_len})
        best_reward_so_far = 0
    try:
        batch_info = []
        for batch in tqdm(range(num_batches)):
            # get model response
            with t.no_grad():
                response_tensor = generate(gen_model, query_tensor, allowed, txt_len=gen_len, top_p=top_p)
            response_txt = tokenizer.batch_decode(response_tensor)

            # reward fn may return auxillary dict of metrics that we are not scoring on
            reward, reward_metrics = reward_fn(response_tensor, response_txt, **reward_fn_args)

            # train model with ppo
            train_stats = ppo_trainer.step(query_tensor, response_tensor, reward)
            batch_info.append(train_stats)

            if logging:
                experiment.log_metric("ppo_reward", train_stats['ppo/returns/mean'][0])
                experiment.log_metric("reward", t.mean(reward))
                experiment.log_metric("policy loss", train_stats['ppo/loss/policy'][0])
                experiment.log_metric("value head loss", train_stats['ppo/loss/value'][0])
                experiment.log_metric("kl", train_stats['objective/kl'])
                experiment.log_metric("kl_coef", train_stats['objective/kl_coef'])
                
                for metric in reward_metrics.items():
                    experiment.log_metric(metric[0], metric[1])
                    
                if batch > num_batches/5:
                    max_reward, max_reward_idx = t.max(reward, dim=0)
                    max_reward, max_reward_idx = max_reward.item(), max_reward_idx.item()
                    if max_reward > best_reward_so_far:
                        best_reward_so_far = max_reward
                        experiment.log_text(response_txt[max_reward_idx], metadata={"reward": max_reward})
                
                if batch % 16 == 0:
                    experiment.log_text(response_txt[0], metadata={"reward": reward[0].item()})
                    
                if batch % 128 == 128-1:
                    fname = f"models/{comet_tag}_{experiment.get_name()}_b{batch}"
                    t.save(gen_model, fname)
                    if batch % 256 == 256-1:
                        experiment.log_model(fname, fname)

        return batch_info
    finally:
        if logging:
            experiment.end()
            
### Reward fns

def head_avg(response_tensor: t.Tensor, response_txt: List, layer: int, head: int, eval_model, gen_prompt: t.Tensor):
    response_with_begin = t.cat((gen_prompt.repeat(response_tensor.shape[0], 1), response_tensor), dim=-1)
    weighted_attns = eval_model.weighted_attention(response_with_begin)
    vwas = weighted_attns[layer,:,head,:,:]
    return t.mean(vwas, dim=[-1,-2]), {"max_attn": t.max(vwas).item(), "avg_attn": t.mean(vwas).item()}

def head_minus_avg_avg(response_tensor: t.Tensor, response_txt: List, layer: int, head: int, eval_model_method, gen_prompt: t.Tensor):
    response_with_begin = t.cat((gen_prompt.repeat(response_tensor.shape[0], 1), response_tensor), dim=-1)
    weighted_attns = eval_model_method(response_with_begin)
    vwas_layer = weighted_attns[layer]
    vwas_head = vwas_layer[:,head,:,:]
    vwas_peers_avg = (t.sum(vwas_layer, dim=1) - vwas_head)/vwas_layer.shape[1] # seq, p, q
    vwas_diff = vwas_head - vwas_peers_avg
    return t.mean(vwas_diff, dim=[-1,-2]), {"max_attn": t.max(vwas_diff).item(), "avg_attn": t.mean(vwas_diff).item()}

def head_max(response_tensor: t.Tensor, response_txt: List, layer: int, head: int, eval_model_method, gen_prompt: t.Tensor):
    response_with_begin = t.cat((gen_prompt.repeat(response_tensor.shape[0], 1), response_tensor), dim=-1)
    weighted_attns = eval_model_method(response_with_begin)
    vwas = weighted_attns[layer,:,head,1:,1:] # exclude begin
    return t.max(t.max(vwas, dim=-1).values, dim=-1).values, {"avg_attn": t.mean(vwas).item(), "max_attn": t.max(vwas).item()}

def head_minus_avg_max(response_tensor: t.Tensor, response_txt: List, layer: int, head: int, eval_model_method, gen_prompt: t.Tensor):
    response_with_begin = t.cat((gen_prompt.repeat(response_tensor.shape[0], 1), response_tensor), dim=-1)
    weighted_attns = eval_model_method(response_with_begin)
    vwas_layer = weighted_attns[layer]
    vwas_head = vwas_layer[:,head,:,:]
    vwas_peers_avg = (t.sum(vwas_layer, dim=1) - vwas_head)/vwas_layer.shape[1] # seq, p, q
    vwas_diff = vwas_head - vwas_peers_avg
    return t.max(t.max(vwas_diff, dim=-1).values, dim=-1).values, {"max_attn": t.max(vwas_diff).item(), "avg_attn": t.mean(vwas_diff).item()}

def is_next_five(response_tensor: t.Tensor, response_txt: List, layer: int, head: int, eval_model, gen_prompt: t.Tensor):
    response_with_begin = t.cat((gen_prompt.repeat(response_tensor.shape[0], 1), response_tensor), dim=-1)
    with t.no_grad(): # TODO move outside
        next_logits = eval_model(response_with_begin)[:,-1] # b, vocab
    next_probs = t.softmax(next_logits, dim=-1)
    return next_probs[:, 5], {}

### Case generation

def reward_fns():
    return [head_avg, head_max, head_minus_avg_avg, head_minus_avg_max] # head_last_avg, head_last_max, head_last_fixed_avg, head_last_fixed_max, head_unweighted_avg, head_unweighted_max

def eval_model_method(model): # TODO pass in
    return [lambda x: eval_model.weighted_attention(x), lambda x: eval_model(x), lambda x: eval_model.weighted_attention_smooth_input(x)]

def adap_kl_ctrl():
    return [True, False]

def init_kl_coef():
    return [0.0, 0.02, 0.2, .5]

def topp():
    return [0.3, 0.6, 1.0]

def vf_coef():
    return [0.02] # 0.1?

def allowed_vocab():
    with open("v2_tok_counts.json") as f:
        v2_dict = json.load(f)
    v2 = t.tensor([v2_dict.get(str(k),0) for k in range(tokenizer.vocab_size)])
    top100_v2 = t.topk(v2, k=100).indices
    top300_v2 = t.topk(v2, k=300).indices
    top1000_v2 = t.topk(v2, k=1000).indices
    return [top100_v2, top300_v2, top1000_v2, t.arange(tokenizer.vocab_size)]

def trainer():
    return [PPOTrainer, PerTokenPPOTrainer]

def layers():
    return range(2)

def heads():
    return range(8)

### Running

if __name__ == "__main__":
    set_start_method('spawn', force=True)
    
    with open("v2_tok_counts.json") as f:
        v2_dict = json.load(f)
    v2 = t.tensor([v2_dict.get(str(k),0) for k in range(tokenizer.vocab_size)])
    top400_v2 = t.topk(v2, k=400).indices

    # layers_heads = [(x,y) for x in range(2) for y in range(8)]
    layers_heads = [(1,y) for y in range(1)]
    for (layer, head), device in zip(layers_heads, DEVICES):
        reward_fn = is_next_five
        head = 1
        ppo_config = {
            "batch_size": 60,
            "forward_batch_size": 6,
            "adap_kl_ctrl": False,
            "init_kl_coef": 0.0,
            "vf_coef": 0.02
        }
        
        gen_prompt = t.tensor(50257, dtype=t.long, device=device).unsqueeze(0)
        eval_model = get_minigpt("model.pt").to(device)
        reward_fn_args = {
            "layer": layer,
            "head": head,
            "eval_model": eval_model,
            "gen_prompt": gen_prompt
        }
        ref_model = GPT2HeadWithValueModel.from_pretrained("gpt2").to(device)
        gen_model = GPT2HeadWithValueModel.from_pretrained("gpt2").to(device)

        p = Process(
            target=train,
            args=(gen_model, ref_model, top400_v2.to(device), ppo_config, reward_fn, reward_fn_args, device),
            kwargs={"gen_len": 24, "num_batches": 1024, "logging": True, "comet_tag": f"test"},
        )
        p.start()
