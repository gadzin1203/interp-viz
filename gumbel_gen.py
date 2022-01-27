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

from gumbel_softmax import GumbelSoftmax
from ppo import PerTokenPPOTrainer
from minigpt_utils import get_minigpt, MiniGPT
from days.utils import *

DEVICES = [f"cuda:{i}" for i in range(8)]

tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
tokenizer._add_tokens(["[BEGIN]", "[END]"])
tokenizer.pad_token = "[END]"
tokenizer.eos_token = "[END]"

### Training

def linear(step: int, batches: int, max: float = 2, min: float = .01):
    return min + (batches - step)*(max - min)/batches

def train(
    gen_model: GumbelSoftmax,
    reward_fn,
    reward_fn_args,
    temp_schedule = linear,
    batch_size:int = 2,
    num_batches:int = 3,
    lr: float = 1e-3,
    logging:bool = True,
    comet_tag:str = "experiment",
):
    optimizer = t.optim.Adam(gen_model.parameters(), lr=lr)
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    if logging:
        experiment = Experiment(
            api_key="Khyn4qyMFwwh4ezfJpFQvydhQ",
            project_name="interp-viz",
            workspace="gadzin1203",
            auto_output_logging=False,
        )
        experiment.add_tag(comet_tag)
        experiment.log_parameters(reward_fn_args)
        experiment.log_parameters({"bs": batch_size})
    try:
        for batch in tqdm(range(num_batches)):
            optimizer.zero_grad()
            temp = temp_schedule(batch, num_batches)

            # get model response
            response_tensor = gen_model(batch_size, temp)
            if t.any(t.isnan(response_tensor)):
                raise Exception
            # reward fn may return auxillary dict of metrics that we are not scoring on
            reward, reward_metrics = reward_fn(response_tensor, **reward_fn_args)

            # train model
            loss = -t.mean(t.log(reward)) # TODO only works for output reward
            loss.backward()
            if t.any(t.isnan(gen_model.weight.grad)):
                raise Exception

            optimizer.step()
            if t.any(t.isnan(gen_model.weight)):
                raise Exception

            val_reward, _ = reward_fn(gen_model(batch_size, 1e-4), **reward_fn_args)
            val_loss = -t.mean(t.log(val_reward))
            #scheduler.step(val_loss)

            if logging:
                experiment.log_metric("reward", t.mean(reward))
                experiment.log_metric("temp", temp)
                experiment.log_metric("val_loss", t.mean(val_loss))
                experiment.log_metric("lr", optimizer.param_groups[0]['lr'])
                
                for metric in reward_metrics.items():
                    experiment.log_metric(metric[0], metric[1])
                
                if batch % 16 == 0:
                    sample_txt = tokenizer.decode(t.argmax(response_tensor[0], dim=-1))
                    experiment.log_text(sample_txt, metadata={"reward": reward[0].item()})
                    
                if batch % 128 == 128-1:
                    fname = f"models/{comet_tag}_{experiment.get_name()}_b{batch}"
                    t.save(gen_model, fname)
                    if batch % 256 == 256-1:
                        experiment.log_model(fname, fname)
    finally:
        if logging:
            experiment.end()
            
### Reward fns

def head_avg(response_tensor: t.Tensor, layer: int, head: int, eval_model, gen_prompt: t.Tensor):
    response_with_begin = t.cat((gen_prompt.repeat(response_tensor.shape[0], 1), response_tensor), dim=-1)
    weighted_attns = eval_model.weighted_attention_smooth_input(response_with_begin)
    vwas = weighted_attns[layer,:,head,:,:]
    return t.mean(vwas, dim=[-1,-2]), {"max_attn": t.max(vwas).item(), "avg_attn": t.mean(vwas).item()}

def head_minus_avg_avg(response_tensor: t.Tensor, layer: int, head: int, eval_model, gen_prompt: t.Tensor):
    response_with_begin = t.cat((gen_prompt.repeat(response_tensor.shape[0], 1), response_tensor), dim=-1)
    weighted_attns = eval_model.weighted_attention_smooth_input(response_with_begin)
    vwas_layer = weighted_attns[layer]
    vwas_head = vwas_layer[:,head,:,:]
    vwas_peers_avg = (t.sum(vwas_layer, dim=1) - vwas_head)/vwas_layer.shape[1] # seq, p, q
    vwas_diff = vwas_head - vwas_peers_avg
    return t.mean(vwas_diff, dim=[-1,-2]), {"max_attn": t.max(vwas_diff).item(), "avg_attn": t.mean(vwas_diff).item()}

def head_max(response_tensor: t.Tensor, layer: int, head: int, eval_model, gen_prompt: t.Tensor):
    response_with_begin = t.cat((gen_prompt.repeat(response_tensor.shape[0], 1), response_tensor), dim=-1)
    weighted_attns = eval_model.weighted_attention_smooth_input(response_with_begin)
    vwas = weighted_attns[layer,:,head,1:,1:] # exclude begin
    return t.max(t.max(vwas, dim=-1).values, dim=-1).values, {"avg_attn": t.mean(vwas).item(), "max_attn": t.max(vwas).item()}

def head_minus_avg_max(response_tensor: t.Tensor, layer: int, head: int, eval_model, gen_prompt: t.Tensor):
    response_with_begin = t.cat((gen_prompt.repeat(response_tensor.shape[0], 1), response_tensor), dim=-1)
    weighted_attns = eval_model.weighted_attention_smooth_input(response_with_begin) # TODO grad
    vwas_layer = weighted_attns[layer]
    vwas_head = vwas_layer[:,head,:,:]
    vwas_peers_avg = (t.sum(vwas_layer, dim=1) - vwas_head)/vwas_layer.shape[1] # seq, p, q
    vwas_diff = vwas_head - vwas_peers_avg
    return t.max(t.max(vwas_diff, dim=-1).values, dim=-1).values, {"max_attn": t.max(vwas_diff).item(), "avg_attn": t.mean(vwas_diff).item()}

def is_next_five(response_tensor: t.Tensor, layer: int, head: int, eval_model, gen_prompt: t.Tensor):
    response_with_begin = t.cat((gen_prompt.expand(response_tensor.shape[0], -1,-1), response_tensor), dim=1) # b, seq, vocab
    next_logits = eval_model.forward_smooth_input(response_with_begin)[:,-1] # b, vocab
    next_probs = t.softmax(next_logits, dim=-1)
    return next_probs[:, 5], {}

### Case generation

def reward_fns():
    return [head_avg, head_max, head_minus_avg_avg, head_minus_avg_max] # head_last_avg, head_last_max, head_last_fixed_avg, head_last_fixed_max, head_unweighted_avg, head_unweighted_max

def adap_kl_ctrl():
    return [True, False]

def init_kl_coef(): # TODO should this be used?
    return [0.0, 0.02, 0.2, .5]

def vf_coef():
    return [0.02] # 0.1?

def layers():
    return range(2)

def heads():
    return range(8)

### Running

if __name__ == "__main__":
    set_start_method('spawn', force=True)

    # layers_heads = [(x,y) for x in range(2) for y in range(8)]
    layers_heads = [(1,y) for y in range(2)]
    for (layer, head), device in zip(layers_heads, DEVICES):
        reward_fn = [head_avg, head_max][head]
        head += 5
        
        gen_prompt = t.nn.functional.one_hot(t.tensor([[50257]], dtype=t.long, device=device), num_classes=50259)
        eval_model = get_minigpt("model.pt").to(device)
        reward_fn_args = {
            "layer": layer,
            "head": head,
            "eval_model": eval_model,
            "gen_prompt": gen_prompt
        }
        gen_model = GumbelSoftmax(24, 50259)

        p = Process(
            target=train,
            args=(gen_model, reward_fn, reward_fn_args, device),
            kwargs={"num_batches": 10241, "batch_size": 32, "logging": True, "comet_tag": f"gumbel-L{layer}H{head}"},
        )
        p.start()
