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

from minigpt_utils import get_minigpt, MiniGPT
from days.utils import *

DEVICES = [f"cuda:{i}" for i in range(8)]

tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
tokenizer._add_tokens(["[BEGIN]", "[END]"])
tokenizer.pad_token = "[END]"
tokenizer.eos_token = "[END]"

def train(
    gen_model,
    ref_model,
    ppo_config,
    reward_fn,
    reward_fn_args,
    device,
    gen_len:int = 20,
    num_batches:int = 2,
    logging:bool = True,
    comet_tag:str = "experiment",
):
    ppo_trainer = PPOTrainer(gen_model, ref_model, **ppo_config)

    # encode a query
    # query_txt = "This morning I went to the "
    query_tensor = (t.tensor([tokenizer.bos_token_id], dtype=t.long, device=device)
                    .unsqueeze(0)
                    .repeat(ppo_config['batch_size'], 1))
    
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
        best_reward_so_far = 0
    try:
        batch_info = []
        for batch in tqdm(range(num_batches)):
            # get model response
            response_tensor  = respond_to_batch(gen_model, query_tensor, txt_len=gen_len)
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
                # experiment.log_metric("unnormalized reward", rewards_per_batch.mean())
                # experiment.log_metric("entropy", entropy)
                experiment.log_metric("kl", train_stats['objective/kl'])
                experiment.log_metric("kl_coef", train_stats['objective/kl_coef'])
                # experiment.log_metric("grad norm", grad_norm)
                
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
                    fname = f"evalmodel_{experiment.get_name()}_b{batch}"
                    t.save(gen_model, fname)
                    if batch % 256 == 256-1:
                        experiment.log_model(fname, fname)

        return batch_info
    finally:
        if logging:
            experiment.end()
            
def head_avg(response_tensor: t.Tensor, response_txt: List, layer: int, head: int, eval_model, gen_prompt: t.Tensor):
    response_with_begin = t.cat((gen_prompt.repeat(response_tensor.shape[0], 1), response_tensor), dim=-1)
    weighted_attns = eval_model.weighted_attention(response_with_begin)
    vwas = weighted_attns[layer,:,head,:,:]
    return t.avg(vwas, dim=[-1,-2]), {"max_attn": t.max(vwas).item(), "avg_attn": t.avg(vwas).item()}

def head_sum(response_tensor: t.Tensor, response_txt: List, layer: int, head: int, eval_model, gen_prompt: t.Tensor):
    response_with_begin = t.cat((gen_prompt.repeat(response_tensor.shape[0], 1), response_tensor), dim=-1)
    weighted_attns = eval_model.weighted_attention(response_with_begin)
    vwas = weighted_attns[layer,:,head,1:,1:] # exclude begin
    return t.max(t.max(vwas, dim=-1).values, dim=-1).values, {"avg_attn": t.avg(vwas).item(), "max_attn": t.max(vwas).item()}

if __name__ == "__main__":
    set_start_method('spawn', force=True)

    # layers_heads = [(x,y) for x in range(2) for y in range(8)]
    layers_heads = [(1,y) for y in range(8)]
    for (layer, head), device in zip(layers_heads, DEVICES):
        # janky combinatorial case generation
        adap_kl_ctrl = True if head > 3 else False
        init_kl_coef = 0.2 if head > 3 else 0.0
        head = 4 + head%4
        ppo_config = {
            "batch_size": 60,
            "forward_batch_size": 6,
            "adap_kl_ctrl": adap_kl_ctrl,
            "init_kl_coef": init_kl_coef,
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
            args=(gen_model, ref_model, ppo_config, head_max, reward_fn_args, device),
            kwargs={"gen_len": 24, "num_batches": 1024, "logging": True, "comet_tag": f"OV,{layer=},{head=}"},
            # kwargs=(gen_len=48, num_batches=256, logging=True, comet_tag=f"{layer=},{head=}"),
        )
        p.start()
