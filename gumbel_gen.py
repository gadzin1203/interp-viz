from comet_ml import Experiment

import gin
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
VOCAB_SIZE = 50259

tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
tokenizer._add_tokens(["[BEGIN]", "[END]"])
tokenizer.pad_token = "[END]"
tokenizer.eos_token = "[END]"

### Training

def linear(step: int, batches: int, max: float = 2, min: float = .01):
    return min + (batches - step)*(max - min)/batches

def generate(model, bs, temp, vocab, device):
    partial_vocab_response = model(bs, temp)
    if t.any(t.isnan(partial_vocab_response)):
        raise Exception
    if vocab is None:
        return partial_vocab_response
    return rearrange(t.stack( # TODO this is v slow for big vocab
        [partial_vocab_response[:,:,vocab.index(i)] if i in vocab else t.zeros(bs, 24, device=device) for i in range(VOCAB_SIZE)]),
        'v b l -> b l v')

@gin.configurable
def train(
    gen_model: GumbelSoftmax,
    device: str,
    reward_fn,
    reward_fn_args,
    penalty_fn = None,
    penalty_fn_args = {},
    vocab = None,
    temp_schedule = linear,
    batch_size:int = 2,
    num_batches:int = 3,
    lr: float = 1e-2,
    logging:bool = True,
    comet_tag:str = "experiment",
):
    optimizer = t.optim.Adam(gen_model.parameters(), lr=lr)
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lambda e: 0.99995**e) # TODO not doing much

    if logging:
        experiment = Experiment(
            api_key="Khyn4qyMFwwh4ezfJpFQvydhQ",
            project_name="interp-viz",
            workspace="gadzin1203",
            auto_output_logging=False,
        )
        experiment.add_tag(comet_tag)
        experiment.log_parameters(reward_fn_args)
        experiment.log_parameters(penalty_fn_args)
        experiment.log_parameters({"bs": batch_size, "reward_fn": reward_fn.__name__, "vocab_size": len(vocab)})
        if penalty_fn is not None:
            experiment.log_parameters({"penalty_fn": penalty_fn.__name__})            
    try:
        for batch in tqdm(range(num_batches)):
            optimizer.zero_grad()
            temp = temp_schedule(batch, num_batches)

            # get model response
            response_tensor = generate(gen_model, batch_size, temp, vocab, device)
            # reward fn may return auxillary dict of metrics that we are not scoring on
            reward, reward_metrics = reward_fn(response_tensor, **reward_fn_args)

            if penalty_fn is not None:
                penalty = penalty_fn(response_tensor, **penalty_fn_args)
            else:
                penalty = 0

            # train model
            loss = -t.mean(reward) + penalty
            loss.backward()
            if t.any(t.isnan(gen_model.weight.grad)):
                raise Exception

            optimizer.step()
            if t.any(t.isnan(gen_model.weight)):
                raise Exception

            with t.no_grad():
                val_sample = generate(gen_model, batch_size, 1e-4, vocab, device)
                val_reward, _ = reward_fn(val_sample, **reward_fn_args)
                val_loss = -t.mean(val_reward)
                scheduler.step(val_loss)

            if logging:
                experiment.log_metric("reward", t.mean(reward))
                experiment.log_metric("penalty", t.mean(penalty))
                experiment.log_metric("temp", temp)
                experiment.log_metric("val_loss", t.mean(val_loss))
                experiment.log_metric("lr", optimizer.param_groups[0]['lr'])
                
                for metric in reward_metrics.items():
                    experiment.log_metric(metric[0], metric[1])
                
                if batch % 128 == 0:
                    sample_txt = tokenizer.decode(t.argmax(response_tensor[0], dim=-1))
                    experiment.log_text(sample_txt, metadata={"reward": reward[0].item()})
                    
                if batch % 1024 == 1024-1:
                    fname = f"models/{comet_tag}_{experiment.get_name()}_b{batch}"
                    t.save(gen_model, fname)
                    if batch % 2048 == 2048-1:
                        experiment.log_model(fname, fname)
    finally:
        if logging:
            experiment.end()
            
### Reward fns

def get_attn_scores(response_tensor: t.Tensor, layer: int, head: int, eval_model, gen_prompt: t.Tensor):
    response_with_begin = t.cat((gen_prompt.expand(response_tensor.shape[0], -1,-1), response_tensor), dim=1) # b, seq, vocab
    weighted_attns, normed_attns = eval_model.weighted_attention_smooth_input(response_with_begin)
    vwas = weighted_attns[layer,:,head,1:,1:] # b, response_len, response_len
    vnas = normed_attns[layer,:,head,1:,1:] # b, response_len, response_len
    return vwas, vnas

def get_metrics(vwas, vnas):
    return {"max_attn": t.max(vwas).item(), "avg_attn": t.mean(vwas).item(), "max_normed_attn": t.max(vnas).item(), "avg_normed_attn": t.mean(vnas).item()}

def head_avg(response_tensor: t.Tensor, layer: int, head: int, eval_model, gen_prompt: t.Tensor):
    vwas, vnas = get_attn_scores(response_tensor, layer, head, eval_model, gen_prompt)
    return t.mean(vwas, dim=[-1,-2]), get_metrics(vwas, vnas)

def head_normed_avg(response_tensor: t.Tensor, layer: int, head: int, eval_model, gen_prompt: t.Tensor):
    vwas, vnas = get_attn_scores(response_tensor, layer, head, eval_model, gen_prompt)
    return t.mean(vnas, dim=[-1,-2]), get_metrics(vwas, vnas)

def head_max(response_tensor: t.Tensor, layer: int, head: int, eval_model, gen_prompt: t.Tensor):
    vwas, vnas = get_attn_scores(response_tensor, layer, head, eval_model, gen_prompt)
    return t.max(t.max(vwas, dim=-1).values, dim=-1).values, get_metrics(vwas, vnas)

def head_normed_max(response_tensor: t.Tensor, layer: int, head: int, eval_model, gen_prompt: t.Tensor):
    vwas, vnas = get_attn_scores(response_tensor, layer, head, eval_model, gen_prompt)
    return t.max(t.max(vnas, dim=-1).values, dim=-1).values, get_metrics(vwas, vnas)

def is_next_five(response_tensor: t.Tensor, layer: int, head: int, eval_model, gen_prompt: t.Tensor):
    response_with_begin = t.cat((gen_prompt.expand(response_tensor.shape[0], -1,-1), response_tensor), dim=1) # b, seq, vocab
    next_logits = eval_model.forward_smooth_input(response_with_begin)[:,-1] # b, vocab
    next_probs = t.softmax(next_logits, dim=-1)
    return t.log(next_probs[:, 5]), {}

### Penalites
def entropy(probs): # blv -> l
    return -t.mean(t.sum(probs*t.log(probs), dim=-1), dim=0)

def dist_from_sentence_basic(probs, sentence): # blv, l -> l
    return t.mean(t.nn.functional.cross_entropy(
        rearrange(probs, 'b l v -> b v l'),
        repeat(sentence, 'l -> b l', b = probs.shape[0]),
        reduction="none"), dim=0)

# TODO not working
def dist_from_sentence_ref_model(probs, sentence, ref_model, ref_model_vocab_size): # blv, l -> l
    batch_sentence = repeat(sentence, 'l -> b l', b = probs.shape[0])
    sentence_logits = rearrange(ref_model(batch_sentence).logits, 'b l v -> b v l')
    probs_logits = rearrange(ref_model(probs[:,:,:ref_model_vocab_size]).logits, 'b l v -> b v l')
    return t.mean(t.nn.functional.cross_entropy(probs_logits, sentence_logits, reduction="none"), dim=0)

### Case generation

def reward_fns():
    return {"head_avg": head_avg, "head_max": head_max, "head_normed_avg": head_normed_avg, "head_normed_max": head_normed_max}

def penalty_fn():
    return {"entropy": entropy, "dist_from_sentence_basic": dist_from_sentence_basic}

def penalty_coef():
    return [0.001, 0.01, 0.1]

def layers():
    return range(2)

def heads():
    return range(8)

def allowed_vocab():
    with open("v2_tok_counts.json") as f:
        v2_dict = json.load(f)
    v2 = t.tensor([v2_dict.get(str(k),0) for k in range(VOCAB_SIZE)])
    top100_v2 = t.topk(v2, k=100).indices
    top300_v2 = t.topk(v2, k=300).indices
    top1000_v2 = t.topk(v2, k=1000).indices
    return [top100_v2, top300_v2, top1000_v2, None]

### Running

if __name__ == "__main__":
    set_start_method('spawn', force=True)

    # layers_heads = [(x,y) for x in range(2) for y in range(8)]
    layers_heads = [(1,y) for y in range(8)]
    for (layer, head), device in zip(layers_heads, DEVICES):
        reward_fn = [head_avg, head_normed_avg][head % 2]
        vocab = allowed_vocab()[0] if head < 4 else allowed_vocab()[-1]
        penalty_fn_args = {"coef": p_coef} # TODO generate cases
        head = 5 if head % 4 < 2 else 6
        
        gen_prompt = t.nn.functional.one_hot(t.tensor([[50257]], dtype=t.long, device=device), num_classes=VOCAB_SIZE)
        eval_model = get_minigpt("model.pt").to(device)
        reward_fn_args = {
            "layer": layer,
            "head": head,
            "eval_model": eval_model,
            "gen_prompt": gen_prompt
        }
        gen_model = GumbelSoftmax(24, len(vocab)).to(device)

        p = Process(
            target=train,
            args=(gen_model, device, reward_fn, reward_fn_args),
            kwargs={
                "penalty_fn": penalty_fn,
                "penalty_fn_args": penalty_fn_args,
                "vocab": vocab.tolist(),
                "num_batches": 50000,
                "batch_size": 32,
                "logging": True,
                "comet_tag": f"gumbel-L{layer}H{head}"
            },
        )
        p.start()
