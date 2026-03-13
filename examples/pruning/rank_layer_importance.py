# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/tree/aa457edc3d64d81530159cd3a182932320c78f8c

# MIT License
#
# Copyright (c) 2020 EleutherAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
from typing import List, Optional, Union
from collections import defaultdict

import torch
from torch import Tensor
import torch.nn.functional as F
import pickle

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.utils import WrappedTensor, get_model_config
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine

from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.models.nemotronh.nemotron_h_provider import NemotronHModelProvider

from transformer_engine.pytorch.module.rmsnorm import RMSNorm
from transformer_engine.pytorch.module.layernorm import LayerNorm


from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank, 
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_data_parallel_group,
    is_pipeline_last_stage,
)

import modelopt.torch.prune as mtp
import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils import get_supported_datasets, num2hrb, print_rank_0, warn_rank_0, run_forward_loop
from modelopt.torch.utils.plugins.mbridge import (
    get_hf_mbridge_calibration_loop,
    load_mbridge_model_from_hf,
)
from modelopt.torch.utils.plugins.megatron_mmlu import megatron_mmlu


kl_loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True).cuda()
mse_loss = torch.nn.MSELoss(reduce=True, reduction='mean').cuda()

def noop_mlp_forward_patch(hidden_states,):
    return (torch.zeros_like(hidden_states), None)

def noop_attn_forward_patch(hidden_states,
    attention_mask,
    context=None,
    context_mask=None,
    rotary_pos_emb=None,
    inference_params=None,
    packed_seq_params=None,):
    return torch.zeros_like(hidden_states), None


def noop_mamba_forward_patch(hidden_states, 
        attention_mask, 
        inference_context=None,
        inference_params=None, 
        rotary_pos_emb=None):
    # print('<> Mamba layer patched <>')
    return hidden_states

def noop_transformer_forward_patch(
        hidden_states,     
        attention_mask, 
        inference_context=None,
        context_mask=None, 
        rotary_pos_emb=None, 
        inference_params=None, 
        packed_seq_params=None):
        # print('<> Transformer layer patched <>')
    return hidden_states.clone(), inference_context

def noop_gpt_block_forward_patch(
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        padding_mask: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,):
    return hidden_states.clone(), inference_context

def normalized_mse_loss_per_sample(hidden_states: torch.Tensor,
                        target_hidden_states: torch.Tensor,
                        ) -> torch.Tensor:
    return torch.stack([normalized_mse_loss(hidden_states[i_sample], target_hidden_states[i_sample])
            for i_sample in range(hidden_states.shape[0])])


def normalized_mse_loss(input: torch.Tensor, target: torch.Tensor, reduction: str = 'mean',
                        epsilon: float = 1e-6) -> torch.Tensor:
    loss = (
            F.mse_loss(input, target, reduction=reduction) /
            F.mse_loss(target, torch.zeros_like(target) + epsilon, reduction=reduction)
    )
    return loss


class LastHiddenImportanceHook(torch.nn.Module):
    def __init__(self, module, name, nlast_tokens=0):
        super(LastHiddenImportanceHook, self).__init__()

        self.forward_hook = module.register_forward_hook(self.hook_fn, with_kwargs=False)
        self.pre_forward_hook = None
        self.name = name
        self.activations_stats = defaultdict(list)
        self.hidden_distance = [] #
        self.logits_distance = [] #
        self.reference_hidden = []
        self.reference_load = True
        self.lm_head = None

    def set_lm_head(self, lm_head):
        self.lm_head = lm_head

    def hook_fn(self, module, input, output):
        # seq x batch x dim
        hidden_out = output.detach().permute(1, 0, 2) # batch x seq x dim

        # if loading the reference form teacher    
        if self.reference_load:
            self.reference_hidden.append(hidden_out)
            return

        # if computing the distance to the reference    
        sample_id = len(self.hidden_distance)
        #MSE
        self.hidden_distance.append( normalized_mse_loss_per_sample(hidden_out, self.reference_hidden[sample_id]).mean() )
        # if computing the distance to the teacher's logits    
        if self.lm_head:
            teacher_logits = self.lm_head(self.reference_hidden[sample_id].permute(1, 0, 2))[0].detach()
            logits = self.lm_head(hidden_out.permute(1, 0, 2))[0].detach()
            self.logits_distance.append( normalized_mse_loss_per_sample(logits, teacher_logits).mean() )


    def load_reference(self):
        self.reference_hidden = []
        self.reference_load = True
        print_rank_0(f'> Loading reference outputs')

    def load_rankings(self):
        if self.reference_load: #the first call only swithches the accumultors
            self.reference_load = False
            return
        print_rank_0(f'> Computing distances to stored refernces')

        if len(self.hidden_distance) > 0:
            hidden_state_stats = self.gather_across_dp(torch.stack(self.hidden_distance))
            logits_stats = self.gather_across_dp(torch.stack(self.logits_distance))
        else:
            hidden_state_stats = torch.empty((0,)).cuda()
            logits_stats = torch.empty((0,)).cuda()
            
        self.activations_stats['mse'].append(hidden_state_stats)
        self.activations_stats['logits'].append(logits_stats)
        self.hidden_distance = []
        self.logits_distance = []

    def gather_across_dp(self, tensor):
        # Get the data parallel group
        dp_group = get_data_parallel_group()
        dp_world_size = get_data_parallel_world_size()

        # Create a list to hold tensors from all DP ranks
        tensor_list = [torch.empty_like(tensor) for _ in range(dp_world_size)]

        # Gather tensors from all DP ranks
        torch.distributed.all_gather(tensor_list, tensor, group=dp_group)
        return torch.cat(tensor_list, dim=0)

    def reset_stats(self):
        self.activations_stats = defaultdict(list)

    def close(self):
        self.forward_hook.remove()


def setup_gates(unwrapped_model):

    def setup_OUT_gate(unwrapped_model):
        logits_importance = torch.nn.ModuleList()
        for name, module in unwrapped_model.named_modules():
            if isinstance(module, (LayerNorm, RMSNorm)) and 'final' in name:
                logits_importance.append(LastHiddenImportanceHook(module, name))
        unwrapped_model.logits_gate_list = logits_importance

    setup_OUT_gate(unwrapped_model)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hf_model_name_or_path", type=str, required=True)
    parser.add_argument("--trust_remote_code", action="store_true")


    # Uneven Pipeline Parallelism parameters
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_layers_in_first_pipeline_stage", type=int, default=None)
    parser.add_argument("--num_layers_in_last_pipeline_stage", type=int, default=None)

    # Calibration dataset parameters
    parser.add_argument(
        "--calib_dataset_name",
        type=str,
        default="nemotron-post-training-dataset-v2",
        choices=get_supported_datasets(),
        help="Dataset name for calibration",
    )
    parser.add_argument(
        "--calib_num_samples", type=int, default=1024, help="Number of samples for calibration"
    )
    parser.add_argument("--calib_gbs", type=int, default=1, help="Calibration global batch size")
    parser.add_argument("--seq_length", type=int, default=4096)

    parser.add_argument(
        "--drop_layers",
        nargs="*",
        type=int,
        default=[],
        help=(
            "Layers to drop from the model (compute importance ranking as if these layers were already dropped)"
            "Useful for iterative pruning"
        ),
    )


    args = parser.parse_args()


    print_rank_0("\n==================== Arguments ====================")
    for k, v in args.__dict__.items():
        print_rank_0(f"{k:<35} {v}")
    print_rank_0("===================================================\n")

    return args


def collect_scores(unwrapped_model, use_metric: str="mse", aggregation: str="mean", drop_blocks: List[int]=[], drop_group: int=1):
    stats=unwrapped_model.logits_gate_list[0].activations_stats
    metrics = list(stats.keys())
    num_layers = len(stats[metrics[0]])

    scores = {}
    for i in range(num_layers):
        scores[i] = {}
        for metric in metrics:
            scores[i][metric] = stats[metric][i]
    print(f'{scores=}')
    pickle.dump(scores, open(f'scores.p', 'wb'))
    print('Layers ordered by <MSE> importance:')
    print(f'{sorted([(k,v['mse'].mean()) for k,v in scores.items() if v['mse'].numel() > 0], key=lambda x:(x[1]))=}')
                        
    return scores

def estimate_layer_importance(args: argparse.Namespace):

    pp_size = dist.size()
    print_rank_0(f"Setting pipeline_model_parallel_size to {pp_size}")

    bridge, provider, model, unwrapped_model, tokenizer = load_mbridge_model_from_hf(
        hf_model_name_or_path=args.hf_model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        provider_overrides={
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": pp_size,
            "num_layers_in_first_pipeline_stage": args.num_layers_in_first_pipeline_stage,
            "num_layers_in_last_pipeline_stage": args.num_layers_in_last_pipeline_stage,
            "pipeline_dtype": torch.bfloat16,
            "seq_length": args.seq_length,
        },
        init_model_parallel=True,
    )

    pp_rank = get_pipeline_model_parallel_rank()
    offset = pp_rank*args.num_layers_in_first_pipeline_stage if args.num_layers_in_first_pipeline_stage else 0


    print_rank_0(f"\nPruning {unwrapped_model=}")
    print_rank_0(
        f"Original model params: {num2hrb(mtp.mcore_minitron.get_mcore_param_count(unwrapped_model))}"
    )
    setup_gates(unwrapped_model)
    # set lm head in the last hidden hook
    if is_pipeline_last_stage():
        unwrapped_model.logits_gate_list[0].set_lm_head(unwrapped_model.output_layer)

    # Prepare model 
    def patch_model(layer_id, block='transformer'):
        if layer_id == -1:
            return None
        patch_register = unwrapped_model.decoder.layers[layer_id].forward
        unwrapped_model.decoder.layers[layer_id].forward = noop_gpt_block_forward_patch
        print_rank_0(f'Patched gpt block {layer_id} to noop_gpt_block_forward')
        
        return patch_register

    def unpatch_model(layer_id, patch_register, block='transformer'):
        if layer_id == -1:
            return None
        print_rank_0(f'Unpatching gpt block {layer_id} ')
        unwrapped_model.decoder.layers[layer_id].forward = patch_register

    
    def layer_id_in_this_rank(layer_id):        
        if layer_id >= offset and layer_id < offset+args.num_layers_in_first_pipeline_stage if args.num_layers_in_first_pipeline_stage else 0:
            return layer_id-offset
        else:
            return -1

    def load_reference():
        if is_pipeline_last_stage(): 
            unwrapped_model.logits_gate_list[0].load_reference()
    def load_rankings():
        if is_pipeline_last_stage():
            unwrapped_model.logits_gate_list[0].load_rankings()
    def reset_stats():
        if is_pipeline_last_stage():
            unwrapped_model.logits_gate_list[0].reset_stats()

    def reset_train_data_iterator():
        forward_loop = get_hf_mbridge_calibration_loop(
            model=model,
            provider=provider,
            tokenizer=tokenizer,
            hf_model_name_or_path=args.hf_model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            dataset_name=args.calib_dataset_name,
            num_samples=args.calib_num_samples,
            micro_batch_size=1,
            global_batch_size=args.calib_gbs,
        )
        return forward_loop

    forward_loop = reset_train_data_iterator()
    load_reference()
    forward_loop(unwrapped_model)
    
    load_rankings()    
    for layer_id in args.drop_layers:
        patch_register = patch_model(layer_id_in_this_rank(layer_id))

    reset_stats()        

    #for each block compute logits and difference to the reference    
    for layer_id in range(args.num_layers):
        # ignore blocks that are already dropped
        if layer_id in args.drop_layers:
            load_rankings()
            continue

        patch_register = patch_model(layer_id_in_this_rank(layer_id))
        forward_loop = reset_train_data_iterator()
        forward_loop(unwrapped_model)

        unpatch_model(layer_id_in_this_rank(layer_id), patch_register) #, block=block)
        load_rankings()

    if is_pipeline_last_stage() and get_data_parallel_rank() == 0:
        scores = collect_scores(unwrapped_model)
        assert scores is not None
        pickle.dump(scores, open(f'scores_{get_pipeline_model_parallel_rank()}.p', 'wb'))


if __name__ == "__main__":
    dist.setup()
    args = get_args()
    try:
        estimate_layer_importance(args)
    finally:
        dist.cleanup()
