# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

import json
import os
import pickle
from typing import Dict, List, Literal, Optional, Union
from collections import defaultdict

import torch
from torch import Tensor
import torch.nn.functional as F

import modelopt.torch.utils.distributed as dist
import modelopt.torch.prune as mtp
from megatron.core.models.gpt import GPTModel
from _test_utils.torch.megatron.models import get_mcore_gpt_model
from _test_utils.torch.misc import set_seed

from megatron.core.models.gpt import GPTModel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.utils import WrappedTensor
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.core.parallel_state import (
    get_expert_model_parallel_rank, 
    get_expert_model_parallel_world_size, 
    get_pipeline_model_parallel_rank, 
    get_pipeline_model_parallel_world_size, 
    get_tensor_model_parallel_rank, 
    get_tensor_model_parallel_world_size, 
    get_data_parallel_rank,
    get_data_parallel_world_size,
    is_pipeline_last_stage,
)

from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine
from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
import torch
import copy

from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.core import mpu
# from megatron.core import parallel_state
from megatron.core.models.mamba import MambaModel
from megatron.training.checkpointing import load_checkpoint
from megatron.training import (
    get_model,
)
from pretrain_mamba import (
    forward_step,
    train_valid_test_datasets_provider
)
from megatron.training.training import (
    build_train_valid_test_data_iterators,
    update_train_iters,
    evaluate_and_print_results
)
from megatron.core.utils import (
    StragglerDetector,
    get_model_config,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    unwrap_model,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.training.initialize import initialize_megatron

from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.transformer.transformer_layer import TransformerLayer

from pretrain_mamba import model_provider as pretrain_model_provider

from transformer_engine.pytorch.module.rmsnorm import RMSNorm
from transformer_engine.pytorch.module.layernorm import LayerNorm
import debugpy

from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
)
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region,scatter_to_tensor_model_parallel_region
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_first_rank,
    get_tensor_model_parallel_rank,
    get_data_parallel_group, 
    get_data_parallel_world_size, 
    get_data_parallel_rank
)
from _test_utils.torch.megatron.utils import (
    run_mcore_inference,
    run_mcore_inference_with_dummy_input,
)

import gc

def is_first_pp_rank():
    return get_pipeline_model_parallel_first_rank() == 0

def is_first_rank():
    return torch.distributed.get_rank() == 0

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
        # tracemalloc.start()
        hidden_out = gather_from_sequence_parallel_region(output).detach().permute(1, 0, 2) # batch x seq x dim

        # if loading the reference form teacher    
        if self.reference_load:
            self.reference_hidden.append(hidden_out)
            return

        # if computing the distance to the reference    
        sample_id = len(self.hidden_distance)
        print_rank_0(f'sample: {sample_id+1}/{len(self.reference_hidden)}')
        #MSE
        self.hidden_distance.append( normalized_mse_loss_per_sample(hidden_out, self.reference_hidden[sample_id]).mean() )
        # if computing the distance to the teacher's logits    
        if self.lm_head:
            teacher_logits = gather_from_tensor_model_parallel_region(self.lm_head(self.reference_hidden[sample_id].permute(1, 0, 2))[0]).detach()
            logits = gather_from_tensor_model_parallel_region(self.lm_head(hidden_out.permute(1, 0, 2))[0]).detach()
            self.logits_distance.append( normalized_mse_loss_per_sample(logits, teacher_logits).mean() )


        # current, peak = tracemalloc.get_traced_memory()
        gc.collect()
        # print(f'{get_tensor_model_parallel_rank()=}, {current=}, {peak=}')    

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
        # self.activations_stats['logits'].append(self.logits_distance)
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


# def get_model_provider():
#     """Based on evaluation metric set the parallel-output flag and
#     return the model provider."""

#     def model_provider(pre_process=True, post_process=True) -> MambaModel:
#         args = get_args()
#         model = pretrain_model_provider(pre_process, post_process)
#         setup_gates(model, args) 
#         return model
    
#     return model_provider


def setup_gates(model):

    def setup_OUT_gate(model):
        logits_importance = torch.nn.ModuleList()
        for name, module in model.named_modules():
            if isinstance(module, (LayerNorm, RMSNorm)) and 'final' in name:
                logits_importance.append(LastHiddenImportanceHook(module, name))
        model.logits_gate_list = logits_importance

    setup_OUT_gate(model)



def get_extra_args(parser):
    """Provide extra arguments required for pruning."""
    group = parser.add_argument_group(title='pruning')

    # group.add_argument('--rank-logits', type=str.lower, choices=['transformer','attn','mlp'],
    #                        default='layer',
    #                        help='Compute logits change when dropping single block')
    group.add_argument('--drop-blocks', type=int, nargs='+', default=[],
                       help='Exclude these blocks from the model.')
    group.add_argument('--use-metric', type=str.lower, choices=['kld','mse'],
                           default='mse', help='Compute logits change when dropping single block')
    group.add_argument('--aggregation', type=str.lower, choices=['median','mean'],
                           default='mean', help='Compute logits change when dropping single block')
    group.add_argument('--drop-group', type=int,
                           default=1, help='How many least important blocks to remove in a single iteration')
    group.add_argument('--scores-file', type=str, default="importance_scores.p",
                       help='scores pickle file')
    return parser 

def collect_scores(model, use_metric: str="mse", aggregation: str="mean", drop_blocks: List[int]=[], drop_group: int=1):
    stats=model.logits_gate_list[0].activations_stats
    print(f'{stats=}')
    res=[]
    for i in range(len(stats[use_metric])):
        stat = stats[use_metric][i]
        res.append(stat) if stat.numel()>0 else res.append(torch.zeros((10,)).cuda())

    res=torch.stack(res).float()
    print(f'{res.median(dim=1)[0].sort()=}')
    print(f'{res.mean(dim=1)[0].sort()=}')
    already_dropped = len(drop_blocks)
    if aggregation == 'median':
        sorted_indices = res.median(dim=1)[0].sort()[1]
    else:
        sorted_indices = res.mean(dim=1).sort()[1]
        
    drop = sorted_indices[already_dropped:already_dropped+drop_group]
                        
    return stats, drop

def _is_cuda(tensor):
    """Check if a tensor is not none and is cuda."""
    assert tensor is not None
    assert tensor.is_cuda
def _is_cuda_contiguous(tensor):
    """Check if a tensor is not none, is cuda, and is contiguous."""
    _is_cuda(tensor)
    assert tensor.is_contiguous()
def broadcast_tensor(size, dtype, tensor=None, rank=0):
    """ Given size and type of a tensor on all ranks and the tensor value
        only on a specific rank, broadcast from that rank to all other ranks.
    """

    if torch.distributed.get_rank() == rank:
        _is_cuda_contiguous(tensor)
    else:
        tensor = torch.empty(size,
                             dtype=dtype,
                             device=torch.cuda.current_device())

    torch.distributed.broadcast(tensor, rank)

    return tensor

def estimate_depth_importance(model: GPTModel, drop_blocks: List[int]=[], scores_file: str="importance_scores.p", use_metric: str="mse"):

    config = get_model_config(model)

    # pp split
    pp_size = get_pipeline_model_parallel_world_size()
    pp_rank = get_pipeline_model_parallel_rank()
    num_layers = config.num_layers
    num_layers_per_pp = num_layers // pp_size
    offset = pp_rank*num_layers_per_pp           

    setup_gates(model)
    # set lm head in the last hidden hook
    model.logits_gate_list[0].set_lm_head(model.output_layer)

    # Prepare model 
    def patch_model(layer_id, block='transformer'):
        if layer_id == -1:
            return None
        patch_register = model.decoder.layers[layer_id].forward
        model.decoder.layers[layer_id].forward = noop_gpt_block_forward_patch
        print_rank_0(f'Patched gpt block {layer_id} to noop_gpt_block_forward')
        
        return patch_register

    def unpatch_model(layer_id, patch_register, block='transformer'):
        if layer_id == -1:
            return None
        print_rank_0(f'Unpatching gpt block {layer_id} ')
        model.decoder.layers[layer_id].forward = patch_register

    
    def layer_id_in_this_rank(layer_id):        
        if layer_id >= offset and layer_id < offset+num_layers_per_pp:
            return layer_id-offset
        else:
            return -1

    def load_reference():
        if is_pipeline_last_stage(): 
            model.logits_gate_list[0].load_reference()
    def load_rankings():
        if is_pipeline_last_stage():
            model.logits_gate_list[0].load_rankings()
    def reset_stats():
        if is_pipeline_last_stage():
            model.logits_gate_list[0].reset_stats()


    # assert args.eval_iters == 1, "eval_iters must be 1"

    # import inspect
    # for i in range(len(model[0].module.decoder.layers)):
    #     print(f'{i=}')
    #     print(f'{model[0].module.decoder.layers[i].__class__=}')
    #     print(inspect.getfullargspec(model[0].module.decoder.layers[i].forward))

    get_rerun_state_machine().mode = RerunMode.REPORT_DETERMINISM_STATS
    print_rank_0(f'{get_rerun_state_machine().get_mode()=}' )

    # args.iteration = 0
    # prefix = f'iteration {args.iteration}'
    # update_train_iters(args)    
    # train_valid_test_datasets_provider.is_distributed = True

    # print_rank_0('Building train_iterator')
    # # _, train_iterator, _ = build_train_valid_test_data_iterators(train_valid_test_datasets_provider)
    # train_iterator, _, _ = build_train_valid_test_data_iterators(train_valid_test_datasets_provider)
    # print_rank_0('done building train_iterator')

    # ###################################################
    # # WARMUP ITER
    # # Data stuff.
    load_reference()
    torch.manual_seed(SEED)
    for i in range(10):
        run_mcore_inference_with_dummy_input(model, 1)
    # evaluate_and_print_results(prefix=prefix,
    #                         forward_step_func=forward_step,
    #                         data_iterator=train_iterator,
    #                         model=model,
    #                         iteration=args.iteration,
    #                         process_non_loss_data_func=None,
    #                         config=config,
    #                         verbose=True,
    #                         write_to_tensorboard=False)
    reset_stats()
    ###################################################

    # Data stuff.
    # args.iteration = 0
    # load_reference()
    # train_iterator.rewind() if (mpu.get_tensor_model_parallel_rank() == 0 ) else None
    
    # evaluate_and_print_results(prefix=prefix,
    #                         forward_step_func=forward_step,
    #                         data_iterator=train_iterator,
    #                         model=model,
    #                         iteration=args.iteration,
    #                         process_non_loss_data_func=None,
    #                         config=config,
    #                         verbose=True,
    #                         write_to_tensorboard=False)

    load_reference()
    torch.manual_seed(SEED)
    for i in range(10):
        run_mcore_inference_with_dummy_input(model, 1)

    torch.distributed.barrier()
    print_rank_0('done reference load:-)')
    
    load_rankings()    

    #LOOP over the layers one at a time - iterative estimation    
    # Exclude blocks that will not take part in estimation
    for layer_id in drop_blocks:
        # block='mamba' if args.hybrid_override_pattern[layer_id] == 'M' else 'transformer'
        # patch_register = patch_model(layer_id_in_this_rank(layer_id), block=block)
        patch_register = patch_model(layer_id_in_this_rank(layer_id))

    reset_stats()        

    #for each block compute logits and difference to the reference    
    for layer_id in range(num_layers):
        # ignore blocks that are already dropped
        if layer_id in drop_blocks:
            load_rankings()
            model.logits_gate_list[0].activations_stats['val_loss'].append(None)
            continue

        # tell the accumulator to collect the modified logits            
        # block='mamba' if args.hybrid_override_pattern[layer_id] == 'M' else 'transformer'
        # patch_register = patch_model(layer_id_in_this_rank(layer_id), block=block)

        patch_register = patch_model(layer_id_in_this_rank(layer_id))

        torch.manual_seed(SEED)
        for i in range(10):
            run_mcore_inference_with_dummy_input(model, 1)

        # train_iterator.rewind() if mpu.get_tensor_model_parallel_rank() == 0 else None
        # val_loss = evaluate_and_print_results(prefix=prefix,
        #                     forward_step_func=forward_step,
        #                     data_iterator=train_iterator,
        #                     model=model,
        #                     iteration=args.iteration,
        #                     process_non_loss_data_func=None,
        #                     config=config,
        #                     verbose=True,
        #                     write_to_tensorboard=False)
    
        model.logits_gate_list[0].activations_stats['val_loss'].append([None])

        unpatch_model(layer_id_in_this_rank(layer_id), patch_register) #, block=block)
        load_rankings()
        # print(f'done {layer_id=} {val_loss=} {model[0].module.logits_gate_list[0].activations_stats['mse'][-1]=}')
        # print_rank_0('done :-)')

    # set the final stats collection       
    # drop = torch.zeros((args.drop_group,), dtype=torch.int64, device='cuda')
    # drop = None
    if is_pipeline_last_stage() and get_tensor_model_parallel_rank() == 0 and get_data_parallel_rank() == 0:
        scores, drop = collect_scores(model)
        scores['mse_drop'] = drop
        assert scores is not None
        pickle.dump(scores, open(f'{scores_file}_{use_metric}_DROP_{len(drop_blocks)}_blocks_{get_data_parallel_rank()}.p', 'wb'))
        print(f'Rankings from DP{get_data_parallel_rank()} dumped to file: {scores_file}_{use_metric}_DROP_{len(drop_blocks)}_blocks_{get_data_parallel_rank()}.p')
        print(drop)

def main():
    """Main program."""
    dist.setup()
    set_seed(SEED)
    
    # Create a small GPT model for testing
    num_layers = 4
    hidden_size = 64
    num_attention_heads = 4
    ffn_hidden_size = 128
    max_sequence_length = 16
    vocab_size = 32
    batch_size = 1
    
    model = get_mcore_gpt_model(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        initialize_megatron=True,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        ffn_hidden_size=ffn_hidden_size,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        activation_func="swiglu",
        bf16=False,
    ).cuda()
    
    model.eval()
    
    estimate_depth_importance(model)
    dist.cleanup()

SEED = 1234

if __name__ == '__main__':
    main()
