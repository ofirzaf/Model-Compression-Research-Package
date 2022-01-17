# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for pruning methods
"""
from collections import Iterable

import torch


class MaskFilledSTE(torch.autograd.Function):
    """Mask filling op with estimated gradients using STE"""

    @staticmethod
    def forward(ctx, input, mask):
        """"""
        return input * mask

    @staticmethod
    def backward(ctx, grad_output):
        """Straight-Through Estimator (STE) according to"""
        return grad_output, None


class STE(torch.autograd.Function):
    """Straigh-Through Estimator, pass gradients from mask to tensor"""
    @staticmethod
    def forward(ctx, input, mask):
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def calc_pruning_threshold(tensor, target_sparsity, current_threshold=0., threshold_decay=0., block_size=1):
    """Calculate a new pruning threhsold for pruning tensor to target sparsity"""
    if tensor.dim() > 2 and block_size != 1:
        raise NotImplementedError(
            "calc_pruning_threshold not implemented yet for 3D tensors and above with block_size > 1, got {}".format(block_size))
    if block_size == 1:
        reshaped_tensor = tensor.flatten()
    else:
        reshaped_tensor = tensor.view(-1, block_size)
    k = int(target_sparsity * reshaped_tensor.shape[-1])
    try:
        threshold = reshaped_tensor.kthvalue(k, dim=-1)[0]
    except RuntimeError:
        threshold = torch.tensor([0.], device=reshaped_tensor.device)
    threshold = current_threshold * threshold_decay + \
        (1 - threshold_decay) * threshold
    return threshold


def handle_block_pruning_dims(block_dims, original_dims):
    if not isinstance(block_dims, Iterable):
        block_dims = original_dims * (block_dims, )
    block_dims = tuple(block_dims)
    if original_dims < len(block_dims):
        raise ValueError("Block number of dimensions {} can't be higher than the number of the weight's dimension {}".format(
            len(block_dims), original_dims))
    if len(block_dims) < original_dims:
        # Extend block dimensions with ones to match the number of dimensions of the pruned tensor
        block_dims = (
            original_dims - len(block_dims)) * (1, ) + block_dims
    # # pytorch transposes the input and output channels
    block_dims = (
        block_dims[1], block_dims[0]) + block_dims[2:]
    return block_dims

def calc_current_sparsity_from_sparsity_schedule(sparsity_schedule, initial_sparsity, target_sparsity):
    return initial_sparsity + (target_sparsity - initial_sparsity) * sparsity_schedule