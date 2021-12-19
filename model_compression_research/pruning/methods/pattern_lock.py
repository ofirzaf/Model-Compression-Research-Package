# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Apply pattern lock pruning mask on weight
"""
from collections import Iterable
from itertools import chain

from torch.nn import functional as F

from .custom_method import CustomMaskPruningMethod
from ..registry import register_method


@register_method('one_shot', name='pattern_lock')
class PatternLockPruningMethod(CustomMaskPruningMethod):
    """Pattern lock pruning method. Locks found sparsity patterns in place and allows only unpruned weights to change"""

    def _init(self, block_dims=None):
        self.block_dims = block_dims
        super()._init(self.get_sparsity_pattern_mask())

    def get_sparsity_pattern_mask(self):
        original = getattr(self.module, self.name)
        if self.block_dims is None:
            return original.ne(0.).to(original.dtype)
        if not isinstance(self.block_dims, Iterable):
            self.block_dims = original.dim() * (self.block_dims, )
        self.block_dims = tuple(self.block_dims)
        # pytorch transposes the input and output channels
        self.block_dims = (
            self.block_dims[1], self.block_dims[0]) + self.block_dims[2:]
        extended_shape = tuple(chain.from_iterable(
            [[d // b, b] for d, b in zip(original.shape, self.block_dims)]))
        pooled_shape = tuple(chain.from_iterable(
            [[d // b, 1] for d, b in zip(original.shape, self.block_dims)]))
        pooled_weight = F.avg_pool2d(
            original.unsqueeze(0), self.block_dims, self.block_dims).squeeze()
        # compute mask according new threshold and expand
        new_mask = (pooled_weight != 0.).to(original.dtype).reshape(pooled_shape).expand(
            extended_shape).reshape_as(original)
        return new_mask

    def extra_repr(self):
        s = super().extra_repr()
        if self.block_dims is not None:
            s += ', block_dims={}'.format(self.block_dims)
        return s


def lock_tensor_sparsity_pattern(module, name='weight', block_dims=None):
    """Apply pattern lock pruning to module"""
    try:
        method = module.get_pruning_parameters(
            'method', name=name).update_mask()
    except AttributeError:
        method = CustomMaskPruningMethod(module, name, block_dims=block_dims)
    return module, method
