# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Movement pruning method
based on:
https://arxiv.org/abs/2005.07683
https://arxiv.org/abs/2109.04838
"""
from collections import defaultdict
from itertools import chain

import torch
from torch import nn

from .method import PruningMethod
from .magnitude_method import UnstructuredSparsityGroup
from .methods_utils import (
    calc_current_sparsity_from_sparsity_schedule,
    calc_pruning_threshold,
    STE,
    handle_block_pruning_dims,
)
from ..registry import register_method


class GroupedBlockStructuredHardMovementPruningMethod(PruningMethod):
    GROUPS = defaultdict(UnstructuredSparsityGroup)

    def _init(self, group='default', group_target_sparsity=0., group_initial_sparsity=0., group_threshold_decay=0., block_dims=1):
        self.group_name = group
        self.group = self.get_group(self.group_name)
        self.group.add(self, group_target_sparsity,
                       group_initial_sparsity, group_threshold_decay)
        self.register_name('original')
        self.register_name('mask')
        self.register_name('scores')
        original = getattr(self.module, self.name)
        delattr(self.module, self.name)
        self.module.register_parameter(self.get_name('original'), original)
        self.module.register_buffer(self.get_name('mask'), torch.ones_like(
            original, dtype=original.dtype, device=original.device))
        if original.dim() > 2:
            raise NotImplementedError(
                "Currently works only for 2D weights, {}D weight was given".format(original.dim()))
        self.block_dims = handle_block_pruning_dims(block_dims, original.dim())
        self.expanded_shape = tuple(chain.from_iterable(
            [[d // b, b] for d, b in zip(original.shape, self.block_dims)]))
        scores_shape = tuple(chain.from_iterable(
            [[d // b, 1] for d, b in zip(original.shape, self.block_dims)]))
        # TODO: add option to decide how to initialize the scores
        self.module.register_parameter(self.get_name('scores'), nn.Parameter(
            torch.zeros(scores_shape, dtype=original.dtype, device=original.device)))

    @torch.no_grad()
    def _compute_mask(self):
        original, scores = self.get_parameters('original', 'scores')
        new_mask = scores.gt(self.group.get_threshold(self)).to(
            scores.dtype).expand(self.expanded_shape).reshape_as(original)
        self.set_parameter('mask', new_mask)

    def masked_weight(self, module):
        original, mask, scores = self.get_parameters(
            'original', 'mask', 'scores', module=module)
        return original * STE.apply(scores.expand(self.expanded_shape).reshape_as(original), mask)

    def extra_repr(self):
        s = super().extra_repr()
        s += ', group="{}", group_target_sparsity={}'.format(
            self.group_name, self.group.target_sparsity)
        if self.group.threshold_decay > 0.:
            s += ', group_threshold_decay={}'.format(
                self.group.threshold_decay)
        return s

    def _update_mask(self, sparsity_schedule=None):
        if sparsity_schedule is not None:
            self.group.update_sparsity(sparsity_schedule)

    def get_scorer(self):
        return self.get_parameters('scores')

    @classmethod
    def get_group(cls, group_name):
        """Get a sparsity group by name"""
        return cls.GROUPS[group_name]

    @classmethod
    def update_group_sparsity(cls, group_name, sparsity_schedule=None):
        """Update the target sparsity of a sparsity group"""
        group_o = cls.get_group(group_name)
        if sparsity_schedule is not None:
            group_o.update_sparsity(sparsity_schedule)
        for m in group_o.method_dict:
            if m() is not None:
                m().update_mask()


@register_method('iterative', name='global_block_structured_hard_movement')
class GlobalBlockStructuredHardMovementPruningMethod(GroupedBlockStructuredHardMovementPruningMethod):

    def _init(self, group_target_sparsity=0., group_initial_sparsity=0., group_threshold_decay=0., block_dims=1):
        super()._init(
            group='global',
            group_target_sparsity=group_target_sparsity,
            group_initial_sparsity=group_initial_sparsity,
            group_threshold_decay=group_threshold_decay,
            block_dims=block_dims,
        )

    @classmethod
    def update_group_sparsity(cls, sparsity_schedule=None):
        super().update_group_sparsity('global', sparsity_schedule=sparsity_schedule)


class GroupedUnstructuredHardMovementPruningMethod(GroupedBlockStructuredHardMovementPruningMethod):

    def _init(self, group='default', group_target_sparsity=0, group_initial_sparsity=0, group_threshold_decay=0):
        super()._init(
            group=group,
            group_target_sparsity=group_target_sparsity,
            group_initial_sparsity=group_initial_sparsity,
            group_threshold_decay=group_threshold_decay,
            block_dims=1,
        )


@register_method('iterative', name='global_unstructured_hard_movement')
class GlobalUnstructuredHardMovementPruningMethod(GroupedUnstructuredHardMovementPruningMethod):

    def _init(self, group_target_sparsity=0, group_initial_sparsity=0, group_threshold_decay=0):
        super()._init(
            group='global',
            group_target_sparsity=group_target_sparsity,
            group_initial_sparsity=group_initial_sparsity,
            group_threshold_decay=group_threshold_decay,
        )

    @classmethod
    def update_group_sparsity(cls, sparsity_schedule=None):
        super().update_group_sparsity('global', sparsity_schedule=sparsity_schedule)


@register_method('iterative', name='block_structured_hard_movement')
class BlockStructuredHardMovementPruningMethod(PruningMethod):
    def _init(self, target_sparsity=0., initial_sparsity=0., threshold_decay=0., block_dims=1):
        self.target_sparsity = target_sparsity
        self.inital_sparsity = initial_sparsity
        self.threshold_decay = threshold_decay
        self._current_sparsity = initial_sparsity
        self._threshold = 0.
        self.register_name('original')
        self.register_name('mask')
        self.register_name('scores')
        original = getattr(self.module, self.name)
        delattr(self.module, self.name)
        self.module.register_parameter(self.get_name('original'), original)
        self.module.register_buffer(self.get_name('mask'), torch.ones_like(
            original, dtype=original.dtype, device=original.device))
        if original.dim() > 2:
            raise NotImplementedError(
                "Currently works only for 2D weights, {}D weight was given".format(original.dim()))
        self.block_dims = handle_block_pruning_dims(block_dims, original.dim())
        self.expanded_shape = tuple(chain.from_iterable(
            [[d // b, b] for d, b in zip(original.shape, self.block_dims)]))
        scores_shape = tuple(chain.from_iterable(
            [[d // b, 1] for d, b in zip(original.shape, self.block_dims)]))
        # TODO: add option to decide how to initialize the scores
        self.module.register_parameter(self.get_name('scores'), nn.Parameter(
            torch.zeros(scores_shape, dtype=original.dtype, device=original.device)))

    @torch.no_grad()
    def _compute_mask(self):
        original, scores = self.get_parameters('original', 'scores')
        self._threshold = calc_pruning_threshold(
            scores, self._current_sparsity, self._threshold, self.threshold_decay)
        new_mask = scores.gt(self._threshold).to(scores.dtype).expand(
            self.expanded_shape).reshape_as(original)
        self.set_parameter('mask', new_mask)

    def masked_weight(self, module):
        original, mask, scores = self.get_parameters(
            'original', 'mask', 'scores', module=module)
        return original * STE.apply(scores.expand(self.expanded_shape).reshape_as(original), mask)

    def _update_mask(self, sparsity_schedule=None):
        # TODO this needs to be in a mixin class for iterative pruning methods with abstract method
        if sparsity_schedule is not None:
            self._current_sparsity = calc_current_sparsity_from_sparsity_schedule(
                sparsity_schedule, self.initial_sparsity, self.target_sparsity)

    def extra_repr(self):
        s = super().extra_repr()
        s += ', target_sparsity={}, block_dims={}'.format(
            self.target_sparsity, self.block_dims)
        if self.threshold_decay > 0.:
            s += ', threshold_decay={}'.format(self.threshold_decay)
        return s


@register_method('iterative', name='unstructured_hard_movement')
class UnstructuredHardMovementPruningMethod(BlockStructuredHardMovementPruningMethod):
    """Unstructured hard movement pruning method"""

    def _init(self, target_sparsity=0., initial_sparsity=0., threshold_decay=0.):
        super()._init(
            target_sparsity=target_sparsity,
            initial_sparsity=initial_sparsity,
            threshold_decay=threshold_decay,
            block_dims=1,
        )
