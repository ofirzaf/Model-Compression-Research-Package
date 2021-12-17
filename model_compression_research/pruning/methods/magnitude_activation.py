# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Magnitude activation pruning
"""

import torch
from torch import nn

from .method import ActivationPruningMethod
from .methods_utils import calc_pruning_threshold


class UnstructuredMagnitudeActivationPruningMethod(ActivationPruningMethod):
    def init_callback(self, target_sparsity=0., per_channel=False):
        self.target_sparsity = target_sparsity
        self.per_channel = per_channel
        self.relu = nn.ReLU()

    def prune_callback(self, module, inputs, outputs):
        # Implementing for now for batched vector output or single matrix output
        if not isinstance(outputs, torch.Tensor) or outputs.dim() > 2:
            raise NotImplementedError("{} doesn't support pruning of output with dimension {}, dim <= 2 supported.".format(self.__class__.__name__, outputs.dim()))
        block_size = 1
        if self.per_channel and outputs.dim() == 2:
            block_size = outputs.size(-1)
        with torch.no_grad():
            outputs_abs = outputs.abs()
            pruning_threshold = calc_pruning_threshold(
                outputs_abs,
                target_sparsity=self.target_sparsity,
                block_size=block_size,
            ).unsqueeze(-1)
            mask = outputs_abs.gt(pruning_threshold).to(outputs.dtype)
        return outputs * mask