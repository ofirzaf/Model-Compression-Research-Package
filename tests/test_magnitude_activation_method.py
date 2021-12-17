# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Test magnitude activation pruning method
"""

import itertools

from absl.testing import parameterized

import torch
from torch import nn

from model_compression_research import (
    UnstructuredMagnitudeActivationPruningMethod,
    get_tensor_sparsity_ratio,
)

@parameterized.parameters(*list(itertools.product(
    [.0, .5, .75],
    [True, False],
    [(1, 8), (3, 8)]
)))
class TestMagnitudeActivationPruningMethod(parameterized.TestCase):
    def test(self, target_sparsity, per_channel, input_tensor_shape):
        linear = nn.Linear(8, 4)
        method = UnstructuredMagnitudeActivationPruningMethod(
            linear,
            target_sparsity=target_sparsity,
            per_channel=per_channel,
        )
        t = torch.randn(*input_tensor_shape)
        res = linear(t)
        self.assertTrue(get_tensor_sparsity_ratio(res) == target_sparsity)
        if per_channel:
            for channel in res:
                self.assertTrue(get_tensor_sparsity_ratio(channel) == target_sparsity)


if __name__ == '__main__':
    parameterized.unittest.main()
