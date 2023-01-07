# Apache v2 license
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test parameter efficient fine-tuning methods"""

# import unittest
import itertools

from absl.testing import parameterized

import torch
from torch import nn

from model_compression_research import (
    IA3AddOn,
    IA3ModelConfig,
    IA3ModuleConfig,
    LoraAddOn,
    LoraModelConfig,
    LoraModuleConfig,
)


class TestIA3(parameterized.TestCase):
    @parameterized.parameters(*list(itertools.product([False, True], [False, True])))
    def test_ia3_linear(self, pre_scale, post_scale):
        l = nn.Linear(8, 4)
        original_parameters = [n for n, _ in l.named_parameters()]
        x = torch.randn(3, 8)
        module_config = IA3ModuleConfig(
            pre_scale=pre_scale, post_scale=post_scale)
        IA3AddOn.apply_to(l, module_config)
        IA3AddOn.get_from(l).freeze_parameters(excluding=True)
        for n in original_parameters:
            self.assertTrue(getattr(l, n).requires_grad == False)
        for _, p in filter(lambda np: np[0] not in original_parameters, l.named_parameters()):
            self.assertTrue(p.requires_grad == True)
        # Modify ia3 parameters from orignal initialization
        for p in IA3AddOn.get_from(l).parameters():
            with torch.no_grad():
                p.data += torch.rand_like(p)
        out = l(x)
        IA3AddOn.embed_remove_from(l, unfreeze=True)
        self.assertTrue(out.sub(l(x)).pow(2).mean() < 1e-9)
        for n in original_parameters:
            self.assertTrue(getattr(l, n).requires_grad == True)

    def test_apply_ia3(self):
        model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
        IA3AddOn.apply_to_model(model, IA3ModelConfig(
            layers={r"\d+": {"post_scale": True}}))
        for m in model:
            self.assertTrue(IA3AddOn.get_from(m) is not None)
        IA3AddOn.embed_remove_from_model(model)
        for m in model:
            self.assertTrue(IA3AddOn.get_from(m) is None)


class TestLora(parameterized.TestCase):
    @parameterized.parameters([0, 4])
    def test_lora_linear(self, rank):
        l = nn.Linear(8, 4)
        original_parameters = [n for n, _ in l.named_parameters()]
        x = torch.randn(3, 8)
        module_config = LoraModuleConfig(rank=rank)
        LoraAddOn.apply_to(l, module_config)
        LoraAddOn.get_from(l).freeze_parameters(excluding=True)
        for n in original_parameters:
            self.assertTrue(getattr(l, n).requires_grad == False)
        for _, p in filter(lambda np: np[0] not in original_parameters, l.named_parameters()):
            self.assertTrue(p.requires_grad == True)
        for p in LoraAddOn.get_from(l).parameters():
            with torch.no_grad():
                p.data += torch.rand_like(p)
        out = l(x)
        LoraAddOn.embed_remove_from(l, unfreeze=True)
        self.assertTrue(out.sub(l(x)).pow(2).mean() < 1e-9)
        for n in original_parameters:
            self.assertTrue(getattr(l, n).requires_grad == True)

    def test_apply_lora(self):
        model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
        LoraAddOn.apply_to_model(
            model, LoraModelConfig(layers={r"\d+": {"rank": 4}}))
        for m in model:
            self.assertTrue(LoraAddOn.get_from(m) is not None)
        LoraAddOn.embed_remove_from_model(model)
        for m in model:
            self.assertTrue(LoraAddOn.get_from(m) is None)


if __name__ == "__main__":
    parameterized.unittest.main()
