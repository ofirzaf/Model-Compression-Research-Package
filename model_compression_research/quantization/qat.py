# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Quantization ops
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch import quantization as tq


class FakeQuantize(tq.FakeQuantize):
    def forward(self, X):
        if not self.training:
            observer_state = self.observer_enabled[0].item()
            self.disable_observer()
            X = super().forward(X)
            self.observer_enabled[0] = observer_state
        else:
            X = super().forward(X)
        return X
    pass


default_activation_asym_fake_quant = FakeQuantize.with_args(
    observer=tq.MovingAverageMinMaxObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
)


default_activation_sym_fake_quant = FakeQuantize.with_args(
    observer=tq.MovingAverageMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric,
)


default_weight_fake_quant = FakeQuantize.with_args(
    observer=tq.MinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric,
)


class QATOutput(FakeQuantize):
    def __init__(self, output_fake_quant=None):
        requantize = output_fake_quant is not None
        if requantize:
            super().__init__(*output_fake_quant.p.args, **output_fake_quant.p.keywords)
        else:
            super().__init__()
            self.fake_quant_enabled[0] = 0
            self.observer_enabled[0] = 0
        self.register_buffer('requantize', torch.tensor(
            [requantize], dtype=torch.uint8))
        self.register_buffer(
            'quant_enabled', torch.ones(1, dtype=torch.uint8))

    def enable_fake_quant(self, enabled=True):
        self.quant_enabled[0] = 1 if enabled else 0
        if self.requantize[0]:
            super().enable_fake_quant(enabled)

    def enable_observer(self, enabled=True):
        if self.requantize[0]:
            super().enable_observer(enabled)

    def extra_repr(self):
        return 'requantize={}, '.format(self.requantize) + super().extra_repr()


class QATLinear(nn.Linear):
    """Linear layer with quantization aware training capability"""

    def __init__(self, in_features, out_features, bias=True, start_step=0,
                 weight_fake_quant=default_weight_fake_quant,
                 input_fake_quant=default_activation_asym_fake_quant,
                 output_fake_quant=default_activation_asym_fake_quant
                 ):
        super().__init__(in_features, out_features, bias)
        self.accumulation_bits = 32
        self.start_step = int(start_step)
        self.weight_fake_quant = weight_fake_quant()
        self.input_fake_quant = input_fake_quant()
        self.output_fake_quant = QATOutput(output_fake_quant)
        self.register_buffer('_step', torch.zeros(1))
        self.register_buffer('fake_quant_enabled',
                             torch.tensor([1], dtype=torch.uint8))
        if self.start_step > 0:
            self.disable_observer()
            self.disable_fake_quant()

    @classmethod
    def from_float(cls, module, start_step=0,
                   weight_fake_quant=default_weight_fake_quant,
                   input_fake_quant=default_activation_asym_fake_quant,
                   output_fake_quant=default_activation_asym_fake_quant
                   ):
        new = cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            start_step=start_step,
            weight_fake_quant=weight_fake_quant,
            input_fake_quant=input_fake_quant,
            output_fake_quant=output_fake_quant
        )
        new.weight.data = module.weight
        if module.bias is not None:
            new.bias.data = module.bias
        return new

    def forward(self, input):
        """fake quantized forward, fake quantizes weights and activations,
        learn quantization ranges if quantization mode is EMA.
        This function should only be used while training"""
        if self.training:
            if self._step == self.start_step:
                self.enable_fake_quant()
                self.enable_observer()
            self._step += 1
        out = self.output_fake_quant(
            F.linear(self.input_fake_quant(input), self.quantized_weight, self.bias))
        return out

    @property
    def quantized_weight(self):
        return self.weight_fake_quant(self.weight)

    def enable_fake_quant(self, enabled=True):
        self.fake_quant_enabled[0] = 1 if enabled else 0
        self.input_fake_quant.enable_fake_quant(enabled)
        self.weight_fake_quant.enable_fake_quant(enabled)
        self.output_fake_quant.enable_fake_quant(enabled)
        return self

    def disable_fake_quant(self):
        return self.enable_fake_quant(False)

    def enable_observer(self, enabled=True):
        self.input_fake_quant.enable_observer(enabled)
        self.weight_fake_quant.enable_observer(enabled)
        self.output_fake_quant.enable_observer(enabled)
        return self

    def disable_observer(self):
        return self.enable_observer(False)


class Matmul(nn.Module):
    def forward(self, input, other, *, out=None):
        return torch.matmul(input, other, out=out)


class QATMatmul(Matmul):
    def __init__(
        self,
        start_step=0,
        input_fake_quant=default_activation_sym_fake_quant,
        other_fake_quant=default_activation_sym_fake_quant,
        output_fake_quant=default_activation_sym_fake_quant,
    ):
        super().__init__()
        self.accumulation_bits = 32
        self.start_step = int(start_step)
        self.other_fake_quant = other_fake_quant()
        self.input_fake_quant = input_fake_quant()
        self.output_fake_quant = QATOutput(output_fake_quant)
        self.register_buffer('_step', torch.zeros(1))
        self.register_buffer('fake_quant_enabled',
                             torch.tensor([1], dtype=torch.uint8))
        if self.start_step > 0:
            self.disable_observer()
            self.disable_fake_quant()

    @classmethod
    def from_float(cls, module, start_step=0,
                   input_fake_quant=default_activation_sym_fake_quant,
                   other_fake_quant=default_activation_sym_fake_quant,
                   output_fake_quant=default_activation_sym_fake_quant,
                   ):
        new = cls(
            start_step=start_step,
            other_fake_quant=other_fake_quant,
            input_fake_quant=input_fake_quant,
            output_fake_quant=output_fake_quant
        )
        return new

    def forward(self, input, other, *, out=None):
        if self.training:
            if self._step == self.start_step:
                self.enable_fake_quant()
                self.enable_observer()
            self._step += 1
        out = self.output_fake_quant(super().forward(input=self.input_fake_quant(
            input), other=self.other_fake_quant(other), out=out))
        return out
    
    def enable_fake_quant(self, enabled=True):
        self.fake_quant_enabled[0] = 1 if enabled else 0
        self.input_fake_quant.enable_fake_quant(enabled)
        self.other_fake_quant.enable_fake_quant(enabled)
        self.output_fake_quant.enable_fake_quant(enabled)
        return self

    def disable_fake_quant(self):
        return self.enable_fake_quant(False)

    def enable_observer(self, enabled=True):
        self.input_fake_quant.enable_observer(enabled)
        self.other_fake_quant.enable_observer(enabled)
        self.output_fake_quant.enable_observer(enabled)
        return self

    def disable_observer(self):
        return self.enable_observer(False)

QUANT_MAPPING = {
    nn.Linear: QATLinear,
    Matmul: QATMatmul,
}

UNQUANT_MAPPING = {
    QATLinear: nn.Linear,
    QATMatmul: Matmul,
}
