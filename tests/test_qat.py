# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Test Quantiation Aware Training module
"""
import unittest

import torch
from torch import nn
from torch import quantization as qt

from model_compression_research.quantization.qat import (
    QATLinear,
    QATMatmul,
    FakeQuantize,
)


def _quantize(input, scale=1., zero_point=0, quant_min=0, quant_max=255):
    """Do linear quantization to input according to a scale and number of bits"""
    return input.mul(1 / scale).round().add(zero_point).clamp(quant_min, quant_max)


def _dequantize(input, scale=1., zero_point=0):
    """linear dequantization according to some scale"""
    return input.sub(zero_point).mul(scale)


def _get_a_n_scale_decomposition(scale, scale_bits=16):
    n = (scale_bits - 1 - torch.log2(scale.abs())).floor()
    a = (scale.abs() * 2 ** n).round().clamp(0,
                                             calc_max_quant_value(scale_bits))
    return n, a


def calc_max_quant_value(bits):
    """Calculate the maximum symmetric quantized value according to number of bits"""
    return 2**(bits) - 1


def _requantize(input, input_scale=1., input_zero_point=0, output_scale=1., output_zero_point=0, quant_min=0, quant_max=255, scale_bits=16):
    if input_zero_point != 0:
        raise NotImplementedError(
            "Requantization is not implemented yet for assymetric input")
    scale = input_scale / output_scale
    n, a = _get_a_n_scale_decomposition(scale, scale_bits)
    out = ((input * a) >> n).round() + output_zero_point
    out = out.clamp(quant_min,
                    quant_max)
    return out


def get_scale(x, sym=False):
    max_thresh = x.max()
    min_thresh = x.min()
    if sym:
        scale = torch.max(-min_thresh, max_thresh) / 127.5
    else:
        scale = (max_thresh - min_thresh) / 255
    return scale.item()


def get_qparams(x, sym=False):
    scale = get_scale(x, sym)
    if sym:
        zp = 0
    else:
        zp = (-x.min() / scale).round().int().item()
    return scale, zp


REPEATS = 1000


class TestFakeQuantize(unittest.TestCase):
    def setUp(self):
        self.asym_fake_quant = FakeQuantize(
            observer=qt.MinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine
        )
        self.sym_fake_quant = FakeQuantize(
            observer=qt.MinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric
        )
        self.x = torch.randn(10, 10) + 1

    def test_asym_training_forward(self):
        qx = self.asym_fake_quant(self.x)
        scale = self.asym_fake_quant.scale
        zero_point = self.asym_fake_quant.zero_point
        qx_hat = _dequantize(
            _quantize(self.x, scale, zero_point), scale, zero_point)
        self.assertTrue((qx == qx_hat).all())

    def test_sym_training_forward(self):
        qx = self.sym_fake_quant(self.x)
        scale = self.sym_fake_quant.scale
        zero_point = self.sym_fake_quant.zero_point
        quant_min = self.sym_fake_quant.quant_min
        quant_max = self.sym_fake_quant.quant_max
        qx_hat = _dequantize(
            _quantize(self.x, scale, zero_point, quant_min, quant_max), scale, zero_point)
        self.assertTrue((qx == qx_hat).all())

    def test_quantization_disable(self):
        self.sym_fake_quant.disable_fake_quant()
        x_hat = self.sym_fake_quant(self.x)
        self.assertTrue((self.x == x_hat).all())
        self.sym_fake_quant.enable_fake_quant()
        x_hat = self.sym_fake_quant(self.x)
        self.assertTrue((self.x != x_hat).any())


class TestQATLinear(unittest.TestCase):
    def setUp(self):
        self.weight_asym_fake_quant = FakeQuantize.with_args(
            observer=torch.quantization.MinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine
        )
        self.x = torch.randn(4, 10)
        self.x_scale, self.x_zp = get_qparams(self.x, sym=False)
        self.fqx = torch.fake_quantize_per_tensor_affine(
            self.x, self.x_scale, self.x_zp, 0, 255)
        self.qx = torch.quantize_per_tensor(
            self.x, self.x_scale, self.x_zp, torch.quint8).int_repr().float()

    def test_training_symmetric_weights_no_requant(self):
        for _ in range(REPEATS):
            ql_sym_weight_no_requant = QATLinear(
                10, 6, bias=False, output_fake_quant=None)
            w_scale, _ = get_qparams(ql_sym_weight_no_requant.weight, sym=True)
            qw = torch.fake_quantize_per_tensor_affine(
                ql_sym_weight_no_requant.weight, w_scale, 0, -128, 127)
            self.assertTrue(
                (qw == ql_sym_weight_no_requant.quantized_weight).all())
            y = ql_sym_weight_no_requant(self.x)
            y_hat = self.fqx @ qw.t()
            self.assertTrue((y_hat - y).pow(2).mean() < 1e-6)

    def test_training_symmetric_weights_requant(self):
        for _ in range(REPEATS):
            ql_sym_weights = QATLinear(10, 6, bias=False)
            w_scale, _ = get_qparams(ql_sym_weights.weight, sym=True)
            qw = torch.fake_quantize_per_tensor_affine(
                ql_sym_weights.weight, w_scale, 0, -128, 127)
            self.assertTrue((qw == ql_sym_weights.quantized_weight).all())
            y = ql_sym_weights(self.x)
            y_hat = self.fqx @ qw.t()
            y_scale, y_zp = get_qparams(y_hat)
            y_hat = torch.fake_quantize_per_tensor_affine(
                y_hat, y_scale, y_zp, 0, 255)
            self.assertTrue((y_hat - y).pow(2).mean() < 1e-6)

    def test_training_asymmetric_weights_no_requant(self):
        for _ in range(REPEATS):
            ql_asym_weight_no_requant = QATLinear(10, 6, bias=False,
                                                        output_fake_quant=None, weight_fake_quant=self.weight_asym_fake_quant)
            w_scale, w_zp = get_qparams(ql_asym_weight_no_requant.weight)
            qw = torch.fake_quantize_per_tensor_affine(
                ql_asym_weight_no_requant.weight, w_scale, w_zp, 0, 255)
            self.assertTrue(
                (qw == ql_asym_weight_no_requant.quantized_weight).all())
            y = ql_asym_weight_no_requant(self.x)
            y_hat = self.fqx @ qw.t()
            self.assertTrue((y_hat - y).pow(2).mean() < 1e-6)

    def test_training_asymmetric_weights(self):
        for _ in range(REPEATS):
            ql_asym_weight_no_requant = QATLinear(10, 6, bias=False,
                                                        weight_fake_quant=self.weight_asym_fake_quant)
            w_scale, w_zp = get_qparams(ql_asym_weight_no_requant.weight)
            qw = torch.fake_quantize_per_tensor_affine(
                ql_asym_weight_no_requant.weight, w_scale, w_zp, 0, 255)
            self.assertTrue(
                (qw == ql_asym_weight_no_requant.quantized_weight).all())
            y = ql_asym_weight_no_requant(self.x)
            y_hat = self.fqx @ qw.t()
            y_scale, y_zp = get_qparams(y_hat)
            y_hat = torch.fake_quantize_per_tensor_affine(
                y_hat, y_scale, y_zp, 0, 255)
            self.assertTrue((y_hat - y).pow(2).mean() < 1e-6)

    def test_training_symmetric_weights_with_bias(self):
        for _ in range(REPEATS):
            ql_sym_weights = QATLinear(10, 6)
            w_scale, _ = get_qparams(ql_sym_weights.weight, sym=True)
            qw = torch.fake_quantize_per_tensor_affine(
                ql_sym_weights.weight, w_scale, 0, -128, 127)
            self.assertTrue((qw == ql_sym_weights.quantized_weight).all())
            y = ql_sym_weights(self.x)
            y_hat = self.fqx @ qw.t() + ql_sym_weights.bias
            y_scale, y_zp = get_qparams(y_hat)
            y_hat = torch.fake_quantize_per_tensor_affine(
                y_hat, y_scale, y_zp, 0, 255)
            self.assertTrue((y_hat - y).pow(2).mean() < 1e-6)

    def test_training_asymmetric_weights_with_bias(self):
        for _ in range(REPEATS):
            ql_asym_weight_no_requant = QATLinear(
                10, 6, weight_fake_quant=self.weight_asym_fake_quant)
            w_scale, w_zp = get_qparams(ql_asym_weight_no_requant.weight)
            qw = torch.fake_quantize_per_tensor_affine(
                ql_asym_weight_no_requant.weight, w_scale, w_zp, 0, 255)
            self.assertTrue(
                (qw == ql_asym_weight_no_requant.quantized_weight).all())
            y = ql_asym_weight_no_requant(self.x)
            y_hat = self.fqx @ qw.t() + ql_asym_weight_no_requant.bias
            y_scale, y_zp = get_qparams(y_hat)
            y_hat = torch.fake_quantize_per_tensor_affine(
                y_hat, y_scale, y_zp, 0, 255)
            self.assertTrue((y_hat - y).pow(2).mean() < 1e-6)

    def test_observer_not_collecting_data_when_evaluating(self):
        ql = QATLinear(10, 6)
        for _ in range(3):
            ql(torch.randn(3, 10))
        input_min_val = ql.input_fake_quant.activation_post_process.min_val.item()
        input_max_val = ql.input_fake_quant.activation_post_process.max_val.item()
        output_min_val = ql.output_fake_quant.activation_post_process.min_val.item()
        output_max_val = ql.output_fake_quant.activation_post_process.max_val.item()
        ql.eval()
        for _ in range(3):
            ql(torch.randn(3, 10))
        self.assertTrue(
            input_min_val == ql.input_fake_quant.activation_post_process.min_val.item())
        self.assertTrue(
            input_max_val == ql.input_fake_quant.activation_post_process.max_val.item())
        self.assertTrue(
            output_min_val == ql.output_fake_quant.activation_post_process.min_val.item())
        self.assertTrue(
            output_max_val == ql.output_fake_quant.activation_post_process.max_val.item())
        ql.train()
        for _ in range(3):
            ql(torch.randn(3, 10))
        self.assertFalse(
            input_min_val == ql.input_fake_quant.activation_post_process.min_val.item())
        self.assertFalse(
            input_max_val == ql.input_fake_quant.activation_post_process.max_val.item())
        self.assertFalse(
            output_min_val == ql.output_fake_quant.activation_post_process.min_val.item())
        self.assertFalse(
            output_max_val == ql.output_fake_quant.activation_post_process.max_val.item())

    def test_disable_quantization(self):
        # Training
        ql = QATLinear(10, 6)
        l = nn.Linear(10, 6)
        l.weight.data = ql.weight
        l.bias.data = ql.bias
        y_hat = ql(self.x)
        y = l(self.x)
        self.assertTrue((y != y_hat).any())
        ql.disable_fake_quant()
        ql.disable_observer()
        y_tilde = ql(self.x)
        self.assertTrue((y == y_tilde).all())
        ql.enable_fake_quant()
        y_double_hat = ql(self.x)
        self.assertTrue((y_double_hat == y_hat).all())
        # Not training
        ql.eval()
        y_hat = ql(self.x)
        self.assertTrue(((y_double_hat - y_hat).abs() < 1e-4).all())
        self.assertTrue((y != y_hat).any())
        ql.disable_fake_quant()
        y_tilde = ql(self.x)
        self.assertTrue((y == y_tilde).all())
        ql.enable_fake_quant()
        y_double_hat = ql(self.x)
        self.assertTrue((y_double_hat == y_hat).all())

    def test_delayed_start(self):
        ql = QATLinear(10, 6, start_step=2)
        l = nn.Linear(10, 6)
        l.weight.data = ql.weight
        l.bias.data = ql.bias
        y_hat = ql(self.x)
        y = l(self.x)
        self.assertTrue((y == y_hat).all())
        ql.eval()
        for _ in range(3):
            y_hat = ql(self.x)
        self.assertTrue((y == y_hat).all())
        ql.train()
        y_hat = ql(self.x)
        y_hat = ql(self.x)
        self.assertTrue((y != y_hat).any())
        ql.eval()
        y_hat = ql(self.x)
        self.assertTrue((y != y_hat).any())

    def test_from_float(self):
        l = nn.Linear(10, 6)
        ql = QATLinear.from_float(l)
        self.assertTrue((l.weight == ql.weight).all())
        self.assertTrue((l.bias == ql.bias).all())
        l.bias = None
        ql = QATLinear.from_float(l)
        self.assertTrue((l.weight == ql.weight).all())
        self.assertTrue(ql.bias is None)

    def test_saving_and_loading(self):
        ql = QATLinear(10, 6)
        state_dict = ql.state_dict()
        ql2 = QATLinear(10, 6)
        self.assertTrue((ql.weight != ql2.weight).any())
        ql2.load_state_dict(state_dict)
        self.assertTrue((ql.weight == ql2.weight).all())
        self.assertTrue((ql.bias == ql2.bias).all())


class TestQATMatmul(unittest.TestCase):
    def setUp(self):
        self.sym_fake_quantize_kwargs = {
            "zero_point": 0,
            "quant_min": -128,
            "quant_max": 127,
        }
        self.asym_fake_quantize_kwargs = {
            "quant_min": 0, 
            "quant_max": 255,
        }
        
    def test_symmetric_input_no_requant(self):
        input, other = torch.randn(3, 4), torch.randn(4, 3)
        input_scale = get_qparams(input, sym=True)[0]
        other_scale = get_qparams(other, sym=True)[0]
        qinput = torch.fake_quantize_per_tensor_affine(input, scale=input_scale, **self.sym_fake_quantize_kwargs)
        qother = torch.fake_quantize_per_tensor_affine(other, scale=other_scale, **self.sym_fake_quantize_kwargs)
        qmatmul = QATMatmul(output_fake_quant=None)
        self.assertTrue((torch.matmul(qinput, qother) - qmatmul(input, other)).pow(2).mean() < 1e-6)

    def test_symmetric_input_symmetric_requant(self):
        input, other = torch.randn(3, 4), torch.randn(4, 3)
        input_scale = get_qparams(input, sym=True)[0]
        other_scale = get_qparams(other, sym=True)[0]
        qinput = torch.fake_quantize_per_tensor_affine(input, scale=input_scale, **self.sym_fake_quantize_kwargs)
        qother = torch.fake_quantize_per_tensor_affine(other, scale=other_scale, **self.sym_fake_quantize_kwargs)
        output = torch.matmul(qinput, qother)
        output_scale = get_qparams(output, sym=True)[0]
        qoutput = torch.fake_quantize_per_tensor_affine(output, scale=output_scale, **self.sym_fake_quantize_kwargs)
        qmatmul = QATMatmul()
        self.assertTrue((qoutput - qmatmul(input, other)).pow(2).mean() < 1e-6)

    def test_observer_not_collecting_data_when_evaluating(self):
        qmatmul = QATMatmul()
        qmatmul(torch.randn(2, 2), torch.randn(2, 2))
        input_scale = qmatmul.input_fake_quant.calculate_qparams()[0].item()
        other_scale = qmatmul.other_fake_quant.calculate_qparams()[0].item()
        output_scale = qmatmul.output_fake_quant.calculate_qparams()[0].item()
        qmatmul.eval()
        qmatmul(torch.randn(2, 2), torch.randn(2, 2))
        self.assertTrue(input_scale == qmatmul.input_fake_quant.calculate_qparams()[0].item())
        self.assertTrue(other_scale == qmatmul.other_fake_quant.calculate_qparams()[0].item())
        self.assertTrue(output_scale == qmatmul.output_fake_quant.calculate_qparams()[0].item())
        qmatmul.train()
        qmatmul(torch.randn(2, 2), torch.randn(2, 2))
        self.assertTrue(input_scale != qmatmul.input_fake_quant.calculate_qparams()[0].item())
        self.assertTrue(other_scale != qmatmul.other_fake_quant.calculate_qparams()[0].item())
        self.assertTrue(output_scale != qmatmul.output_fake_quant.calculate_qparams()[0].item())


if __name__ == "__main__":
    unittest.main()
