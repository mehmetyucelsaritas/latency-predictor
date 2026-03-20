# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
from torch import nn
from .interface import BaseTestCase
from nn_meter.builder.nn_modules.torch_networks.utils import get_padding


class SingleOpModel(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, inputs):
        return self.op(inputs)


class TwoOpModel(nn.Module):
    def __init__(self, op1, op2, op1_is_two_inputs, op2_is_two_inputs):
        super().__init__()
        self.op1 = op1
        self.op2 = op2
        self.op1_is_two_inputs = op1_is_two_inputs
        self.op2_is_two_inputs = op2_is_two_inputs

    def forward(self, inputs):
        if self.op1_is_two_inputs:
            x = self.op1([inputs[0], inputs[1]])
        else:
            if self.op2_is_two_inputs:
                x = self.op1(inputs[0])
            else:
                x = self.op1(inputs)
        if self.op2_is_two_inputs:
            x = self.op2([x, inputs[-1]])
        else:
            x = self.op2(x)
        return x


class MultipleOutNodes(BaseTestCase):
    name = 'MON'
    cases = {
        'case1': ['relu_relu', 'relu_dwconv', 'dwconv'],
        'case2': ['dwconv_relu_relu', 'relu_dwconv'],
        'case3': ['dwconv_relu', 'dwconv', 'relu_relu']
    }
    true_case = 'case1'
    deps = {
        'BF_dwconv_relu': True,
    }
    implement = 'torch'

    def load_config(self):
        # BaseTestCase assumes input format [H, W, C]. For torch models we need [C, H, W].
        super().load_config()
        config = self.config
        self.input_shape = [config['CIN'], config['HW'], config['HW']]

    def _make_dwconv(self):
        cin = self.input_shape[0]
        stride = int(self.config.get("STRIDES", 1))
        padding = get_padding(int(self.kernel_size), stride, int(self.input_shape[1]))
        return nn.Conv2d(
            cin,
            cin,
            kernel_size=int(self.kernel_size),
            stride=stride,
            padding=padding,
            groups=cin,
            bias=True,
        )

    def _model_block(self):
        dwconv1 = self._make_dwconv()
        dwconv2 = self._make_dwconv()
        relu = nn.ReLU()
        leaky_relu = nn.LeakyReLU(negative_slope=2)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.dwconv1 = dwconv1
                self.dwconv2 = dwconv2
                self.relu = relu
                self.leaky_relu = leaky_relu

            def forward(self, inputs):
                x = self.dwconv1(inputs)
                # branch_1: relu -> relu
                branch_1 = self.relu(self.relu(x))
                # branch_2: leaky relu(slope=2) -> dwconv
                branch_2 = self.dwconv2(self.leaky_relu(x))
                return branch_1, branch_2

        return Model(), [self.input_shape]

    def _model_relu_relu(self):
        relu = nn.ReLU()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = relu

            def forward(self, inputs):
                return self.relu(self.relu(inputs))

        return Model(), [self.input_shape]

    def _model_dwconv_relu_relu(self):
        dwconv = self._make_dwconv()
        relu = nn.ReLU()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.dwconv = dwconv
                self.relu = relu

            def forward(self, inputs):
                x = self.dwconv(inputs)
                return self.relu(self.relu(x))

        return Model(), [self.input_shape]

    def _model_relu_dwconv(self):
        dwconv = self._make_dwconv()
        relu = nn.ReLU()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = relu
                self.dwconv = dwconv

            def forward(self, inputs):
                x = self.relu(inputs)
                return self.dwconv(x)

        return Model(), [self.input_shape]

    def _model_dwconv_relu(self):
        dwconv = self._make_dwconv()
        relu = nn.ReLU()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.dwconv = dwconv
                self.relu = relu

            def forward(self, inputs):
                x = self.dwconv(inputs)
                return self.relu(x)

        return Model(), [self.input_shape]

    def _model_dwconv(self):
        dwconv = self._make_dwconv()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.dwconv = dwconv

            def forward(self, inputs):
                return self.dwconv(inputs)

        return Model(), [self.input_shape]

