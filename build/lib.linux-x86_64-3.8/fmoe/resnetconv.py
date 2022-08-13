r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
from numpy import dtype
import torch
import torch.nn as nn
import math
import sys
import os

basedir = os.getenv('basedir')
sys.path.append(basedir + 'fastmoe/fmoe')

from layers import FMoE
from linear import FMoELinear


class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

    def forward(self, inp):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        original_shape = inp.shape
        # debug
        # print('In _Expert.forward: original_shape: ', original_shape)

        h = int((math.sqrt(original_shape[-1] / self.num_channels)))
        inp = inp.reshape(inp.shape[0], self.num_channels, h, h)
        x = self.conv(inp)
        # debug
        # print('In _Expert.forward: shape after conv: ', x.shape)
        return x.reshape(original_shape)
        # print(inp)
        # return self.conv(inp)


class FMoEResNetConv(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        num_channels=256,
        d_model=512,
        expert_dp_comm="none",
        **kwargs
    ):
        expert = []
        # expert.append(_Expert(num_channels=num_channels))
        for i in range(num_expert):
            expert.append(_Expert(num_channels=num_channels))

        super().__init__(num_expert=num_expert, d_model=d_model, expert=expert, **kwargs)

        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        # debug
        # print('In class FMoEResNetConv: original_shape: ', original_shape)
        inp = inp.reshape(original_shape[0], -1)
        output = super().forward(inp)
        # debug
        # print('In class FMoEResNetConv: output_shape: ', output.shape)
        return output.reshape(original_shape)
        # return super().forward(inp)
