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
    """

    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

    def forward(self, inp):
        r"""
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

    def forward(self, inp: torch.Tensor, selected_experts_log):
        r"""
        """
        original_shape = inp.shape
        # debug
        # print('In class FMoEResNetConv: original_shape: ', original_shape)
        inp = inp.reshape(original_shape[0], -1)
        (output, selected_experts_log) = super().forward(inp, selected_experts_log)
        # debug
        # print('In class FMoEResNetConv: output_shape: ', output.shape)
        return (output.reshape(original_shape), selected_experts_log)
        # return super().forward(inp)
