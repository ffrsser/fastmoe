r"""
Noisy gate for gshard and switch
"""
from .base_gate import BaseGate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math


class NoisyGate(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.d_model = torch.prod(torch.Tensor(d_model)).int()

        self.w_gate = nn.Parameter(
            torch.zeros(self.d_model, self.tot_expert), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(self.d_model, self.tot_expert), requires_grad=True
        )
        self.top_k = top_k
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        self.noise_epsilon = 1e-2

        self.reset_parameters()

    def reset_parameters(self):
        # Approach is the same as in torch.nn.Linear
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L88

        torch.nn.init.kaiming_uniform_(self.w_gate, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w_noise, a=math.sqrt(5))


    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.top_k
        )
        #! [batch]
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        #! [batch]
        is_in = torch.gt(noisy_values, threshold_if_in)
        #debug
        # print('noisy_values.shape', noisy_values.shape)
        # print('threshold_if_in.shape', threshold_if_in.shape)
        # print('is_in.shape', is_in.shape)

        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(
            torch.tensor([0.0], device=clean_values.device),
            torch.tensor([1.0], device=clean_values.device),
            validate_args=False,
            #! Compatibility issue between pytorch 1.7 and 1.8.
        )
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
        #?

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_expert = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, inp):
        inp = inp.view(-1, self.d_model)
        clean_logits = inp @ self.w_gate
        raw_noise_stddev = inp @ self.w_noise
        noise_stddev = (
            self.softplus(raw_noise_stddev) + self.noise_epsilon
        ) * self.training
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        logits = noisy_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.tot_expert), dim=1
        )
        top_k_logits = top_logits[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.top_k < self.tot_expert:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)

        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        self.set_loss(loss)

        # return (
        #     top_k_indices.contiguous().view(-1),
        #     top_k_gates.contiguous().unsqueeze(1),
        # )
        return (
            top_k_indices.contiguous().reshape(-1, self.top_k),
            top_k_gates.contiguous().reshape(-1, self.top_k),
        )

class NoisyConvGate(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        in_channels, height, width = d_model
        out_channels = in_channels * 2
        #debug
        # print('in_channels ...', in_channels, height, width)

        self.d_model = in_channels * height * width
        self.w_gate = nn.Parameter(
            torch.zeros(out_channels, self.tot_expert), requires_grad=True
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

        self.w_noise = nn.Parameter(
            torch.zeros(self.d_model, self.tot_expert), requires_grad=True
        )
        self.top_k = top_k
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        self.noise_epsilon = 1e-2

        self.reset_parameters()

    def reset_parameters(self):
        # Approach is the same as in torch.nn.Linear
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L88

        torch.nn.init.kaiming_uniform_(self.w_gate, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w_noise, a=math.sqrt(5))


    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.top_k
        )
        #! [batch]
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        #! [batch]
        is_in = torch.gt(noisy_values, threshold_if_in)
        #debug
        # print('noisy_values.shape', noisy_values.shape)
        # print('threshold_if_in.shape', threshold_if_in.shape)
        # print('is_in.shape', is_in.shape)

        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(
            torch.tensor([0.0], device=clean_values.device),
            torch.tensor([1.0], device=clean_values.device),
            validate_args=False,
            #! Compatibility issue between pytorch 1.7 and 1.8.
        )
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
        #?

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_expert = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, inp):
        clean_logits = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(inp))))) + self.shortcut(inp))
        clean_logits = F.avg_pool2d(clean_logits, clean_logits.size()[3])
        clean_logits = torch.squeeze(clean_logits)
        clean_logits = clean_logits @ self.w_gate

        #debug
        # print('clean_logits', clean_logits.shape)

        raw_noise_stddev = inp.view(-1, self.d_model) @ self.w_noise
        noise_stddev = (
            self.softplus(raw_noise_stddev) + self.noise_epsilon
        ) * self.training
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        logits = noisy_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.tot_expert), dim=1
        )
        top_k_logits = top_logits[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.top_k < self.tot_expert:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)

        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        self.set_loss(loss)

        # return (
        #     top_k_indices.contiguous().view(-1),
        #     top_k_gates.contiguous().unsqueeze(1),
        # )
        return (
            top_k_indices.contiguous().reshape(-1, self.top_k),
            top_k_gates.contiguous().reshape(-1, self.top_k),
        )