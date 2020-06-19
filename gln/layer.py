import torch
import torch.nn as nn


class Layer(nn.Module):
    """
    A single layer in a Gated Linear Network.
    """

    def __init__(
        self,
        num_side,
        num_inputs,
        num_outputs,
        half_spaces=4,
        epsilon=1e-4,
        weight_clip=10.0,
    ):
        self.num_side = num_side
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.half_spaces = half_spaces
        self.epsilon = epsilon
        self.weight_clip = weight_clip
        self.bias_term = 1 - epsilon

        init_gates = torch.randn(num_side, num_outputs * half_spaces)
        init_gates /= (init_gates ** 2).sum(dim=0, keepdim=True).sqrt()
        self.gates = nn.Linear(num_side, num_outputs * half_spaces)
        self.gates.weight.detach().copy_(init_gates)
        self.gates.bias.detach().copy_(torch.randn(half_spaces * num_outputs))

        self.weights = nn.Linear(num_inputs + 1, half_spaces * num_outputs, bias=False)
        self.weights.weight.detach().copy_(
            torch.randn(*self.weights.weight.shape) / num_outputs
        )

    def forward(self, x, z):
        """
        Apply the layer on top of the previous layer's outputs.

        Args:
            x: an [N x D] Tensor of probabilities from the previous layer.
            z: an [N x Z] Tensor of side information from the input.

        Returns:
            An [N x K] Tensor of output probabilities.
        """
        biases = torch.ones_like(x[:, :1]) * self.bias_term
        x = torch.cat([x, biases], dim=-1)
        y = self.weights(logit(x))
        y = y.view(-1, self.half_spaces, self.num_outputs)
        gate_values = self.gates(z).view(-1, self.half_spaces, self.num_outputs)
        gate_choices = torch.argmin(gate_values, dim=1, keepdim=True)
        y = torch.gather(y, 1, gate_choices).view(-1, self.num_outputs)
        return torch.sigmoid(y).clamp(self.epsilon, 1 - self.epsilon)


def logit(x):
    """
    Inverse of sigmoid.
    """
    return torch.log(x / (1 - x))
