import torch
import torch.nn as nn


class OneVsAll(nn.Module):
    """
    A one-versus-all model for discrete classification using binary GLNs.
    """

    def __init__(self, num_classes, module_fn):
        super().__init__()
        self.models = nn.ModuleList([module_fn() for _ in range(num_classes)])

    def forward(self, inputs):
        """
        Compute the binary probabilities for each class.

        Args:
            inputs: an [N x D] tensor of inputs.

        Returns:
            An [N x K] tensor of sigmoid probabilities, where K is the number
              of models.
        """
        outs = [model(model.base_predictions(inputs), inputs) for model in self.models]
        return torch.cat(outs, dim=-1)

    def forward_grad(self, inputs, targets):
        """
        Apply the models and compute their gradients.

        Args:
            inputs: an [N x D] tensor of inputs.
            targets: an [N] tensor of integer classes.

        Returns:
            An [N x K] tensor of sigmoid probabilities, where K is the number
              of models. The output is non-differentiable.
        """
        outs = [
            model.forward_grad(model.base_predictions(inputs), inputs, targets == i)
            for i, model in enumerate(self.models)
        ]
        return torch.cat(outs, dim=-1)

    def clip_weights(self):
        for model in self.models:
            model.clip_weights()
