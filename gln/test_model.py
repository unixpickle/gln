import torch

from model import logit


def test_logit():
    inputs = torch.tensor([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    outputs = logit(torch.sigmoid(inputs))
    assert not torch.isnan(outputs.sum()).item()
    assert not torch.isinf(outputs.sum()).item()
    assert torch.abs(inputs - outputs).max().item() < 1e-5
