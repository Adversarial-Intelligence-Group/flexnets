import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter


class GeneralizedSoftPlus(nn.Module):
    __constants__ = ['alpha', 'beta']
    alpha: Parameter
    beta: Parameter

    def __init__(self, alpha: float = 2.718, beta: float = 1.5) -> None:
        super(GeneralizedSoftPlus, self).__init__()
        self.alpha = Parameter(torch.tensor(
            alpha, dtype=torch.float64), requires_grad=True)
        self.beta = Parameter(torch.tensor(
            beta, dtype=torch.float64), requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        return torch.log(1 + self.alpha ** (self.beta * input)) / (self.beta * torch.log(self.alpha))

    def extra_repr(self) -> str:
        return 'beta={}, alpha={}'.format(self.beta, self.alpha)