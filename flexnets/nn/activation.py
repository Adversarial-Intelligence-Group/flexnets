import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter

from flexnets.nn.preprocessing import LNorm, PNorm

class GLSoftMax(nn.Module):

    def __init__(self) -> None:
        super(GLSoftMax, self).__init__()
        self.lnorm = LNorm()

    def forward(self, input: Tensor) -> Tensor:
        return nn.Softmax()(self.lnorm(input))


class GPSoftMax(nn.Module):
    def __init__(self) -> None:
        super(GPSoftMax, self).__init__()
        self.pnorm = PNorm()

    def forward(self, input: Tensor) -> Tensor:
        return nn.Softmax()(self.pnorm(input))


class GReLU(nn.Module):
    def __init__(self, alpha: float = 1.5, beta: float = 1.2) -> None:
        super(GReLU, self).__init__()
        self.alpha = Parameter(torch.tensor(
            alpha, dtype=torch.float64), requires_grad=True)
        self.beta = Parameter(torch.tensor(
            beta, dtype=torch.float64), requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        return (torch.log(1 + torch.pow(self.alpha, self.beta*input)) /
                (self.beta*torch.log(self.alpha)))

    def extra_repr(self) -> str:
        return 'beta={}, alpha={}'.format(self.beta, self.alpha)
