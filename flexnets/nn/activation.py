import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter

from flexnets.nn.functions import GLM, GPM


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
        return torch.log(1 + torch.pow(self.alpha, (self.beta * input))) / (self.beta * torch.log(self.alpha))

    def extra_repr(self) -> str:
        return 'beta={}, alpha={}'.format(self.beta, self.alpha)


class GeneralizedLehmerSoftMax(nn.Module):
    __constants__ = ['alpha_1', 'alpha_2', 'beta_1', 'beta_2']
    alpha_1: Parameter
    alpha_2: Parameter
    beta_1: Parameter
    beta_2: Parameter

    def __init__(self, alpha_1: float, alpha_2: float, beta_1: float, beta_2: float) -> None:
        super(GeneralizedLehmerSoftMax, self).__init__()
        self.alpha_1 = Parameter(torch.tensor(
            alpha_1, dtype=torch.float64), requires_grad=True)
        self.alpha_2 = Parameter(torch.tensor(
            alpha_2, dtype=torch.float64), requires_grad=True)
        self.beta_1 = Parameter(torch.tensor(
            beta_1, dtype=torch.float64), requires_grad=True)
        self.beta_2 = Parameter(torch.tensor(
            beta_2, dtype=torch.float64), requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        l_mean = GLM(self.alpha_1, self.beta_1, input)
        l_std = GLM(self.alpha_2, self.beta_2, input - l_mean)
        l_norm = (input - l_mean) / l_std
        
        exp = torch.exp(l_norm)
        out = exp / torch.sum(exp, 1, keepdim=True)
        return out


class GeneralizedPowerSoftMax(nn.Module):
    __constants__ = ['gamma_1', 'gamma_2', 'delta_1', 'delta_2']
    gamma_1: Parameter
    gamma_2: Parameter
    delta_1: Parameter
    delta_2: Parameter

    def __init__(self, gamma_1: float, gamma_2: float, delta_1: float, delta_2: float) -> None:
        super(GeneralizedPowerSoftMax, self).__init__()
        self.gamma_1 = Parameter(torch.tensor(
            gamma_1, dtype=torch.float64), requires_grad=True)
        self.gamma_2 = Parameter(torch.tensor(
            gamma_2, dtype=torch.float64), requires_grad=True)
        self.delta_1 = Parameter(torch.tensor(
            delta_1, dtype=torch.float64), requires_grad=True)
        self.delta_2 = Parameter(torch.tensor(
            delta_2, dtype=torch.float64), requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        p_mean = GPM(self.gamma_1, self.delta_1, input)
        p_std = GLM(self.gamma_2, self.delta_2, input - p_mean)
        p_norm = (input - p_mean) / p_std
        
        exp = torch.exp(p_norm)
        out = exp / torch.sum(exp, 1, keepdim=True)
        return out