import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter


class GLM(nn.Module):
    def __init__(self, alpha: float = 1.5, beta: float = 1.2) -> None:
        super(GLM, self).__init__()
        self.alpha = Parameter(torch.tensor(
            alpha, dtype=torch.float64), requires_grad=True)
        self.beta = Parameter(torch.tensor(
            beta, dtype=torch.float64), requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        return (torch.log((torch.pow(self.alpha, (self.beta+1)*input)) / torch.pow(self.alpha, self.beta*input))
                / torch.log(self.alpha))


class GPM(nn.Module):
    def __init__(self, delta: float = 1.5, gamma: float = 1.2) -> None:
        super(GPM, self).__init__()
        self.delta = Parameter(torch.tensor(
            delta, dtype=torch.float64), requires_grad=True)
        self.gamma = Parameter(torch.tensor(
            gamma, dtype=torch.float64), requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        return (torch.log(torch.pow(self.gamma, input*self.delta)) /
                self.delta*torch.log(self.gamma))


class LNorm(nn.Module):
    def __init__(self) -> None:
        super(LNorm, self).__init__()
        self.glm_mean = GLM()
        self.glm_std = GLM()

    def forward(self, input: Tensor) -> Tensor:
        numerator = input - self.glm_mean(input)
        return (numerator / self.glm_std(numerator))


class PNorm(nn.Module):
    def __init__(self) -> None:
        super(PNorm, self).__init__()
        self.gpm_mean = GPM()
        self.gpm_std = GPM()

    def forward(self, input: Tensor) -> Tensor:
        numerator = input - self.gpm_mean(input)
        return (numerator / self.gpm_std(numerator))
