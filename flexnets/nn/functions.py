import torch
from torch import Tensor
import torch.nn.functional as F


def GLM(alpha: Tensor, beta: Tensor, input: Tensor) -> Tensor:
    a = alpha.pow((beta + 1) * input)
    b = alpha.pow(beta * input)

    pa = (torch.sign(a) * F.relu(torch.abs(a)))
    pb = (torch.sign(b) * F.relu(torch.abs(b)))
    z = torch.log(pa / pb)
    out = z / torch.log(alpha)

    return out


def GPM(gamma: Tensor, delta: Tensor, input: Tensor) -> Tensor:
    a = gamma.pow(delta * input)
    pa = torch.sign(a) * F.relu(torch.abs(a))
    n = input.size()[0]
    out = (torch.log(pa) - torch.log(torch.tensor(n, dtype=torch.float32))
           ) / (delta * torch.log(gamma))

    return out
