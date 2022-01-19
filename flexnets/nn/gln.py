import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor


class GeneralizedLehmerLayer(nn.Module):
    __constants__ = ['in_features', 'out_features', 'alpha', 'beta']
    in_features: int
    out_features: int
    alpha: Tensor
    beta: Tensor
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 alpha: float = 1.5, beta: float = 0.5, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(GeneralizedLehmerLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(
            (out_features, in_features), **factory_kwargs))
        self.alpha = Parameter(torch.tensor(alpha, dtype=torch.float64))
        self.beta = Parameter(torch.tensor(beta, dtype=torch.float64))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        n = input.size()[0]
        m = input.matmul(self.weight.t()) + self.bias

        a = (n + 1) / torch.log(self.alpha)
        b = torch.log(self.alpha**((self.beta + 1) * m) / self.alpha**(self.beta * m))
        return a * b
