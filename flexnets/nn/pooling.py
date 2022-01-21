import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair


class LehmerPool2d(nn.Module):
    def __init__(self, k, kernel_size, stride, padding=0, dilation=1):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding
        self.dilation = _pair(dilation)
        self.k = Parameter(torch.tensor(
            k, dtype=torch.float64), requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        kw, kh = self.kernel_size
        a = F.avg_pool2d(input.pow(self.k+1), self.kernel_size,
                         self.stride, self.padding)
        b = F.avg_pool2d(input.pow(self.k), self.kernel_size,
                         self.stride, self.padding)
        pa = (torch.sign(a) * F.relu(torch.abs(a))).mul(kw * kh)
        pb = (torch.sign(b) * F.relu(torch.abs(b))).mul(kw * kh)
        out = pa/pb
        out = torch.max(torch.zeros_like(out), out)
        return out


class SoftLehmerPool2d(nn.Module):
    def __init__(self, k, kernel_size, stride, padding=0, dilation=1):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding
        self.dilation = _pair(dilation)
        self.k = Parameter(torch.tensor(
            k, dtype=torch.float64), requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        kw, kh = self.kernel_size
        a = F.avg_pool2d(torch.exp((self.k+1)*input), self.kernel_size,
                         self.stride, self.padding)
        b = F.avg_pool2d(torch.exp(self.k*input), self.kernel_size,
                         self.stride, self.padding)
        pa = (torch.sign(a) * F.relu(torch.abs(a))).mul(kw * kh)
        pb = (torch.sign(b) * F.relu(torch.abs(b))).mul(kw * kh)
        out = torch.log(pa/pb)
        return out


class GeneralizedLehmerPool2d(nn.Module):
    def __init__(self, alpha: float, beta: float, kernel_size, stride, padding=0, dilation=1):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding
        self.dilation = _pair(dilation)
        self.alpha = Parameter(torch.tensor(
            alpha, dtype=torch.float64, requires_grad=True))
        self.beta = Parameter(torch.tensor(
            beta, dtype=torch.float64, requires_grad=True))

    def forward(self, input: Tensor) -> Tensor:
        kw, kh = self.kernel_size
        a = F.avg_pool2d(self.alpha.pow((self.beta + 1) * input), self.kernel_size,
                         self.stride, self.padding)
        b = F.avg_pool2d(self.alpha.pow(self.beta * input), self.kernel_size,
                         self.stride, self.padding)
        pa = (torch.sign(a) * F.relu(torch.abs(a))).mul(kw * kh)
        pb = (torch.sign(b) * F.relu(torch.abs(b))).mul(kw * kh)
        z = torch.log(pa / pb)
        out = z / torch.log(self.alpha)
        return out


class GeneralizedPowerMeanPool2d(nn.Module):
    def __init__(self, gamma, delta, kernel_size, stride, padding=0, dilation=1):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding
        self.dilation = _pair(dilation)
        self.gamma = Parameter(torch.tensor(
            gamma, dtype=torch.float64, requires_grad=True))
        self.delta = Parameter(torch.tensor(
            delta, dtype=torch.float64, requires_grad=True))

    def forward(self, input: Tensor) -> Tensor:
        kw, kh = self.kernel_size
        a = F.avg_pool2d(self.gamma.pow(self.delta * input),
                         self.kernel_size, self.stride, self.padding)
        pa = (torch.sign(a) * F.relu(torch.abs(a))).mul(kw * kh)
        
        n = input.size()[0]
        out = (torch.log(pa) - torch.log(torch.tensor(n, dtype=torch.float32))) / (self.delta * torch.log(self.gamma))
        return out
