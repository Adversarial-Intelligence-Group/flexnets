from turtle import forward
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t
from typing import Optional, Union
import math

class GeneralizedLehmerConvolution(torch.nn.modules.conv._ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(GeneralizedLehmerConvolution, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)
        # TODO 
        self.alpha = nn.Parameter(torch.tensor(1.5), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(1.3), requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            x = F.pad(input, self._reversed_padding_repeated_twice,
                      mode=self.padding_mode)
        else:
            x = F.pad(input, self._reversed_padding_repeated_twice,
                      "constant", 0)

        output_shape = ((x.shape[2] - self.kernel_size[0])//self.stride[0] + 1,
                        (x.shape[3] - self.kernel_size[1])//self.stride[1] + 1)
        # x: torch.Size([64, 3, 34, 34]) 3 -> 32 ch
        x = F.unfold(input=x, kernel_size=self.kernel_size, dilation=self.dilation, stride=self.stride)  # torch.Size([64, 27, 1024])
        # n = self.kernel_size[0]*self.kernel_size[1]
        # multiplier = n/torch.log(self.alpha)
        dotprod = x.transpose(1, 2).matmul(weight.reshape(weight.shape[0], -1).t()).transpose(1, 2)  #torch.Size([64, 32, 1024])
        # numerator = torch.pow(self.alpha, (self.beta+1)*dotprod)
        # denominator = torch.pow(self.alpha, self.beta*dotprod)
        # x = multiplier*torch.log(numerator/denominator)  #torch.Size([64, 32, 1024])
        # x = x.reshape(input.shape[0], self.out_channels, *output_shape) # torch.Size([64, 32, 32, 32])
        x = F.fold(dotprod, output_shape, (1, 1))
        if bias is not None:
            x += bias.reshape(1, bias.shape[0], 1, 1)
        return x


class GeneralizedPowerConvolution(torch.nn.modules.conv._ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(GeneralizedPowerConvolution, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)
        # TODO 
        self.delta = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)

    
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            x = F.pad(input, self._reversed_padding_repeated_twice,
                      mode=self.padding_mode)
        else:
            x = F.pad(input, self._reversed_padding_repeated_twice,
                      "constant", 0)

        output_shape = ((x.shape[2] - self.kernel_size[0])//self.stride[0] + 1,
                        (x.shape[3] - self.kernel_size[1])//self.stride[1] + 1)
        # x: torch.Size([64, 3, 34, 34]) 3 -> 32 ch
        x = F.unfold(input=x, kernel_size=self.kernel_size, dilation=self.dilation, stride=self.stride)  # torch.Size([64, 27, 1024])        
        n = self.kernel_size[0]*self.kernel_size[1]
        multiplier = n/(self.delta*torch.log(self.gamma))
        dotprod = x.transpose(1, 2).matmul(weight.reshape(weight.shape[0], -1).t()).transpose(1, 2) #torch.Size([64, 32, 1024])
        dotprod_yd = torch.log(torch.pow(self.gamma, self.delta*dotprod)) - torch.log(n)
        x = multiplier*dotprod_yd #torch.Size([64, 32, 1024])
        x = x.reshape(input.shape[0], self.out_channels, *output_shape) # torch.Size([64, 32, 32, 32])
        if bias is not None:
            x += bias.reshape(1, bias.shape[0], 1, 1)
        return x

