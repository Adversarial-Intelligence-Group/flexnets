from turtle import forward
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t
from typing import Optional, Union


class LehmerConv2d(torch.nn.modules.conv._ConvNd):
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
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(LehmerConv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)
        self.k = nn.Parameter(torch.tensor(-0.5), requires_grad=False)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        batch_size = input.shape[0]
        if self.padding_mode != 'zeros':
            x = F.pad(input, self._reversed_padding_repeated_twice,
                      mode=self.padding_mode)
        else:
            x = F.pad(input, self._reversed_padding_repeated_twice,
                      "constant", 0)
        output_shape = ((x.shape[2] - self.kernel_size[0])//self.stride[0] + 1,
                        (x.shape[3] - self.kernel_size[1])//self.stride[1] + 1)
        x = F.unfold(x, self.kernel_size, self.dilation,
                     self.padding, self.stride)
        x = (torch.pow(x, self.k+1).transpose(1, 2).matmul(weight.view(weight.size(
            0), -1).t()).transpose(1, 2)/torch.pow(x, self.k).transpose(1, 2).matmul(weight.view(weight.size(
                0), -1).t()).transpose(1, 2))*torch.sum(weight)
        x = x.reshape(batch_size, self.out_channels, *output_shape)
        return x

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)

class SoftLehmerConv2d(torch.nn.modules.conv._ConvNd):

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
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(SoftLehmerConv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)
        self.k = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            x = F.pad(input, self._reversed_padding_repeated_twice,
                      mode=self.padding_mode)
        else:
            x = F.pad(input, self._reversed_padding_repeated_twice,
                      "constant", 0)

        output_shape = ((x.shape[2] - self.kernel_size[0])//self.stride[0] + 1,
                        (x.shape[3] - self.kernel_size[1])//self.stride[1] + 1)
        x = F.unfold(x, self.kernel_size, self.dilation,
                     self.padding, self.stride)  # torch.Size([32, 25, 576])
        x = torch.log((torch.exp(x*(self.k+1)).transpose(1, 2).matmul(weight.reshape(weight.size(
            0), -1).t()).transpose(1, 2)/torch.exp(x*self.k).transpose(1, 2).matmul(weight.reshape(weight.size(
                0), -1).t()).transpose(1, 2)))  # torch.Size([32, 10, 576]) # FIXME I think that *n is redundant
        # torch.Size([32, 10, 24, 24])
        x = x.reshape(input.shape[0], self.out_channels, *output_shape)
        return x

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)


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
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(SoftLehmerConv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)
        # TODO #FIXME
        # self.k = nn.Parameter(torch.tensor(0.), requires_grad=False)

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
        # TODO #FIXME
        # x = F.unfold(x, self.kernel_size, self.dilation,
        #              self.padding, self.stride)  # torch.Size([32, 25, 576])
        # x = torch.log((torch.exp(x*(self.k+1)).transpose(1, 2).matmul(weight.reshape(weight.size(
        #     0), -1).t()).transpose(1, 2)/torch.exp(x*self.k).transpose(1, 2).matmul(weight.reshape(weight.size(
        #         0), -1).t()).transpose(1, 2)))  # torch.Size([32, 10, 576]) # FIXME I think that *n is redundant
        # torch.Size([32, 10, 24, 24])
        x = x.reshape(input.shape[0], self.out_channels, *output_shape)
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
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(SoftLehmerConv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)
        # TODO #FIXME
        # self.k = nn.Parameter(torch.tensor(0.), requires_grad=False)

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
        # TODO #FIXME
        # x = F.unfold(x, self.kernel_size, self.dilation,
        #              self.padding, self.stride)  # torch.Size([32, 25, 576])
        # x = torch.log((torch.exp(x*(self.k+1)).transpose(1, 2).matmul(weight.reshape(weight.size(
        #     0), -1).t()).transpose(1, 2)/torch.exp(x*self.k).transpose(1, 2).matmul(weight.reshape(weight.size(
        #         0), -1).t()).transpose(1, 2)))  # torch.Size([32, 10, 576]) # FIXME I think that *n is redundant
        # torch.Size([32, 10, 24, 24])
        x = x.reshape(input.shape[0], self.out_channels, *output_shape)
        return x

