import abc
import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.parameter as P
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.common_types as ttypes

from dendsn import functional


class BaseSynapseConn(nn.Module, abc.ABC):

    def __init__(self, step_mode: str = "s"):
        super().__init__()
        self.step_mode = step_mode

    @abc.abstractmethod
    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode == "s":
            return self.single_step_forward(x)
        elif self.step_mode == "m":
            return self.multi_step_forward(x)
        else:
            raise ValueError(
                f"BaseSynapseConn.step_mode should be 'm' or 's', "
                f"but get {self.step_mode} instead."
            )


class LinearSynapseConn(nn.Linear, BaseSynapseConn):

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False,
        device = None, dtype = None, step_mode: str = "s"
    ):
        """
        A wrapper for nn.Linear (fully connected).

        Args:
            in_features (int)
            out_features (int)
            bias (bool, optional): Defaults to False.
            device (_type_, optional): Defaults to None.
            dtype (_type_, optional): Defaults to None.
            step_mode (str, optional): Defaults to "s".
        """
        # print(self.__class__.__mro__)
        super().__init__(in_features, out_features, bias, device, dtype)
        self.step_mode = step_mode

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forwrad(x, self.weight, self.bias)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return super().forward(x_seq, self.weight, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode == "s":
            return self.single_step_forward(x)
        elif self.step_mode == "m":
            return self.multi_step_forward(x)
        else:
            raise ValueError(
                f"LinearSynapseConn.step_mode should be 'm' or 's', "
                f"but get {self.step_mode} instead."
            ) 


class MaskedLinearSynapseConn(BaseSynapseConn):

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False,
        init_sparsity: float = 0.75, device = None, dtype = None,
        step_mode: str = "s"
    ):
        """
        nn.Linear with a 0-1 sparsity mask (to simulate synaptic connections).

        Args:
            in_features (int)
            out_features (int)
            bias (bool, optional): Defaults to False.
            init_sparsity (float, optional): the sparsity of the 0-1 mask when 
                it is initialized [higher -> sparser]. Defaults to 0.75 .
            device (_type_, optional): Defaults to None.
            dtype (_type_, optional): Defaults to None.
            step_mode (str, optional): Defaults to "s".
        """
        super().__init__(step_mode = step_mode)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        # self.xxx = Parameter(val) is (almost) equivalent to 
        # self.register_parameter(xxx, val) [`=` is overrode]
        # torch.empty won't set the memory of the tensor to specific values
        self.weight = P.Parameter(
            torch.empty([out_features, in_features], **factory_kwargs),
        )
        if bias:
            self.bias = P.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.weight_mask = P.Parameter(
            torch.empty(
                size = [out_features, in_features],
                **factory_kwargs
            ), requires_grad = False
        )
        self.init_sparsity = init_sparsity
        self.reset_parameters()

    def reset_parameters(self):
        # the use of torch.nn.init module
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        init.sparse_(self.weight_mask, sparsity = self.init_sparsity)
        self.weight_mask.data = (self.weight_mask.data!=0).to(dtype=torch.int32)

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.weight_mask, self.bias)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return F.linear(x_seq, self.weight * self.weight_mask, self.bias)


class Conv2dSynapseConn(nn.Conv2d, BaseSynapseConn):

    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: ttypes._size_2_t, stride: ttypes._size_2_t = 1,
        padding: Union[ttypes._size_2_t, str] = 0,
        dilation: ttypes._size_2_t = 1, groups: int = 1,
        bias: bool = False, padding_mode: str = "zeros", 
        device = None, dtype = None, step_mode: str = "s"
    ):
        # print(self.__class__.__mro__)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype
        )
        # print(self.step_mode)
        self.step_mode = step_mode

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return functional.unfold_forward_fold(x_seq, super().forward)

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode == "s":
            return self.single_step_forward(x)
        elif self.step_mode == "m":
            return self.multi_step_forward(x)
        else:
            raise ValueError(
                f"LinearSynapseConn.step_mode should be 'm' or 's', "
                f"but get {self.step_mode} instead."
            ) 