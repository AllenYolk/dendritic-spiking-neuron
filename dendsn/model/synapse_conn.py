"""Synaptic connection models.

This module defines several wrappers for  commonly used weight layers like 
Linear and Conv2d so as to 
1. support both single-step and multi-step mode.
2. enable 0-1 masks to mimic sparse synaptic connections.
3. enable weight clamping.
"""

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
    """Base class for all synaptic connections.

    A synaptic connection module is a wrapper for weight layers in Pytorch. 
    Subclasses of BaseSynapseConn must support single-step and multi-step mode
    by implementing singe_step_forward() and multi_step_forward(), and 
    optionally use a 0-1 mask to mimic sparse synaptic connections.

    Attributes:
        wmin, wmax (torch.Tensor): the range of synaptic weights.
        step_mode (str): "s" for single-step mode, and "m" for multi-step mode.
    """

    def __init__(
        self, 
        wmin: Union[float, torch.Tensor] = -float("inf"), 
        wmax: Union[float, torch.Tensor] = float("inf"), 
        step_mode: str = "s",
    ):
        """The constructor of BaseSynapseConn.

        Args:
            wmin, wmax (Union[float, torch.Tensor]): the range of synaptic 
                weights. These values must be broadcastable to the shape of 
                the synaptic weight tensor.
            step_mode (str, optional): "s" for single-step mode, and "m" for 
                multi-step mode. Defaults to "s".
        """
        super().__init__()
        self.step_mode = step_mode
        self.wmin = wmin
        self.wmax = wmax

    @property
    def wmin(self):
        return self._wmin

    @wmin.setter
    def wmin(self, w):
        self._wmin = torch.tensor(w)

    @property
    def wmax(self):
        return self._wmax

    @wmax.setter
    def wmax(self, w):
        self._wmax = torch.tensor(w)

    def clamp_weights(self, clamp_bias: bool = False):
        torch.clamp_(self.weight, min=self.wmin, max=self.wmax)
        if clamp_bias and (self.bias is not None):
            torch.clamp_(self.bias, min=self.wmin, max=self.wmax)

    @abc.abstractmethod
    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        T = x_seq.shape[0]
        y_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
        return torch.stack(y_seq)

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
    """Wrapped nn.Linear.

    Attributes:
        See base class: nn.Linear, BaseSynapseConn.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False,
        device = None, dtype = None, 
        wmin: Union[float, torch.Tensor] = -float("inf"), 
        wmax: Union[float, torch.Tensor] = float("inf"), 
        step_mode: str = "s"
    ):
        """The constructor of LinearSynapseConn.

        Args:
            in_features (int)
            out_features (int)
            bias (bool, optional): Defaults to False.
            device (_type_, optional): Defaults to None.
            dtype (_type_, optional): Defaults to None.
            wmin, wmax (Union[float, torch.Tensor]): the range of synaptic 
                weights. These values must be broadcastable to the shape of 
                the synaptic weight tensor.
            step_mode (str, optional): "s" for single-step mode, and "m" for
                multi-step mode. Defaults to "s".
        """
        # print(self.__class__.__mro__)
        super().__init__(in_features, out_features, bias, device, dtype)
        self.step_mode = step_mode
        self.wmin, self.wmax = wmin, wmax

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return super().forward(x_seq)


class MaskedLinearSynapseConn(BaseSynapseConn):
    """Reimplemented nn.Linear with a sparsity mask.

    The source code of torch.nn.Linear is modified to incorporate a 0-1 sparsity
    mask and support both single-step and multi-step mode. The sparsity mask is
    randomly initialized given a sparsity value.

    Attributes:
        See nn.Linear and base class: BaseSynapseConn.
        weight_mask (torch.Tensor): the sparsity mask of synaptic connection,
            whose shape is [out_features, in_features].
        init_sparsity (torch.Tensor): weight_mask's initial coefficient of 
            sparsity, used to initialize weight_mask. Higher value indicates 
            higher sparsity.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False,
        init_sparsity: float = 0.75, device = None, dtype = None,
        wmin: Union[float, torch.Tensor] = -float("inf"), 
        wmax: Union[float, torch.Tensor] = float("inf"), 
        step_mode: str = "s"
    ):
        """The constructor of MaskedLinearSynapseConn.

        Args:
            in_features (int)
            out_features (int)
            bias (bool, optional): Defaults to False.
            init_sparsity (float, optional): the sparsity of the 0-1 mask when 
                it is initialized [higher -> sparser]. Defaults to 0.75 .
            device (optional): Defaults to None.
            dtype (optional): Defaults to None.
            wmin, wmax (Union[float, torch.Tensor]): the range of synaptic 
                weights. These values must be broadcastable to the shape of 
                the synaptic weight tensor.
            step_mode (str, optional): "s" for single-step mode, and "m" for
                multi-step mode. Defaults to "s".
        """
        super().__init__(wmin=wmin, wmax=wmax, step_mode=step_mode)
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
        """Reset weight, bias and weight_mask.
        """
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


class Conv1dSynapseConn(nn.Conv1d, BaseSynapseConn):
    """Wrapped nn.Conv1d.

    Attributes:
        See base class: nn.Conv1d, BaseSynapseConn.
    """

    def __init__(
        self, in_channels: int, out_channels: int, 
        kernel_size: ttypes._size_1_t, stride: ttypes._size_1_t = 1, 
        padding: Union[ttypes._size_1_t, str]=0, 
        dilation: ttypes._size_1_t = 1, groups: int = 1, 
        bias: bool = True, padding_mode: str = 'zeros', 
        device = None, dtype = None,
        wmin: Union[float, torch.Tensor] = -float("inf"), 
        wmax: Union[float, torch.Tensor] = float("inf"), 
        step_mode: str = "s"
    ):
        """The constructor of Conv2dSynapseConn.

        Args:
            in_channels (int)
            out_channels (int)
            kernel_size (ttypes._size_1_t)
            stride (ttypes._size_1_t, optional): Defaults to 1.
            padding (Union[ttypes._size_1_t, str], optional): Defaults to 0.
            dilation (ttypes._size_1_t, optional): Defaults to 1.
            groups (int, optional): Defaults to 1.
            bias (bool, optional): Defaults to False.
            padding_mode (str, optional): Defaults to "zeros".
            device (optional): Defaults to None.
            dtype (optional): Defaults to None.
            wmin, wmax (Union[float, torch.Tensor]): the range of synaptic 
                weights. These values must be broadcastable to the shape of 
                the synaptic weight tensor.
            step_mode (str, optional): Defaults to "s".
        """
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype
        )
        self.step_mode = step_mode
        self.wmin, self.wmax = wmin, wmax

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return functional.unfold_forward_fold(x_seq, super().forward)


class Conv2dSynapseConn(nn.Conv2d, BaseSynapseConn):
    """Wrapped nn.Conv2d.

    Attributes:
        See base class: nn.Conv2d, BaseSynapseConn.
    """

    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: ttypes._size_2_t, stride: ttypes._size_2_t = 1,
        padding: Union[ttypes._size_2_t, str] = 0,
        dilation: ttypes._size_2_t = 1, groups: int = 1,
        bias: bool = False, padding_mode: str = "zeros", 
        device = None, dtype = None, 
        wmin: Union[float, torch.Tensor] = -float("inf"), 
        wmax: Union[float, torch.Tensor] = float("inf"), 
        step_mode: str = "s"
    ):
        """The constructor of Conv2dSynapseConn.

        Args:
            in_channels (int)
            out_channels (int)
            kernel_size (ttypes._size_2_t)
            stride (ttypes._size_2_t, optional): Defaults to 1.
            padding (Union[ttypes._size_2_t, str], optional): Defaults to 0.
            dilation (ttypes._size_2_t, optional): Defaults to 1.
            groups (int, optional): Defaults to 1.
            bias (bool, optional): Defaults to False.
            padding_mode (str, optional): Defaults to "zeros".
            device (optional): Defaults to None.
            dtype (optional): Defaults to None.
            wmin, wmax (Union[float, torch.Tensor]): the range of synaptic 
                weights. These values must be broadcastable to the shape of 
                the synaptic weight tensor.
            step_mode (str, optional): Defaults to "s".
        """
        # print(self.__class__.__mro__)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype
        )
        self.step_mode = step_mode
        self.wmin, self.wmax = wmin, wmax

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return functional.unfold_forward_fold(x_seq, super().forward)


class Conv3dSynapseConn(nn.Conv3d, BaseSynapseConn):
    """Wrapped nn.Conv3d.

    Attributes:
        See base class: nn.Conv3d, BaseSynapseConn.
    """

    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: ttypes._size_3_t, stride: ttypes._size_3_t = 1,
        padding: Union[str, ttypes._size_3_t] = 0,
        dilation: ttypes._size_3_t = 1, groups: int = 1,
        bias: bool = True, padding_mode: str = 'zeros',
        device = None, dtype = None,
        wmin: Union[float, torch.Tensor] = -float("inf"), 
        wmax: Union[float, torch.Tensor] = float("inf"), 
        step_mode: str = "s"
    ):
        """The constructor of Conv3dSynapseConn.

        Args:
            in_channels (int)
            out_channels (int)
            kernel_size (ttypes._size32_t)
            stride (ttypes._size_3_t, optional): Defaults to 1.
            padding (Union[ttypes._size_3_t, str], optional): Defaults to 0.
            dilation (ttypes._size_3_t, optional): Defaults to 1.
            groups (int, optional): Defaults to 1.
            bias (bool, optional): Defaults to False.
            padding_mode (str, optional): Defaults to "zeros".
            device (optional): Defaults to None.
            dtype (optional): Defaults to None.
            wmin, wmax (Union[float, torch.Tensor]): the range of synaptic 
                weights. These values must be broadcastable to the shape of 
                the synaptic weight tensor.
            step_mode (str, optional): Defaults to "s".
        """
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype
        )
        self.step_mode = step_mode
        self.wmin, self.wmax = wmin, wmax

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return functional.unfold_forward_fold(x_seq, super().forward)