import abc
import math

import torch
import torch.nn as nn
import torch.nn.parameter as P
import torch.nn.functional as F
import torch.nn.init as init


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


class MaskedLinearSynapseConn(BaseSynapseConn):

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False,
        init_sparsity: float = 0.75, device = None, dtype = None,
        step_mode: str = "s"
    ):
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