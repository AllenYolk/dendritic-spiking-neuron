import abc

import torch
import torch.nn as nn
from spikingjelly.activation_based import base


class BaseSynapseFilter(base.MemoryModule, abc.ABC):

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
                f"BaseSynapseFilter.step_mode should be 'm' or 's', "
                f"but get {self.step_mode} instead."
            )


class IdentitySynapseFilter(BaseSynapseFilter):

    def __init__(self, step_mode: str = "s"):
        """
        This synaptic filter conducts identity mapping (thus do nothing).

        Args:
            step_mode (str, optional): Defaults to "s".
        """
        super().__init__(step_mode = step_mode)

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return x_seq