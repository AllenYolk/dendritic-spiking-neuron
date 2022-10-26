"""
This module defines the voltage dynamics of the dendritic compartments.

Dendritic compartments are treated independently. The relationship (wiring 
or couping) among a set of compartments is not considered here.
"""

import abc

import torch
from spikingjelly.activation_based import base


class BaseDendCompartment(base.MemoryModule, abc.ABC):

    def __init__(self, v_init: float = 0., step_mode: str = "s"):
        super().__init__()
        self.register_memory("v", v_init)
        self.step_mode = step_mode

    def v_float2tensor(self, x: torch.Tensor):
        """
        If self.v is a float, turn it into a tensor with the same shape of x

        Args:
            x (torch.Tensor)
        """
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)

    @abc.abstractmethod
    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

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
                f"BaseDendCompartment.step_mode shoud be 'm' or 's', "
                f"but get {self.step_mode} instead."
            )


class PassiveDendCompartment(BaseDendCompartment):

    def __init__(
        self, tau: float = 2, decay_input: bool =  True, v_rest: float = 0.,
        step_mode: str = "s"
    ):
        super().__init__(v_init = v_rest, step_mode = step_mode)
        self.tau = tau
        self.decay_input = decay_input
        self.v_rest = v_rest

    @staticmethod
    @torch.jit.script
    def single_step_decay_input(
        v: torch.Tensor, x: torch.Tensor, v_rest: float, tau: float
    ):
        v = v + (x - (v - v_rest)) / tau
        return v

    @staticmethod
    @torch.jit.script
    def single_step_not_decay_input(
        v: torch.Tensor, x: torch.Tensor, v_rest: float, tau: float
    ):
        v = v - (v - v_rest) / tau + x
        return v

    def single_step_forward(self, x: torch.Tensor):
        if self.decay_input:
            self.v = PassiveDendCompartment.single_step_decay_input(
                self.v, x, self.v_rest, self.tau
            )
        else:
            self.v = PassiveDendCompartment.single_step_not_decay_input(
                self.v, x, self.v_rest, self.tau
            )
        return self.v
