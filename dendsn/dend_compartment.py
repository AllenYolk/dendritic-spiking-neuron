import abc

import torch
from spikingjelly.activation_based import base


class BaseDendCompartment(base.MemoryModule, abc.ABC):

    def __init__(self, v_init: float = 0.):
        super().__init__()
        self.register_memory("v", v_init)

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class PassiveDendCompartment(BaseDendCompartment):

    def __init__(
        self, tau: float = 2, decay_input: bool =  True, v_rest: float = 0.,
    ):
        super().__init__(v_init = v_rest)
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

    def forward(self, x: torch.Tensor):
        if self.decay_input:
            self.v = PassiveDendCompartment.single_step_decay_input(
                self.v, x, self.v_rest, self.tau
            )
        else:
            self.v = PassiveDendCompartment.single_step_not_decay_input(
                self.v, x, self.v_rest, self.tau
            )
        return self.v
