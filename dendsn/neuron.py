import abc

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron as sj_neuron

from dendsn import dend_soma_conn, dendrite


class BaseDendNeuron(nn.Module, abc.ABC):

    def __init__(
        self, dend: dendrite.BaseDend, soma: sj_neuron.BaseNode,
        dend_soma_conn: dend_soma_conn.BaseDendSomaConn,
        step_mode: str = "s"
    ):
        super().__init__()
        self.dend = dend
        self.soma = soma
        self.dend_soma_conn = dend_soma_conn
        self.step_mode = step_mode

    def reset(self):
        self.dend.reset()
        self.soma.reset()

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
                f"BaseDendNeuron.step_mode should be 'm' or 's', "
                f"but get {self.step_mode} instead."
            )


class VForwardDendNeuron(BaseDendNeuron):

    def __init__(self):
        pass


class VBidirectionDendNeuron(BaseDendNeuron):

    def __init__(self):
        pass