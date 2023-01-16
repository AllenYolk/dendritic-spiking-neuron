"""Synaptic filter models.

This module defines various synaptic filters which filter (along time dimension)
the raw signal given out by synaptic connections.
"""

import abc

import torch
from spikingjelly.activation_based import base
from spikingjelly.activation_based import neuron


class BaseSynapseFilter(base.MemoryModule, abc.ABC):
    """Base class for all synaptic filter models.

    A synaptic filter often follows a synaptic connection layer, filtering the 
    output of synaptic connection along the time dimension. Typically, it's a
    module with state. To define a subclass, just implement reset() and
    single_step_forward() methods.

    Attributes:
        step_mode (str): "s" for single-step mode, and "m" for multi-step mode.
    """

    def __init__(self, step_mode: str = "s"):
        """The constructor of BaseSynapseConn.

        Args:
            step_mode (str, optional): "s" for single-step mode, and "m" for 
                multi-step mode. Defaults to "s".
        """
        super().__init__()
        self.step_mode = step_mode

    @abc.abstractmethod
    def reset(self):
        pass

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
                f"BaseSynapseFilter.step_mode should be 'm' or 's', "
                f"but get {self.step_mode} instead."
            )


class IdentitySynapseFilter(BaseSynapseFilter):
    """Synaptic filter which conducts identity mapping (thus do nothing).

    Attributes:
        See base class: BaseSynapseFilter.
    """

    def __init__(self, step_mode: str = "s"):
        """The constructor of IdentitySynapseFilter.

        Args:
            step_mode (str, optional): "s" for single-step mode, and "m" for
                multi-step mode. Defaults to "s".
        """
        super().__init__(step_mode = step_mode)

    def reset(self):
        pass

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return x_seq


class LISynapseFilter(BaseSynapseFilter):
    """Synaptic filter that acts as a leaky integrator (doesn't decay input).

    Given the input x[t] (which is the output of synaptic connection module) at
    the current time step and state s[t-1] at the previous time step, the 
    current state s[t] and output y[t] is defined as
        s[t] = (1 - 1/tau) * s[t-1] + x[t]
        y[t] = s[t]
    To implement a leaky integrator, we can use a LIF neuron with infinite
    firing threshold (so that it'll never fire).

    Args:
        tau (float): the time constant.
        leaky_integrator (LIFNode): the leak integrator computing the state
            update. A LIF neuron with infinite firing threshold (thus it will
            never emit a spike).
        step_mode (str): see base class BaseSynapseFilter.
    """

    def __init__(self, tau: float, step_mode: str = "s"):
        """The constructor of LISynapseFilter.

        Args:
            tau (float): the time constant.
            step_mode (str, optional): "s" for single-step mode, and "m" for
                multi-step mode. Defaults to "s".
        """
        super().__init__(step_mode)
        self.tau = tau
        self.leaky_integrator = neuron.LIFNode(
            tau = tau, decay_input = False, v_threshold = float("inf"),
            step_mode = step_mode
        )

    def reset(self):
        self.leaky_integrator.reset()

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        self.leaky_integrator.single_step_forward(x)
        return self.leaky_integrator.v

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        self.leaky_integrator.store_v_seq = True
        self.leaky_integrator.multi_step_forward(x_seq)
        self.leaky_integrator.store_v_seq = False
        return self.v_seq