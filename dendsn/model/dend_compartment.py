"""The voltage dynamics of dendritic compartments.

This package contains a series of classes depicting different types of dendritic
compartments so that we can compute the dendritic voltage dynamics step by step,
given the input to all the compartments. When computing the dendritic voltage 
dynamics for a single time step, the compartments are treated independently. 
The relationship (wiring) among a set of compartments is not considered here.
"""

import abc
from typing import Callable

import torch
from spikingjelly.activation_based import base


class BaseDendCompartment(base.MemoryModule, abc.ABC):
    """Base class for all dendritic compartments.

    Attributes:
        v (Union[float, torch.Tensor]): voltage of the dendritic compartment(s)
            at the current time step.
        step_mode (str): "s" for single-step mode, and "m" for multi-step mode.
    """

    def __init__(self, v_init: float = 0., step_mode: str = "s"):
        """The constructor of BaseDendCompartment.

        Args:
            v_init (float, optional): initial voltage (at time step 0). 
                Defaults to 0..
            step_mode (str, optional): "s" for single-step mode, and "m" for
                multi-step mode. Defaults to "s".
        """
        super().__init__()
        self.register_memory("v", v_init)
        self.step_mode = step_mode

    def v_float2tensor(self, x: torch.Tensor):
        """If self.v is a float, turn it into a tensor with x's shape.
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
    """Passive dendritic compartments.

    A passive dendritic compartment is just a leaky integrator without a firing
    mechanism.

    Attributes:
        v (Union[float, torch.Tensor]): voltage of the dendritic compartment(s)
            at the current time step
        step_mode (str): "s" for single-step mode, and "m" for multi-step mode
        tau(float): the time constant
        decay_input (bool, optional): whether the input to the compartments
            should be divided by tau.
        v_rest (float, optional): resting potential.
    """

    def __init__(
        self, tau: float = 2, decay_input: bool = True, v_rest: float = 0.,
        step_mode: str = "s"
    ):
        """The constructor of PassiveDendCompartment

        Args:
            tau (float, optional): the time constant. Defaults to 2.
            decay_input (bool, optional): whether the input to the compartments
                should be divided by tau. Defaults to True.
            v_rest (float, optional): resting potential. Defaults to 0..
            step_mode (str, optional): "s" for single-step mode, and "m" for
                multi-step mode. Defaults to "s".
        """
        super().__init__(v_init=v_rest, step_mode=step_mode)
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


class PAComponentDendCompartment(BaseDendCompartment):
    """Dendritic compartment with passive and active voltage components.

    The passive component acts just like a leaky integrator without a firing
    mechanism, while the active voltage component is a function of the passive 
    voltage component. The overall voltage is the sum of active and passive 
    components: 
        v[t] = va[t] + vp[t] = f_dca(vp[t]) + vp[t]

    Attributes:
        v (Union[float, torch.Tensor]): voltage of the dendritic compartment(s)
            at the current time step.
        va (Union[float, torch.Tensor]): the active component of the 
            compartmental voltage at the current time step.
        vp (Union[float, torch.Tensor]): the passive component of the 
            compartmental voltage at the current time step.
        step_mode (str): "s" for single-step mode, and "m" for multi-step mode
        tau(float): the time constant for the passive component.
        decay_input (bool, optional): whether the input to the compartments
            should be divided by tau.
        v_rest (float, optional): resting potential.
        f_dca (Callable): the dendritic compartment activation function, mapping
            the passive voltage component to the active component. The input and
            output should have the same shape.
    """

    def __init__(
        self, tau: float = 2., decay_input: bool = True, v_rest: float = 0., 
        f_dca: Callable = lambda x: 0., step_mode: str = "s"
    ):
        """The constructor of PAComponentDendCompartment

        Args:
            tau (float, optional): the time constant. Defaults to 2.
            decay_input (bool, optional): whether the input to the compartments
                should be divided by tau. Defaults to True.
            v_rest (float, optional): resting potential. Defaults to 0..
            f_dc (Callable): the dendritic compartment activation function, 
                mapping the passive voltage component to the active component. 
                The input and output should have the same shape. Defaults to 
                the constant zero.
            step_mode (str, optional): "s" for single-step mode, and "m" for
                multi-step mode. Defaults to "s".
        """
        super().__init__(v_rest, step_mode)
        self.tau = tau
        self.decay_input = decay_input
        self.v_rest = v_rest
        self.f_dca = f_dca
        self.register_memory("va", 0.)
        self.register_memory("vp", v_rest)

    def v_float2tensor(self, x: torch.Tensor):
        """If self.v | vp | va is a float, turn it into a tensor with x's shape.
        """
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)
        if isinstance(self.va, float):
            v_init = self.va
            self.va = torch.full_like(x.data, v_init)
        if isinstance(self.vp, float):
            v_init = self.vp
            self.vp = torch.full_like(x.data, v_init)

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.decay_input:
            self.vp = self.vp + (x - (self.vp - self.v_rest)) / self.tau
        else:
            self.vp = self.vp + x - (self.vp - self.v_rest) / self.tau

        self.va = self.f_dca(self.vp)
        self.v = self.vp + self.va
        return self.v
