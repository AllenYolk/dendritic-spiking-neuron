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
        store_v_seq (bool): whether to store the compartmental potential at 
            every time step when using multi-step mode. If True, there is 
            another attribute called v_seq.
    """

    def __init__(
        self, v_init: float = 0., 
        step_mode: str = "s", store_v_seq: bool = False,
    ):
        """The constructor of BaseDendCompartment.

        Args:
            v_init (float, optional): initial voltage (at time step 0). 
                Defaults to 0..
            step_mode (str, optional): "s" for single-step mode, and "m" for
                multi-step mode. Defaults to "s".
            store_v_seq (bool, optional): whether to store the compartmental 
                potential at every time step when using multi-step mode. 
                Defaults to False.
        """
        super().__init__()
        self.register_memory("v", v_init)
        self.step_mode = step_mode
        self.store_v_seq = store_v_seq

    @property
    def store_v_seq(self) -> bool:
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, val: bool):
        self._store_v_seq = val
        if val and (not hasattr(self, "v_seq")):
            self.register_memory("v_seq", None)

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
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
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
            at the current time step.
        step_mode (str): "s" for single-step mode, and "m" for multi-step mode.
        store_v_seq (bool): whether to store the compartmental potential at 
            every time step when using multi-step mode. If True, there is 
            another attribute called v_seq.
        tau(float): the time constant
        decay_input (bool, optional): whether the input to the compartments
            should be divided by tau.
        v_rest (float, optional): resting potential.
    """

    def __init__(
        self, tau: float = 2, decay_input: bool = True, v_rest: float = 0.,
        step_mode: str = "s", store_v_seq: bool = False
    ):
        """The constructor of PassiveDendCompartment

        Args:
            tau (float, optional): the time constant. Defaults to 2.
            decay_input (bool, optional): whether the input to the compartments
                should be divided by tau. Defaults to True.
            v_rest (float, optional): resting potential. Defaults to 0..
            step_mode (str, optional): "s" for single-step mode, and "m" for
                multi-step mode. Defaults to "s".
            store_v_seq (bool, optional): whether to store the compartmental 
                potential at every time step when using multi-step mode. 
                Defaults to False.
        """
        super().__init__(v_rest, step_mode, store_v_seq)
        self.tau = tau
        self.decay_input = decay_input
        self.v_rest = v_rest

        self._tau_matrix = None
        self._tau_vector_for_v_init = None
        self._tau_vector_for_v_rest = None

    @staticmethod
    @torch.jit.script
    def single_step_decay_input(
        v: torch.Tensor, x: torch.Tensor, v_rest: float, tau: float
    ):
        v = v + (x - (v - v_rest)) / tau
        # v = (1 - 1/tau)*v + v_rest/tau + x/tau
        return v

    @staticmethod
    @torch.jit.script
    def single_step_not_decay_input(
        v: torch.Tensor, x: torch.Tensor, v_rest: float, tau: float
    ):
        v = v - (v - v_rest) / tau + x
        # v = (1 - 1/tau)*v + v_rest/tau + x
        return v

    def _init_tau_matrix(self, T: int):
        # v[t=0] = (1-1/tau)*v_init + x[t=0] + v_rest/tau
        # v[t=1] = (1-1/tau)*v[t=0] + x[t=1] + v_rest/tau
        #        = (1-1/tau)^2*v_init + (1-1/tau)*x[t=0] + x[t=1] + (1-1/tau)*v_rest/tau + v_rest/tau
        #        = (1-1/tau)^2*v_init + (1-1/tau)*x[t=0] + x[t=1] + (1 + (1-1/tau)) * v_rest/tau
        # v[t=2] = (1-1/tau)*v[t=1] + x[t=2] + v_rest/tau
        #        = (1-1/tau)^3*v_init  (1-1/tau)^2*x[t=0] + (1-1/tau)*x[t=1] + x[t=2] + ((1 + (1-1/tau))(1-1/tau)+1)
        # ......

        # have initialized
        if (self._tau_matrix is not None) and (T == self._tau_matrix.shape[0]):
            return

        self._tau_matrix = torch.zeros((T, T))
        self._tau_vector_for_v_init = torch.zeros((T))
        self._tau_vector_for_v_rest = torch.zeros((T))

        # self._tau_matrix[i, j]: the coefficient of x[t=i] in v[t=j]
        # lower triangle
        for i in range(T):
            for j in range(i, T):
                self._tau_matrix[i, j] = (1. - 1. / self.tau) ** (j - i)

        # self._tau_vector_for_v_init[i]: the coefficient of v_init in v[t=i]
        for i in range(T):
            self._tau_vector_for_v_init[i] = (1. - 1. / self.tau) ** (i + 1.)

        # self._tau_vector_for_v_rest[i]: the coefficient of v_rest in v[t=i]
        c = 0.
        for i in range(T):
            self._tau_vector_for_v_rest[i] = c * (1. - 1. / self.tau) + 1.
            c = self._tau_vector_for_v_rest[i]
        self._tau_vector_for_v_rest = self._tau_vector_for_v_rest / self.tau

    def single_step_forward(self, x: torch.Tensor):
        if isinstance(self.v, float):
            self.v = torch.full_like(x, fill_value=self.v)
        if self.decay_input:
            self.v = PassiveDendCompartment.single_step_decay_input(
                self.v, x, self.v_rest, self.tau
            )
        else:
            self.v = PassiveDendCompartment.single_step_not_decay_input(
                self.v, x, self.v_rest, self.tau
            )
        return self.v

    def multi_step_forward(self, x_seq: torch.Tensor):
        print("multi_step_forward of PassiveDendCompartment")
        input_shape = x_seq.shape
        T = input_shape[0]
        self._init_tau_matrix(T)

        # x_seq
        tau_matrix = self._tau_matrix.view(
            T, T, *[1 for _ in range(len(input_shape) - 1)]
        )
        if self.decay_input:
            x_seq = x_seq / self.tau
        y1 = torch.sum(tau_matrix * x_seq.unsqueeze(1), dim=0) # [T, ...]

        # v_init
        v_init = self.v
        if isinstance(v_init, float):
            v_init = torch.full_like(x_seq, fill_value=v_init)
        tau_vector_for_v_init = self._tau_vector_for_v_init.view(
            -1, *[1 for _ in range(len(input_shape) - 1)]
        )
        y2 = tau_vector_for_v_init * v_init

        # v_rest
        v_rest = self.v_rest
        if isinstance(v_rest, float):
            v_rest = torch.full_like(x_seq, fill_value=v_rest)
        tau_vector_for_v_rest = self._tau_vector_for_v_rest.view(
            -1, *[1 for _ in range(len(input_shape) - 1)]
        )
        y3 = tau_vector_for_v_rest * v_rest

        y = y1 + y2 + y3
        if self.store_v_seq:
            self.v_seq = y
        self.v = y[-1]
        return y


class PAComponentDendCompartment(BaseDendCompartment):
    """Dendritic compartment with passive and active voltage components.

    The passive component acts just like a leaky integrator without a firing
    mechanism, while the active voltage component is a function of the passive 
    voltage component. The overall voltage is the sum of active and passive 
    components: 
        v[t] = va[t] + vp[t] = f_dca(vp[t]) + vp[t]
    Get inspiration from:
    Legenstein, R., & Maass, W. (2011). Branch-specific plasticity enables 
    self-organization of nonlinear computation in single neurons. The Journal 
    of Neuroscience: The Official Journal of the Society for Neuroscience, 
    31(30), 10787–10802. https://doi.org/10.1523/JNEUROSCI.5684-10.2011


    Attributes:
        v (Union[float, torch.Tensor]): voltage of the dendritic compartment(s)
            at the current time step.
        va (Union[float, torch.Tensor]): the active component of the 
            compartmental voltage at the current time step.
        vp (Union[float, torch.Tensor]): the passive component of the 
            compartmental voltage at the current time step.
        step_mode (str): "s" for single-step mode, and "m" for multi-step mode.
        store_v_seq (bool): whether to store the compartmental potential at 
            every time step when using multi-step mode. If True, there is 
            another attribute called v_seq.
        store_vp_seq (bool): whether to store the passive component of the 
            compartmental potential at every time step when using multi-step 
            mode. If True, there is another attribute called vp_seq.
        store_va_seq (bool): whether to store the active component of the 
            compartmental potential at every time step when using multi-step 
            mode. If True, there is another attribute called va_seq.
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
        f_dca: Callable = lambda x: 0., step_mode: str = "s", 
        store_v_seq: bool = False, store_vp_seq: bool = False, 
        store_va_seq: bool = False
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
            store_v_seq (bool, optional): whether to store the compartmental 
                potential at every time step when using multi-step mode. 
                Defaults to False.
            store_vp_seq (bool, optional): whether to store the passive 
                component of the compartmental potential at every time step when 
                using multi-step mode. Defaults to False.
            store_v_seq (bool, optional): whether to store the active component 
                of the compartmental potential at every time step when using 
                multi-step mode. Defaults to False.
        """
        super().__init__(v_rest, step_mode, store_v_seq)
        self.tau = tau
        self.decay_input = decay_input
        self.v_rest = v_rest
        self.f_dca = f_dca
        self.register_memory("va", 0.)
        self.register_memory("vp", v_rest)
        self.store_vp_seq = store_vp_seq
        self.store_va_seq = store_va_seq

    @property
    def store_vp_seq(self) -> bool:
        return self._store_vp_seq

    @store_vp_seq.setter
    def store_vp_seq(self, val: bool):
        self._store_vp_seq = val
        if val and (not hasattr(self, "vp_seq")):
            self.register_memory("vp_seq", None)

    @property
    def store_va_seq(self) -> bool:
        return self._store_va_seq

    @store_va_seq.setter
    def store_va_seq(self, val: bool):
        self._store_va_seq = val
        if val and (not hasattr(self, "va_seq")):
            self.register_memory("va_seq", None)

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

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        if self.store_vp_seq:
            vp_seq = []
        if self.store_va_seq:
            va_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)
            if self.store_vp_seq:
                vp_seq.append(self.vp)
            if self.store_va_seq:
                va_seq.append(self.va)
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        if self.store_vp_seq:
            self.vp_seq = torch.stack(vp_seq)
        if self.store_va_seq:
            self.va_seq = torch.stack(va_seq)
        return torch.stack(y_seq)
