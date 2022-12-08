import abc

import torch
from spikingjelly.activation_based import base
from spikingjelly.activation_based import neuron


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


class LISynapseFilter(BaseSynapseFilter):

    def __init__(self, tau: float, step_mode: str = "s"):
        super().__init__(step_mode)
        self.tau = tau
        self.leaky_integrator = neuron.LIFNode(
            tau = tau, decay_input = False, v_threshold = float("inf"),
            step_mode = step_mode
        )

    def reset(self):
        self.leaky_integrator.reset()

    @property
    def state(self) -> torch.Tensor:
        return self.leaky_integrator.v

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        self.leaky_integrator.single_step_forward(x)
        return self.state

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        self.leaky_integrator.store_v_seq = True
        self.leaky_integrator.multi_step_forward(x_seq)
        self.leaky_integrator.store_v_seq = False
        return self.v_seq