"""Dendritic models.

Taking the dendritic compartment dynamics and the dendritic wiring diagram into
consideration, we can define a dendritic model, compute the compartmental
voltage step by step given the external input (synaptic / backward), and yield 
the signals fed to the soma. Since we assume that all the neurons in the same 
layer have the same morphology, we only have to specify the dendritic model of a
single neuron in the layer. That saves a large amount of space!!!
"""

from typing import Union, Optional
import abc

import torch
from spikingjelly.activation_based import base

from dendsn import functional
from dendsn.model import dend_compartment
from dendsn.model import wiring as wr


class BaseDend(base.MemoryModule, abc.ABC):

    def __init__(
        self, compartment: dend_compartment.BaseDendCompartment,
        wiring: wr.BaseWiring, 
        step_mode: str = "s"
    ):
        super().__init__()
        self.compartment = compartment
        self.wiring = wiring
        self.step_mode = step_mode

    def reset(self):
        self.compartment.reset()

    def get_output(
        self, v_compartment: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get the voltages of the output compartments.

        Use slicing mechanism to get the voltages of the output compartments
        from the whole compartmental voltage tensor. If the argument 
        v_compartment is given, the result is extracted from it; otherwise, the
        result is extracted from self.compartment.v .

        Args:
            v_compartment (torch.Tensor, optional): 
                shape=[..., self.wiring.n_compartment]

        Returns:
            torch.Tensor: shape=[..., self.wiring.n_output]
        """
        if v_compartment is None:
            return self.compartment.v[..., self.wiring.output_index]
        return v_compartment[..., self.wiring.output_index]

    def extend_external_input(self, x: torch.Tensor) -> torch.Tensor:
        """Extend external input to fit the shape of dendritic compartments.

        Extend tensor x with shape [..., dim] to shape 
        [..., self.wiring.n_compartment] using trailing zeros,
        where dim <= self.wiring.n_compartment .

        Args:
            x (torch.Tensor): a tensor with shape [..., dim], where
            dim <= self.wiring.n_compartment .

        Returns:
            A torch.Tensor with shape [..., self.wiring.n_compartment]
        """
        x_external = torch.zeros(*x.shape[:-1], self.wiring.n_compartment)
        x_external[..., :self.wiring.n_input] = x
        return x_external

    @abc.abstractmethod
    def get_internal_input(self) -> torch.Tensor:
        """Compute the internal input to the dendritic compartments.

        Dendritic compartments may receive signals from their adjacent
        compartments, which is referred to as internal inputs. This function
        calculates internal inputs to all the compartments, given the states of
        these compartments and the wiring diagram (stored in self). Implement
        this method while defining a new subclass!

        Returns:
            torch.Tensor: internal input with shape 
                [..., self.wiring.n_compartment]
        """
        pass

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = [N, *soma_shape, self.wiring.n_input]
        input_external = self.extend_external_input(x)
        # input_external.shape = [N, *soma_shape, self.wiring.n_compartment]
        self.compartment.v_float2tensor(input_external)
        input_internal = self.get_internal_input()
        v_compartment = self.compartment.single_step_forward(
            input_external + input_internal
        )
        return self.get_output(v_compartment)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x.shape = [T, N, *soma_shape, self.wiring.n_input]
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
                f"BaseDend.step_mode should be 'm' or 's', "
                f"but get {self.step_mode} instead."
            )


class SegregatedDend(BaseDend):

    def __init__(
        self, compartment: dend_compartment.BaseDendCompartment,
        wiring: wr.SegregatedDendWiring,
        step_mode: str = "s"
    ):
        """
        A dendrite model (for a single neuron )with segregated wiring diagram.
        Since wiring is an instance of SegregatedDendWiring
        (n_compartment = n_input = n_output, empty adjacency matrix),
        the computation can be much simpler than that defined in BaseDend.

        Args:
            compartment (dend_compartment.BaseDendCompartment)
            wiring (wiring.SegregatedDendWiring)
            step_mode (str, optional): Defaults to "s".

        Raises:
            ValueError: when `wiring` is not 
                an instance of SegregatedDendWiring.
        """
        if not isinstance(self.wiring, wr.SegregatedDendWiring):
            raise ValueError(
                f"The dendritic wiring should be an instance of"
                f"wiring.SegregatedDendWiring, but get {self.wiring} instead."
            )
        super().__init__(compartment, wiring, step_mode)

    def get_internal_input(self, x: torch.Tensor) -> torch.Tensor:
        return 0

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        v_compartment = self.compartment.single_step_forward(x)
        return v_compartment

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        v_compartment_seq = self.compartment.multi_step_forward(x_seq)
        return v_compartment_seq


class VDiffDend(BaseDend):

    def __init__(
        self, compartment: dend_compartment.BaseDendCompartment,
        wiring: wr.BaseWiring, 
        coupling_strength: Union[float, torch.Tensor] = 1.,
        step_mode: str = "s" 
    ):
        """
        A dendrite model (for a single neuron) whose internal information flow 
        is determined by the difference in compartmental voltage.

        Args:
            compartment (dend_compartment.BaseDendCompartment):
                the dendritic compartment dynamics model
            wiring (wr.BaseWiring): the dendritic wiring diagram
            coupling_strength (Union[float, torch.Tensor], optional): 
                the coupling strength corresponding to the connections 
                in `wiring`. If it's a torch.Tensor, its shape should be 
                [wiring.n_compartment, wiring.n_compartment]. 
                If it's a float, the same coupling strength is applied to all 
                the connections. Defaults to 1. .
            step_mode (str, optional): Defaults to "s".
        """
        super().__init__(compartment, wiring, step_mode)
        self.coupling_strength = coupling_strength

    def get_internal_input(self) -> torch.Tensor:
        v = self.compartment.v # v.shape = [..., n_comp]
        input_internal = functional.diff_mask_mult_sum(
            x1 = v, x2 = v, mask = self.wiring.adjacency_matrix,
            factor = self.coupling_strength
        )
        return input_internal