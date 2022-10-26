from typing import Union, Optional
import abc

import torch
import torch.nn as nn

from dendsn import dend_compartment, operations
from dendsn import wiring as wr


class BaseDend(nn.Module, abc.ABC):

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

    def extend_external_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tool function 1.
        Extend tensor x with shape [..., self.wiring.n_input]
        to shape [..., self.wiring.n_compartment] using trailing zeros.

        Args:
            x (torch.Tensor): a tensor with shape [..., self.wiring.n_input]

        Returns:
            A torch.Tensor with shape [..., self.wiring.n_compartment]
        """
        x_external = torch.zeros(*x.shape[:-1], self.wiring.n_compartment)
        x_external[..., :self.wiring.n_input] = x
        return x_external

    def get_output(
        self, v_compartment: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Tool function 2.
        Use slicing mechanism to get the output from the 
        compartmental voltage tensor.

        Args:
            v_compartment (torch.Tensor, optional): 
                shape=[..., self.wiring.n_compartment]

        Returns:
            torch.Tensor: shape=[..., self.wiring.n_output]
        """
        if v_compartment is None:
            return self.compartment.v[..., self.wiring.output_index]
        return v_compartment[..., self.wiring.output_index]

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
        A dendrite model with segregated wiring diagram.
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
        if not self.segregation_check:
            raise ValueError(
                f"The dendritic wiring should be an instance of"
                f"wiring.SegregatedDendWiring, but get {self.wiring} instead."
            )
        super().__init__(compartment, wiring, step_mode)

    def segregation_check(self):
        return isinstance(self.wiring, wr.SegregatedDendWiring)

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
        super().__init__(compartment, wiring, step_mode)
        self.coupling_strength = coupling_strength

    def get_internal_input(self) -> torch.Tensor:
        v = self.compartment.v # v.shape = [..., n_comp]
        input_internal = operations.diff_mask_mult_sum(
            x1 = v, x2 = v, mask = self.wiring.adjacency_matrix,
            factor = self.coupling_strength
        )
        return input_internal

    def get_input_to_compartment(self, x: torch.Tensor) -> torch.Tensor:
        input_external = self.extend_external_input(x)
        self.compartment.v_float2tensor(input_external)
        input_internal = self.get_internal_input()
        input = input_external + input_internal
        return input

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        input = self.get_input_to_compartment(x)
        v_compartment = self.compartment.single_step_forward(input)
        return self.get_output(v_compartment)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        T = x_seq.shape[0]
        y_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
        return torch.stack(y_seq)