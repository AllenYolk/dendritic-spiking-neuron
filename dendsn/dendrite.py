from typing import Union

import torch
import torch.nn as nn

from dendsn import dend_compartment, wiring


class BaseDend(nn.Module):

    def __init__(
        self, compartment: dend_compartment.BaseDendCompartment,
        wiring: wiring.BaseWiring, 
        coupling_strength: Union[float, torch.Tensor] = 1.,
        step_mode: str = "s"
    ):
        self.compartment = compartment
        self.wiring = wiring
        self.coupling_strength = coupling_strength
        self.step_mode = step_mode

    def reset(self):
        self.compartment.reset()

    def extend_external_input(self, x: torch.Tensor) -> torch.Tensor:
        """
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

    def get_output(self, output: torch.Tenosr) -> torch.Tensor:
        return output[..., self.wiring.output_index]

    def get_internal_input(self) -> torch.Tensor:
        """
        Get the input to the dendritic compartments caused by the gap of 
        voltages compared with adjacent compartments (or other internal sources).

        Returns:
            torch.Tensor: a tensor with shape [..., self.wiring.n_compartment]
        """
        v = self.compartment.v # v.shape = [..., n_comp]
        v_gap = v.unsqueeze(-1) - v.unsqueeze(-2) # [..., n_comp, n_comp]
        v_gap = v_gap * self.wiring.adjacency_matrix
        input_internal = (v_gap * self.coupling_strength).sum(dim = -2)
        return input_internal

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = [..., self.wiring.n_input]
        input_external = self.extend_external_input(x)
        self.compartment.v_float2tensor(input_external)
        input_internal = self.get_internal_input()
        input = input_external + input_internal
        v_compartment = self.compartment.single_step_forward(input)
        return self.get_output(v_compartment)

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
                f"BaseDend.step should be 'm' or 's', "
                "but get {self.step_mode} instead."
            )