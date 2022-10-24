import torch
import torch.nn as nn

from dendsj import dend_compartment, wiring


class BaseDend(nn.Module):

    def __init__(
        self, compartment: dend_compartment.BaseDendCompartment,
        wiring: wiring.BaseWiring,
    ):
        self.compartment = compartment
        self.wiring = wiring
        self.step_mode = compartment.step_mode

    def reset(self):
        self.compartment.reset()

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        pass

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
