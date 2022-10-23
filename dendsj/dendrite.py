import torch
from spikingjelly.activation_based import base


class DendCompartment(base.MemoryModule):

    def __init__(
        self, tau: float = 2, decay_input: bool =  True, v_rest: float = 0.,
        step_mode: str = "s"
    ):
        super().__init__()
        self.register_memory("v", 0.)
        self.tau = tau
        self.decay_input = decay_input
        self.v_rest = v_rest
        self.step_mode = step_mode

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
            self.v = DendCompartment.single_step_decay_input(
                self.v, x, self.v_rest, self.tau
            )
        else:
            self.v = DendCompartment.single_step_not_decay_input(
                self.v, x, self.v_rest, self.tau
            )
        return self.v

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        v_seq = []
        for t in range(T):
            v = self.single_step_forward(x_seq[t])
            v_seq.append(v)
        return torch.stack(v_seq)

    def forward(self, x: torch.Tensor):
        if self.step_mode == "s":
            return self.single_step_forward(x)
        elif self.step_mode == "m":
            return self.multi_step_forward(x)
        else:
            raise ValueError(
                f"DendCompartment.step should be 'm' or 's', "
                "but get {self.step_mode} instead."
            )


if __name__ == "__main__":
    T, N = 10, 2
    x_seq = torch.randn(size = [T, N]) + 0.5

    dend = DendCompartment(step_mode = "m", decay_input = True)
    v_seq = dend(x_seq)

    for t in range(T):
        print(f"t = {t}:")
        print(f"    x = {x_seq[t]}")
        print(f"    v = {v_seq[t]} [decay input]")

    dend.reset()
    dend.decay_input = False
    v_seq = dend(x_seq)

    for t in range(T):
        print(f"t = {t}:")
        print(f"    x = {x_seq[t]}")
        print(f"    v = {v_seq[t]} [not decay input]") 