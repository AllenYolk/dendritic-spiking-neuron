import torch
import torch.nn as nn

from dendsn import synapse_conn, synapse_filter


class BaseSynapse(nn.Module):

    def __init__(
        self, conn: synapse_conn.BaseSynapseConn, 
        filter: synapse_filter.BaseSynapseFilter,
        step_mode: str = "s",
    ):
        super().__init__()
        self.conn = conn
        self.filter = filter
        self.step_mode = step_mode

    def reset(self):
        self.filter.reset()

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conn.single_step_forward(x)
        y = self.filter.single_step_forward(z)
        return y

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        z_seq = self.conn.multi_step_forward(x_seq)
        y_seq = self.filter.multi_step_forward(z_seq)
        return y_seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode == "s":
            return self.single_step_forward(x)
        elif self.step_mode == "m":
            return self.multi_step_forward(x)
        else:
            raise ValueError(
                f"BaseSynapse.step_mode should be 'm' or 's', "
                f"but get {self.step_mode} instead."
            )


class MaskedLinearIdenditySynapse(BaseSynapse):

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False,
        init_sparsity: float = 0.75, device = None, dtype = None,
        step_mode: str = "s"
    ):
        super().__init__(
            conn = synapse_conn.MaskedLinearSynapseConn(
                in_features, out_features, bias, init_sparsity, device, dtype
            ),
            filter = synapse_filter.IdentitySynapseFilter(),
            step_mode = step_mode
        )