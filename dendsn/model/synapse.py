from typing import Union

import torch
import torch.nn as nn
import torch.nn.common_types as ttypes

from dendsn.model import synapse_conn, synapse_filter


class BaseSynapse(nn.Module):

    def __init__(
        self, conn: synapse_conn.BaseSynapseConn, 
        filter: synapse_filter.BaseSynapseFilter,
        step_mode: str = "s",
    ):
        """
        A generic synapse model consists of a synaptic connection model and 
        a filter.

        Args:
            conn (synapse_conn.BaseSynapseConn)
            filter (synapse_filter.BaseSynapseFilter)
            step_mode (str, optional): Defaults to "s".
        """
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


class LinearIdenditySynapse(BaseSynapse):

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False,
        device = None, dtype = None, step_mode: str = "s"
    ):
        """
        A synapse model whose connection model is a LinearSynapseConn
        and the synaptic filter is an IdentitySynapseFilter

        Args:
            in_features (int): the argument for MaskedLinearSynapseConn.
            out_features (int): the argument for MaskedLinearSynapseConn.
            bias (bool, optional): the argument for MaskedLinearSynapseConn. 
            device (_type_, optional): Defaults to None.
            dtype (_type_, optional): Defaults to None.
            step_mode (str, optional): Defaults to "s".
        """
        super().__init__(
            conn = synapse_conn.LinearSynapseConn(
                in_features, out_features, bias, device, dtype
            ),
            filter = synapse_filter.IdentitySynapseFilter(),
            step_mode = step_mode
        )


class MaskedLinearIdenditySynapse(BaseSynapse):

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False,
        init_sparsity: float = 0.75, device = None, dtype = None,
        step_mode: str = "s"
    ):
        """
        A synapse model whose connection model is a MaskedLinearSynapseConn
        and synaptic filter is an IdentitySynapseFilter

        Args:
            in_features (int): the argument for MaskedLinearSynapseConn.
            out_features (int): the argument for MaskedLinearSynapseConn.
            bias (bool, optional): the argument for MaskedLinearSynapseConn. 
            init_sparsity (float, optional): the argument for MaskedLinearSynapseConn.
                the sparsity of the 0-1 mask when it is initialized 
                [higher -> sparser]. Defaults to False. Defaults to 0.75.
            device (_type_, optional): Defaults to None.
            dtype (_type_, optional): Defaults to None.
            step_mode (str, optional): Defaults to "s".
        """
        super().__init__(
            conn = synapse_conn.MaskedLinearSynapseConn(
                in_features, out_features, bias, init_sparsity, device, dtype
            ),
            filter = synapse_filter.IdentitySynapseFilter(),
            step_mode = step_mode
        )


class Conv2dIdentitySynapse(BaseSynapse):

    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: ttypes._size_2_t, stride: ttypes._size_2_t = 1,
        padding: Union[ttypes._size_2_t, str] = 0,
        dilation: ttypes._size_2_t = 1, groups: int = 1,
        bias: bool = False, padding_mode: str = "zeros", 
        device = None, dtype = None, step_mode: str = "s"
    ):
        super().__init__(
            conn = synapse_conn.Conv2dSynapseConn(
                in_channels, out_channels, kernel_size, stride, padding, 
                dilation, groups, bias, padding_mode, device, dtype,
            ),
            filter = synapse_filter.IdentitySynapseFilter(),
            step_mode = step_mode
        )