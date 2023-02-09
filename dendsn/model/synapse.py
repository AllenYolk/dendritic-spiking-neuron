"""Synaptic Models.

A synaptic model consists of a synaptic connection layer and a synaptic filter
model.
"""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.common_types as ttypes

from dendsn.model import synapse_conn, synapse_filter


class BaseSynapse(nn.Module):
    """Base class for all synaptic models.

    A generic synapse model consists of a synaptic connection model and 
    a filter. The design pattern here is: Template Method, Factory Method and 
    Abstract Factory.

    Args:
        conn (BaseSynapseConn): the synaptic connection module.
        filter (BaseSynapseFilter): the synaptic filter model.
        step_mode (str): "s" for single-step mode, and "m" for multi-step mode.
    """

    def __init__(
        self, conn: synapse_conn.BaseSynapseConn, 
        filter: synapse_filter.BaseSynapseFilter,
        step_mode: str = "s",
    ):
        """The constructor of BaseSynapse.

        Args:
            conn (BaseSynapseConn): the synaptic connection module.
            filter (BaseSynapseFilter): the synaptic filter model.
            step_mode (str, optional): "s" for single-step mode, and "m" for
                multi-step mode. Defaults to "s".
        """
        super().__init__()
        self.conn = conn
        self.filter = filter
        self.step_mode = step_mode

    def reset(self):
        self.filter.reset()

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        self.conn.step_mode = "s"
        z = self.conn.forward(x)
        self.filter.step_mode = "s"
        y = self.filter.forward(z)
        return y

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        self.conn.step_mode = "m"
        z_seq = self.conn.forward(x_seq)
        self.filter.step_mode = "m"
        y_seq = self.filter.forward(z_seq)
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


class LinearIdentitySynapse(BaseSynapse):
    """Synaptic model combining LinearSynapseConn with IdentitySynapseFilter.

    Attributes:
        See base class BaseSynapse.
        Also, see LinearSynapseConn nd IdentitySynapseFilter.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False,
        device = None, dtype = None, step_mode: str = "s"
    ):
        """The constructor of LinearIdentitySynapse.

        Args:
            in_features (int): the argument for LinearSynapseConn.
            out_features (int): the argument for LinearSynapseConn.
            bias (bool, optional): the argument for LinearSynapseConn. Defaults
                to None.
            device (str, optional): Defaults to None.
            dtype (str, optional): Defaults to None.
            step_mode (str, optional): "s" for single-step mode, and "m" for 
                multi-step mode. Defaults to "s".
        """
        super().__init__(
            conn = synapse_conn.LinearSynapseConn(
                in_features, out_features, bias, device, dtype
            ),
            filter = synapse_filter.IdentitySynapseFilter(),
            step_mode = step_mode
        )


class MaskedLinearIdentitySynapse(BaseSynapse):
    """Synapse combining MaskedLinearSynapseConn and IdentitySynapseFilter.

    Attributes:
        See base class: BaseSynapse.
        Also, see MaskedLinearSynapseConn and IdentitySynapseFilter.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False,
        init_sparsity: float = 0.75, device = None, dtype = None,
        step_mode: str = "s"
    ):
        """The constructor of MaskedLinearIdentitySynapse.

        Args:
            in_features (int): the argument for MaskedLinearSynapseConn.
            out_features (int): the argument for MaskedLinearSynapseConn.
            bias (bool, optional): the argument for MaskedLinearSynapseConn.
                Defaults to None.
            init_sparsity (float, optional): the argument for 
                MaskedLinearSynapseConn. The sparsity of the 0-1 mask when it is
                initialized [higher -> sparser]. Defaults to 0.75.
            device (str, optional): Defaults to None.
            dtype (str, optional): Defaults to None.
            step_mode (str, optional): "s" for single-step mode, and "m" for
                multi-step mode. Defaults to "s".
        """
        super().__init__(
            conn = synapse_conn.MaskedLinearSynapseConn(
                in_features, out_features, bias, init_sparsity, device, dtype
            ),
            filter = synapse_filter.IdentitySynapseFilter(),
            step_mode = step_mode
        )


class Conv2dIdentitySynapse(BaseSynapse):
    """Synapse combining Conv2dSynapseConn and IdentitySynapseFilter.

    Attributes:
        See base class: BaseSynapse.
        Also, see Conv2dSynapseConn and IdentitySynapseFilter.
    """

    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: ttypes._size_2_t, stride: ttypes._size_2_t = 1,
        padding: Union[ttypes._size_2_t, str] = 0,
        dilation: ttypes._size_2_t = 1, groups: int = 1,
        bias: bool = False, padding_mode: str = "zeros", 
        device = None, dtype = None, step_mode: str = "s"
    ):
        """The constructor of Conv2dIdentitySynapse.

        Args:
            in_channels (int): the argument for Conv2dSynapseConn.
            out_channels (int): the argument for Conv2dSynapseConn.
            kernel_size (ttypes._size_2_t): the argument for Conv2dSynapseConn.
            stride (ttypes._size_2_t, optional): the argument for 
                Conv2dSynapseConn. Defaults to 1.
            padding (Union[ttypes._size_2_t, str], optional): the argument for 
                Conv2dSynapseConn.. Defaults to 0.
            dilation (ttypes._size_2_t, optional): the argument for
                Conv2dSynapseConn. Defaults to 1.
            groups (int, optional): the argument for Conv2dSynapseConn. 
                Defaults to 1.
            bias (bool, optional): the argument for Conv2dSynapseConn. Defaults 
                to False.
            padding_mode (str, optional): the argument for Conv2dSynapseConn.
                Defaults to "zeros".
            device (_type_, optional): Defaults to None.
            dtype (_type_, optional): Defaults to None.
            step_mode (str, optional): "s" for single-step mode, and "m" for
                multi-step mode. Defaults to "s".
        """
        super().__init__(
            conn = synapse_conn.Conv2dSynapseConn(
                in_channels, out_channels, kernel_size, stride, padding, 
                dilation, groups, bias, padding_mode, device, dtype,
            ),
            filter = synapse_filter.IdentitySynapseFilter(),
            step_mode = step_mode
        )