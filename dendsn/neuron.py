import abc
from typing import Union, Optional, List
from numpy import isin

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron as sj_neuron

from dendsn import dend_soma_conn, dendrite, functional


class BaseDendNeuron(nn.Module, abc.ABC):

    def __init__(
        self, dend: dendrite.BaseDend, soma: sj_neuron.BaseNode,
        dend_soma_conn: dend_soma_conn.BaseDendSomaConn,
        step_mode: str = "s"
    ):
        super().__init__()
        self.dend = dend
        self.soma = soma
        self.dend_soma_conn = dend_soma_conn
        self.step_mode = step_mode

    def reset(self):
        self.dend.reset()
        self.soma.reset()

    def v_soma_float2tensor_by_shape_ints(self, *shape):
        if isinstance(self.soma.v, float):
            v_init = self.soma.v
            self.soma.v = torch.full(size = shape, fill_value = v_init)

    def v_soma_float2tensor_by_shape_list(self, shape):
        self.v_soma_float2tensor_by_shape_ints(*shape)

    def v_soma_float2tensor_by_tensor(self, x: torch.Tensor):
        self.soma.v_float_to_tensor(x)

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
                f"BaseDendNeuron.step_mode should be 'm' or 's', "
                f"but get {self.step_mode} instead."
            )


class VForwardDendNeuron(BaseDendNeuron):

    def __init__(
        self, dend: dendrite.BaseDend, soma: sj_neuron.BaseNode, 
        dend_soma_conn: dend_soma_conn.BaseDendSomaConn,
        forward_strength: Union[float, torch.Tensor] = 1.,
        step_mode: str = "s"
    ):
        super().__init__(dend, soma, dend_soma_conn, step_mode)
        self.forward_strength = forward_strength

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        v2soma = self.dend.single_step_forward(x)
        self.v_soma_float2tensor_by_shape_ints(
            *v2soma.shape[:-1], self.dend_soma_conn.n_soma
        )
        v_soma = self.soma.v
        input2soma = functional.diff_mask_mult_sum(
            x1 = v2soma, x2 = v_soma,
            mask = self.dend_soma_conn.forward_adjacency_matrix,
            factor = self.forward_strength
        )
        # input2soma.shape = [..., n_soma]
        soma_spike = self.soma.single_step_forward(input2soma)
        return soma_spike

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        v2soma_seq = self.dend.multi_step_forward(x_seq)
        # v2soma_seq.shape = [T, ..., n_output_compartment]
        T = x_seq.shape[0]
        soma_spike_seq = []
        self.v_soma_float2tensor_by_shape_ints(
            *v2soma_seq[0].shape[:-1], self.dend_soma_conn.n_soma
        )
        for t in range(T):
            v_soma = self.soma.v
            input2soma = functional.diff_mask_mult_sum(
                x1 = v2soma_seq[t], x2 = v_soma,
                mask = self.dend_soma_conn.forward_adjacency_matrix,
                factor = self.forward_strength
            )
            soma_spike = self.soma.single_step_forward(input2soma)
            soma_spike_seq.append(soma_spike)
        return torch.stack(soma_spike_seq)


class VForwardSBackwardDendNeuron(BaseDendNeuron):

    def __init__(
        self, dend: dendrite.BaseDend, soma: sj_neuron.BaseNode, 
        dend_soma_conn: dend_soma_conn.BaseDendSomaConn,
        forward_strength: Union[float, torch.Tensor] = 1.,
        backward_strength: Union[float, torch.Tensor] = 1.,
        step_mode: str = "s"
    ):
        super().__init__(dend, soma, dend_soma_conn, step_mode)
        self.forward_strength = forward_strength
        self.backward_strength = backward_strength

    def soma_spike_backprop(self, soma_spike: torch.Tensor):
        # soma_spike.shape = [..., n_soma]
        conn = (self.dend_soma_conn.backward_adjacency_matrix 
                * self.backward_strength)
        bp_v = soma_spike @ conn # [..., n_output_compartment]
        self.dend.compartment.v[..., self.dend.wiring.output_index] += bp_v

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        v2soma = self.dend.single_step_forward(x)
        self.v_soma_float2tensor_by_shape_ints(
            *v2soma.shape[:-1], self.dend_soma_conn.n_soma
        )
        v_soma = self.soma.v
        input2soma = functional.diff_mask_mult_sum(
            x1 = v2soma, x2 = v_soma,
            mask = self.dend_soma_conn.forward_adjacency_matrix,
            factor = self.forward_strength
        )
        soma_spike = self.soma.single_step_forward(input2soma)
        self.soma_spike_backprop(soma_spike)
        return soma_spike

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        T = x_seq.shape[0]
        soma_spike_seq = []
        for t in range(T):
            soma_spike = self.single_step_forward(x_seq[t])
            soma_spike_seq.append(soma_spike)
        return torch.stack(soma_spike_seq)