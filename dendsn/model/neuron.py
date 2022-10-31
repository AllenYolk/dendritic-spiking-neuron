"""
This module defines a series of dendritic neuron layer models (rather than 
single neurons). Each layer consists of a set of dendritic neurons with exactly
the same dendritic morphology.
"""

import abc
from typing import Union, List

import torch
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import neuron as sj_neuron

from dendsn.model import dendrite


class BaseDendNeuron(nn.Module, abc.ABC):

    def __init__(
        self, dend: dendrite.BaseDend, soma: sj_neuron.BaseNode,
        soma_shape: List[int],
        step_mode: str = "s"
    ):
        super().__init__()
        self.dend = dend
        self.soma = soma
        self._soma_shape = soma_shape
        self._n_soma = np.prod(soma_shape)
        self.step_mode = step_mode

    @property
    def soma_shape(self) -> List[int]:
        return self._soma_shape

    @property
    def n_soma(self) -> int:
        return self._n_soma

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
        soma_shape: List[int],
        forward_strength: Union[float, torch.Tensor] = 1.,
        step_mode: str = "s"
    ):
        """
        A layer of dendritic neurons. The input to the soma is determined by the
        voltage difference between the soma and the output compartments.

        Args:
            dend (dendrite.BaseDend): the dendrite model (for a single neuron)
            soma (sj_neuron.BaseNode): the soma model (a point-neuron model)
            soma_shape (List[int]): the shape of somatic voltage tensor. Hence, 
                there are np.prod(soma_shape) neurons in this layer.
            forward_strength (Union[float, torch.Tensor], optional): 
                the coupling strength of the feed-forward connections from the
                output compartments to the soma (defined for a single neuron). 
                If it's a torch.Tensor, its shape should be 
                [dend.wiring.n_output, 1]. If it's a float, the same strength
                will be applied to all the connections. Defaults to 1. .
            step_mode (str, optional): Defaults to "s".
        """
        super().__init__(dend, soma, soma_shape, step_mode)
        self.forward_strength = forward_strength

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = [N, self.n_soma * self.dend.wiring.n_input]
        N = x.shape[0]
        x = x.reshape([N, *self.soma_shape, -1])
        v2soma = self.dend.single_step_forward(x)
        # v2soma.shape = [N, *self.soma_shape, self.dend.wiring.n_output]
        self.v_soma_float2tensor_by_shape_ints(*v2soma.shape[:-1])
        v_soma = self.soma.v
        # v_soma.shape = [N, *self.soma_shape]
        input2soma = (v2soma - v_soma.unsqueeze(-1)) @ self.forward_strength
        input2soma = input2soma.squeeze(dim = -1)
        # input2soma.shape = [N, *self.soma_shape]
        soma_spike = self.soma.single_step_forward(input2soma)
        return soma_spike.reshape([N, -1])

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq = [T, N, self.n_soma * self.dend.wiring.n_input]
        T, N = x_seq.shape[0], x_seq.shape[1]
        x_seq = x_seq.reshape([T, N, *self.soma_shape, -1])
        v2soma_seq = self.dend.multi_step_forward(x_seq)
        # v2soma_seq.shape = [T, N, *self.soma_shape, self.dend.wiring.n_output]
        soma_spike_seq = []
        self.v_soma_float2tensor_by_shape_ints(*v2soma_seq[0].shape[:-1])
        for t in range(T):
            v_soma = self.soma.v
            # v_soma.shape = [N, *self.soma_shape]
            input2soma = v2soma_seq[t] - v_soma.unsqueeze(-1)
            input2soma = (input2soma @ self.forward_strength).squeeze(dim = -1)
            # input2soma.shape = [N, *self.soma_shape]
            soma_spike = self.soma.single_step_forward(input2soma)
            soma_spike_seq.append(soma_spike.reshape(N, -1))
        return torch.stack(soma_spike_seq)


class VForwardSBackwardDendNeuron(BaseDendNeuron):

    def __init__(
        self, dend: dendrite.BaseDend, soma: sj_neuron.BaseNode, 
        soma_shape: List[int],
        forward_strength: Union[float, torch.Tensor] = 1.,
        backward_strength: Union[float, torch.Tensor] = 1.,
        step_mode: str = "s"
    ):
        """
        A layer of dendritic neurons. The input to the soma is determined by the
        voltage difference between the soma and the output compartments. If 
        a spike is generated in the soma, a back-propagated voltage will be 
        added the the ouput compartments.

        Args:
            dend (dendrite.BaseDend): the dendrite model (for a single neuron)
            soma (sj_neuron.BaseNode): the soma model (a point-neuron model)
            soma_shape (List[int]): the shape of somatic voltage tensor. Hence, 
                there are np.prod(soma_shape) neurons in this layer.
            forward_strength (Union[float, torch.Tensor], optional): 
                the coupling strength of the feed-forward connections from the
                output compartments to the soma (defined for a single neuron). 
                If it's a torch.Tensor, its shape should be 
                [dend.wiring.n_output, 1]. If it's a float, the same strength
                will be applied to all the connections. Defaults to 1..
            backward_strength (Union[float, torch.Tensor], optional):
                the coupling strength of the feedback connections from the
                soma to the output compartments (defined for a single neuron). 
                If it's a torch.Tensor, its shape should be 
                [1, dend.wiring.n_output]. If it's a float, the same strength
                will be applied to all the connections. Defaults to 1..
            step_mode (str, optional): _description_. Defaults to "s".
        """
        super().__init__(dend, soma, soma_shape, step_mode)
        self.forward_strength = forward_strength
        self.backward_strength = backward_strength

    def soma_spike_backprop(self, soma_spike: torch.Tensor):
        # soma_spike.shape = [N, *self.soma_shape]
        conn = torch.ones(size = [1, self.dend.wiring.n_output])
        conn = conn * self.backward_strength
        bp_v = soma_spike.unsqueeze(-1) @ conn 
        # bp_v = [N, *self.soma_shape, self.dend.wiring.n_output]
        self.dend.compartment.v[..., self.dend.wiring.output_index] += bp_v

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        x = x.reshape([N, *self.soma_shape, -1])
        v2soma = self.dend.single_step_forward(x)
        # v2soma.shape = [N, *self.soma_shape, self.dend.wiring.n_output]
        self.v_soma_float2tensor_by_shape_ints(*v2soma.shape[:-1])
        v_soma = self.soma.v
        # v_soma.shape = [N, *self.soma_shape]
        input2soma = (v2soma - v_soma.unsqueeze(-1)) @ self.forward_strength
        input2soma = input2soma.squeeze(dim = -1)
        # input2soma.shape = [N, *self.soma_shape]
        soma_spike = self.soma.single_step_forward(input2soma)
        self.soma_spike_backprop(soma_spike)
        return soma_spike.reshape([N, -1])

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        T = x_seq.shape[0]
        soma_spike_seq = []
        for t in range(T):
            soma_spike = self.single_step_forward(x_seq[t])
            soma_spike_seq.append(soma_spike)
        return torch.stack(soma_spike_seq)