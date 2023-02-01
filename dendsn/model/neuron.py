"""Dendritic neuron modules.

This module defines a series of dendritic neuron layer models (rather than 
single neurons). Each layer consists of a set of dendritic neurons with exactly
the same dendritic morphology. A dendritic neuron layer is often placed after an
FC or Conv layer, acting as a activation function.
"""

import abc
from typing import Union, List, Optional

import torch
import numpy as np
from spikingjelly.activation_based import neuron as sj_neuron
from spikingjelly.activation_based import base

from dendsn.model import dendrite


class BaseDendNeuron(base.MemoryModule, abc.ABC):
    """Base class for all dendritic neuron modules.

    This class defines a layer of dendritic neurons with exactly the same 
    dendritic morphology and dynamical characteristics. A single dendritic 
    neuron is composed of a dendritic model, a somatic model, and optionally 
    connective strength between dendrite and soma. The key point is signal
    transmission between the output dendritic compartments and the soma.

    Attributes:
        dend (BaseDend): dendritic model for a single neuron in the layer.
        soma (BaseNode): somatic model considering all the neurons in the layer. 
            Typically an activation-based neuron layer in spikingjelly.
        soma_shape (List[int]): the shape of the somatic voltage tensor in this 
            module. The spatial structure of the neurons in this layer.
        n_soma (int): the number of neurons / somas in the layer.
        step_mode (str): "s" for single-step mode, and "m" for multi-step mode.
    """

    def __init__(
        self, dend: dendrite.BaseDend, soma: sj_neuron.BaseNode,
        soma_shape: List[int],
        step_mode: str = "s"
    ):
        """The constructor of BaseDendNeuron.

        Args:
            dend (BaseDend): dendritic model for a single neuron.
            soma (BaseNode): somatic model considering all the neurons.
            soma_shape (List[int]): the shape of the somatic voltage tensor in
                this module. The spatial structure of the neurons in this layer.
            step_mode (str, optional): "s" for single-step mode, and "m" for 
                multi-step mode.. Defaults to "s".
        """
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
        """If soma.v is a float, turn it into a tensor with specified shape.

        Args:
            shape: several integers describing the shape we want soma.v to be.
        """
        if isinstance(self.soma.v, float):
            v_init = self.soma.v
            self.soma.v = torch.full(size = shape, fill_value = v_init)

    def v_soma_float2tensor_by_shape_list(self, shape):
        """If soma.v is a float, turn it into a tensor with specified shape.

        Args:
            shape (List[int]): a list of integers describing the shape we want
                soma.v to be.
        """
        self.v_soma_float2tensor_by_shape_ints(*shape)

    def v_soma_float2tensor_by_tensor(self, x: torch.Tensor):
        """If soma.V is a float, turn it into a tensor with x's shape.
        """
        self.soma.v_float_to_tensor(x)

    def reshape_input(
        self, x: torch.Tensor, step_mode: Optional[str] = None
    ) -> torch.Tensor:
        """Reshape the input tensor so that BaseDendNeuron can handle it.

        Since a dendritic neuron layer often follows an FC or Conv layer which
        acts as synaptic connections, a typical input tensor to this module has
        shape [batch_size, dim] or [batch_size, n_channels, ...] for 
        single-step mode, and [T, batch_size, dim] or 
        [T, batch_size, n_channels, ...] for multi-step mode. This method folds
        the "dim" or "n_channels" dimension, yields a dimension of size 
        dend.wiring.n_input (now the shape is [batch_size, n_input, *soma_shape]
        or [T, batch_size, n_input, *soma_shape]), and places the new dimension 
        with size n_input at the last position. Consequently, the output has 
        shape [batch_size, *soma_shape, n_input] or 
        [T, batch_size, *soma_shape, n_input] .

        Args:
            x (torch.Tensor): the input tensor to the dendritic neuron module.
                Its shape should be [batch_size, dim] or 
                [batch_size, n_channels, ...] for single-step mode, and
                [T, batch_size, dim] or [T, batch_size, n_channels, ...]
                for multi-step mode.
            step_mode (Optional[str], optional): if None, retrieve its value
                from self.step_mode. Defaults to None.

        Raises:
            ValueError: wrong step_mode value.

        Returns:
            torch.Tensor: with shape [batch_size, *soma_shape, n_input] for
                single-step mode, and [T, batch_sie, *soma_shape, n_input] for
                multi-step mode.
        """
        step_mode = self.step_mode if (step_mode is None) else step_mode
        if step_mode == "s":
            shape_prefix = [x.shape[0]]
        elif step_mode == "m":
            shape_prefix = [x.shape[0], x.shape[1]]
        else:
            raise ValueError(
                f"BaseDendNeuron.step_mode should be 'm' or 's', "
                f"but get {step_mode} instead."
            )

        if len(self.soma_shape) == 1:
            # after a nn.Linear layer
            return x.reshape([*shape_prefix, *self.soma_shape, -1])
        elif len(self.soma_shape) > 1:
            # after a Conv2d layer
            x = x.reshape([*shape_prefix, -1, *self.soma_shape])
            ax = len(shape_prefix)
            perm_trans = [i for i in range(len(x.shape))]
            perm_trans.pop(ax)
            perm_trans.append(ax)
            return x.permute(perm_trans)

    @abc.abstractmethod
    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        T = x_seq.shape[0]
        soma_spike_seq = []
        for t in range(T):
            soma_spike = self.single_step_forward(x_seq[t])
            soma_spike_seq.append(soma_spike)
        return torch.stack(soma_spike_seq)

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


class VDiffForwardDendNeuron(BaseDendNeuron):
    """A layer of neurons that transmit (v_dend - v_soma) forward to the soma.

    The input to the soma is determined by the voltage difference between the 
    soma and the output compartments. The input to soma a in this layer is:
        J_{a} = \sum_i phi_{ia}(v_i - v_a)
    where i can be all output dendritic compartments in neuron a, and phi_{ij} 
    is the feed forward coupling strength.

    Attributes:
        dend, soma, soma_shape, n_soma, step_mode: see BaseDendNeuron.
        forward_strength: the feed forward coupling strength phi from the output
            compartments to the soma, defined for a single neuron, which can be
            a Tensor with shape [dend.wiring.n_output, 1] or a float (can be 
            broadcasted into a Tensor).
    """

    def __init__(
        self, dend: dendrite.BaseDend, soma: sj_neuron.BaseNode,
        soma_shape: List[int],
        forward_strength: Union[float, torch.Tensor] = 1.,
        step_mode: str = "s"
    ):
        """The constructor of VDiffForwardDendNeuron.

        Args:
            dend (BaseDend): dendrite model for a single neuron.
            soma (BaseNode): somatic model considering all the neurons.
            soma_shape (List[int]): the shape of the somatic voltage tensor in
                this module. The spatial structure of the neurons in this layer.
                The batch dimension should not be included in this list.
            forward_strength (Union[float, torch.Tensor], optional): 
                the coupling strength of the feed-forward connections from the
                output compartments to the soma (defined for a single neuron). 
                If it's a torch.Tensor, its shape should be 
                [dend.wiring.n_output, 1]. If it's a float, the same strength
                will be applied to all the connections. Defaults to 1. .
            step_mode (str, optional): "s" for single-step mode, and "m" for 
                multi-step mode. Defaults to "s".
        """
        super().__init__(dend, soma, soma_shape, step_mode)
        self.forward_strength = forward_strength

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = [N, self.n_soma * self.dend.wiring.n_input]
        N = x.shape[0]
        x = self.reshape_input(x, "s")
        v2soma = self.dend.single_step_forward(x)
        # v2soma.shape = [N, *self.soma_shape, self.dend.wiring.n_output]
        self.v_soma_float2tensor_by_shape_ints(*v2soma.shape[:-1])
        v_soma = self.soma.v
        # v_soma.shape = [N, *self.soma_shape]
        input2soma = (v2soma - v_soma.unsqueeze(-1)) * self.forward_strength
        input2soma = input2soma.sum(dim = -1)
        # input2soma.shape = [N, *self.soma_shape]
        soma_spike = self.soma.single_step_forward(input2soma)
        return soma_spike.reshape([N, *self.soma_shape])

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # This implementation is equivalent but more efficient 
        # than the default version defined in BaseDendNeuron!
        # x_seq = [T, N, self.n_soma * self.dend.wiring.n_input]
        T, N = x_seq.shape[0], x_seq.shape[1]
        x_seq = self.reshape_input(x_seq, "m")
        v2soma_seq = self.dend.multi_step_forward(x_seq)
        # v2soma_seq.shape = [T, N, *self.soma_shape, self.dend.wiring.n_output]
        soma_spike_seq = []
        self.v_soma_float2tensor_by_shape_ints(*v2soma_seq[0].shape[:-1])
        for t in range(T):
            v_soma = self.soma.v
            # v_soma.shape = [N, *self.soma_shape]
            input2soma = v2soma_seq[t] - v_soma.unsqueeze(-1)
            input2soma = (input2soma * self.forward_strength).sum(dim = -1)
            # input2soma.shape = [N, *self.soma_shape]
            soma_spike = self.soma.single_step_forward(input2soma)
            soma_spike_seq.append(soma_spike.reshape(N, *self.soma_shape))
        return torch.stack(soma_spike_seq)



class VDiffForwardSBackwardDendNeuron(BaseDendNeuron):
    """Neurons with bidirectional signal transmission between dendrite and soma.

    Besides the delta_v-driven feed forward signal transmission from dendrite to
    soma defined in VDiffForwardDendNeuron, this module introduces a 
    somatic-spike-driven feedback mechanism from soma to dendrite. If a spike is
    generated in the soma, a back-propagated voltage will be added to all the 
    output compartments.

    Args:
        dend, soma, soma_shape, n_soma, step_mode: see BaseDendNeuron.
        forward_strength: see VDiffForwardDendNeuron.
        backward_strength: the feedback coupling strength from the soma to the
            output compartments, defined for a single neuron, which can be
            a Tensor with shape [1, dend.wiring.n_output] or a float (can be 
            broadcasted into a Tensor).
    """

    def __init__(
        self, dend: dendrite.BaseDend, soma: sj_neuron.BaseNode, 
        soma_shape: List[int],
        forward_strength: Union[float, torch.Tensor] = 1.,
        backward_strength: Union[float, torch.Tensor] = 1.,
        step_mode: str = "s"
    ):
        """The constructor of VDiffForwardSBackwardDendNeuron.

        Args:
            dend (BaseDend): dendrite model for a single neuron.
            soma (BaseNode): somatic model considering all the neurons.
            soma_shape (List[int]): the shape of the somatic voltage tensor in
                this module. The spatial structure of the neurons in this layer.
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
            step_mode (str, optional): "s" for single-step mode, and "m" for 
                multi-step mode. Defaults to "s".
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
        x = self.reshape_input(x, "s")
        v2soma = self.dend.single_step_forward(x)
        # v2soma.shape = [N, *self.soma_shape, self.dend.wiring.n_output]
        self.v_soma_float2tensor_by_shape_ints(*v2soma.shape[:-1])
        v_soma = self.soma.v
        # v_soma.shape = [N, *self.soma_shape]
        input2soma = (v2soma - v_soma.unsqueeze(-1)) * self.forward_strength
        input2soma = input2soma.sum(dim = -1)
        # input2soma.shape = [N, *self.soma_shape]
        soma_spike = self.soma.single_step_forward(input2soma)
        self.soma_spike_backprop(soma_spike)
        return soma_spike.reshape([N, *self.soma_shape])