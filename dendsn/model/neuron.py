"""Dendritic neuron modules.

This module defines a series of dendritic neuron layer models (rather than 
single neurons). Each layer consists of a set of dendritic neurons with exactly
the same dendritic morphology. A dendritic neuron layer is often placed after an
FC or Conv layer, acting as a activation function.
"""

import abc
from typing import Union, List, Optional, Callable

import torch
from torch.nn import parameter
import numpy as np
from spikingjelly.activation_based import base

from dendsn.model import dendrite
from dendsn.model import soma


class BaseDendNeuron(base.MemoryModule, abc.ABC):
    """Base class for all dendritic neuron modules.

    This class defines a layer of dendritic neurons with exactly the same 
    dendritic morphology and dynamical characteristics. A single dendritic 
    neuron is composed of a dendritic model, a somatic model, and optionally 
    connective strength between dendrite and soma. The key point is signal
    transmission between the output dendritic compartments and the soma.

    Attributes:
        dend (BaseDend): dendritic model for a single neuron in the layer.
        v_dend (torch.Tensor): dendritic compartmental potential at the curret 
            time step.
        soma (BaseSoma): somatic model considering all the neurons in the layer. 
        v_soma (torch.Tensor): somatic potential at the current time step.
        soma_shape (List[int]): the shape of the somatic voltage tensor in this 
            module. The spatial structure of the neurons in this layer. Read
            only.
        n_soma (int): the number of neurons / somas in the layer. Read only.
        step_mode (str): "s" for single-step mode, and "m" for multi-step mode.
        store_v_dend_seq (bool): whether to store the dendritic compartmental 
            potential at every time step when using multi-step mode. If True, 
            there is another attribute called v_dend_seq.
        store_v_soma_seq (bool): whether to store the post-spike somatic  
            potential at every time step when using multi-step mode. If True, 
            there is another attribute called v_soma_seq.
        store_v_dend_pre_spike (bool): whether to store the dendritic potential 
            before spike emission when forward() is called the last time, no 
            matter which step mode the module is in. If True, there's another 
            attribute called v_dend_pre_spike. The shape of v_dend_pre_spike
            depends on which step mode ("s" or "m") this module is in.
        store_v_soma_pre_spike (bool): whether to store the somatic potential
            right before spike emission when forward() is called the last time, 
            whether step_mode = "m" or "s". If True, there's another attribute 
            called v_soma_pre_spike. The shape of v_soma_pre_spike depends on 
            which step mode ("s" or "m") this module is in.
    """

    def __init__(
        self, dend: dendrite.BaseDend, soma: soma.BaseSoma,
        soma_shape: List[int], step_mode: str = "s", 
        store_v_dend_seq: bool = False, store_v_soma_seq: bool = False,
        store_v_dend_pre_spike: bool = False,
        store_v_soma_pre_spike: bool = False
    ):
        """The constructor of BaseDendNeuron.

        Args:
            dend (BaseDend): dendritic model for a single neuron.
            soma (BaseSoma): somatic model considering all the neurons.
            soma_shape (List[int]): the shape of the somatic voltage tensor in
                this module. The spatial structure of the neurons in this layer.
            step_mode (str, optional): "s" for single-step mode, and "m" for 
                multi-step mode.. Defaults to "s".
            store_v_dend_seq (bool, optional): whether to store the dendritic 
                compartmental potentials at every time step when using 
                multi-step mode. Defaults to False.
            store_v_soma_seq (bool, optional): whether to store the somatic
                potential at every time step when using multi-step mode. 
                Defaults to False.
            store_v_dend_pre_spike (bool): whether to store the dendritic 
                potential before spike emission when forward() is called the 
                last time. Defaults to False.
            store_v_soma_pre_spike (bool): whether to store the pre-spike 
                somatic potential when calling forward() the last time. 
                Defaults to False.
        """
        super().__init__()
        self.dend = dend
        self.soma = soma
        self._soma_shape = soma_shape
        self._n_soma = np.prod(soma_shape)
        self.step_mode = step_mode
        self.store_v_dend_seq = store_v_dend_seq
        self.store_v_soma_seq = store_v_soma_seq
        self.store_v_dend_pre_spike = store_v_dend_pre_spike
        self.store_v_soma_pre_spike = store_v_soma_pre_spike

    @property
    def soma_shape(self) -> List[int]:
        return self._soma_shape

    @property
    def n_soma(self) -> int:
        return self._n_soma

    @property
    def v_dend(self) -> torch.Tensor:
        return self.dend.compartment.v

    @property
    def v_soma(self) -> torch.Tensor:
        return self.soma.v

    @property
    def store_v_dend_seq(self) -> bool:
        return self._store_v_dend_seq

    @store_v_dend_seq.setter
    def store_v_dend_seq(self, val: bool):
        self._store_v_dend_seq = val
        self.dend.compartment.store_v_seq = val
        if val and (not hasattr(self, "v_dend_seq")):
            self.register_memory("v_dend_seq", None)

    @property
    def store_v_soma_seq(self) -> bool:
        return self._store_v_soma_seq

    @store_v_soma_seq.setter
    def store_v_soma_seq(self, val: bool):
        self._store_v_soma_seq = val
        self.soma.store_v_seq = val
        if val and (not hasattr(self, "v_soma_seq")):
            self.register_memory("v_soma_seq", None)

    @property
    def store_v_dend_pre_spike(self) -> bool:
        return self._store_v_dend_pre_spike

    @store_v_dend_pre_spike.setter
    def store_v_dend_pre_spike(self, val: bool):
        self._store_v_dend_pre_spike = val
        if val and (not hasattr(self, "v_dend_pre_spike")):
            self.register_memory("v_dend_pre_spike", None)

    @property
    def store_v_soma_pre_spike(self) -> bool:
        return self._store_v_soma_pre_spike

    @store_v_soma_pre_spike.setter
    def store_v_soma_pre_spike(self, val: bool):
        self._store_v_soma_pre_spike = val
        self.soma.store_v_pre_spike = val
        if val and (not hasattr(self, "v_soma_pre_spike")):
            self.register_memory("v_soma_pre_spike", None)

    def reset(self):
        self.dend.reset()
        self.soma.reset()

    def v_soma_float2tensor_by_shape_ints(self, device, *shape):
        """If soma.v is a float, turn it into a tensor with specified shape.

        Args:
            device: the device that soma.v will be on.
            shape: several integers describing the shape we want soma.v to be.
        """
        if isinstance(self.soma.v, float):
            v_init = self.soma.v
            self.soma.v = torch.full(
                size = shape, fill_value = v_init, device=device
            )

    def v_soma_float2tensor_by_shape_list(self, device, shape):
        """If soma.v is a float, turn it into a tensor with specified shape.

        Args:
            device: the device that soma.v will be on.
            shape (List[int]): a list of integers describing the shape we want
                soma.v to be.
        """
        self.v_soma_float2tensor_by_shape_ints(device, *shape)

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
        if self.store_v_dend_seq:
            v_dend_seq = []
        if self.store_v_soma_seq:
            v_soma_seq = []
        if self.store_v_dend_pre_spike:
            v_dend_pre_spike = []
        if self.store_v_soma_pre_spike:
            v_soma_pre_spike = []
        for t in range(T):
            soma_spike = self.single_step_forward(x_seq[t])
            soma_spike_seq.append(soma_spike)
            if self.store_v_dend_seq:
                v_dend_seq.append(self.v_dend)
            if self.store_v_soma_seq:
                v_soma_seq.append(self.v_soma)
            if self.store_v_dend_pre_spike: 
                # after single_step_forward() is called, self.v_dend_pre_spike 
                # now contains v_dend_pre_spike at time step t
                v_dend_pre_spike.append(self.v_dend_pre_spike)
            if self.store_v_soma_pre_spike:
                # after single_step_forward() is called, self.v_soma_pre_spike 
                # now contains v_soma_pre_spike at time step t
                v_soma_pre_spike.append(self.v_soma_pre_spike)
        y = torch.stack(soma_spike_seq)
        if self.store_v_dend_seq:
            self.v_dend_seq = torch.stack(v_dend_seq)
        if self.store_v_soma_seq:
            self.v_soma_seq = torch.stack(v_soma_seq)
        if self.store_v_dend_pre_spike:
            self.v_dend_pre_spike = torch.stack(v_dend_pre_spike)
        if self.store_v_soma_pre_spike:
            self.v_soma_pre_spike = torch.stack(v_soma_pre_spike)
        return y

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


class VForwardDendNeuron(BaseDendNeuron, abc.ABC):
    """Base class for all neurons transmitting v_dend forward to the soma.

    The input to the soma is a function of the dendritic voltage and is 
    moderated by the somatic voltage. See the docstring of get_input2soma().

    Attributes:
        dend, v_dend, soma, v_soma, soma_shape, n_soma, step_mode, 
        store_v_dend_seq, store_v_soma_seq, store_v_dend_pre_spike, 
        store_v_soma_pre_spike: 
            see BaseDendNeuron.
        forward_strength: the feed forward coupling strength phi from the output
            compartments to the soma, defined for a single neuron, which can be
            a Tensor with shape [dend.wiring.n_output] or a float (can be 
            broadcasted into a Tensor).
        forward_strength_learnable: whether to compute the grad of
            forward_strength or not.
    """

    def __init__(
        self, dend: dendrite.BaseDend, soma: soma.BaseSoma,
        soma_shape: List[int], 
        forward_strength: Union[float, torch.Tensor] = 1.,
        forward_strength_learnable: bool = False, step_mode: str = "s", 
        store_v_dend_seq: bool = False, store_v_soma_seq: bool = False,
        store_v_dend_pre_spike: bool = False,
        store_v_soma_pre_spike: bool = False
    ):
        """The constructor of VForwardDendNeuron.

        Args:
            dend (BaseDend): dendrite model for a single neuron.
            soma (BaseSoma): somatic model considering all the neurons.
            soma_shape (List[int]): the shape of the somatic voltage tensor in
                this module. The spatial structure of the neurons in this layer.
                The batch dimension should not be included in this list.
            forward_strength (Union[float, torch.Tensor], optional): 
                the coupling strength of the feed-forward connections from the
                output compartments to the soma (defined for a single neuron). 
                If it's a torch.Tensor, its shape should be 
                [dend.wiring.n_output]. If it's a float, the same strength
                will be always be applied to all the connections (even if new 
                weights are learned). Defaults to 1. .
            forward_strength_learnable (bool): whether to compute the grad of
                forward_strength or not. Defaults to None.
            step_mode (str, optional): "s" for single-step mode, and "m" for 
                multi-step mode. Defaults to "s".
            store_v_dend_seq (bool, optional): whether to store the dendritic 
                compartmental potentials at every time step when using 
                multi-step mode. Defaults to False.
            store_v_soma_seq (bool, optional): whether to store the somatic
                potential at every time step when using multi-step mode. 
                Defaults to False.
            store_v_dend_pre_spike (bool): whether to store the dendritic 
                potential before spike emission when forward() is called the 
                last time. Defaults to False.
            store_v_soma_pre_spike (bool): whether to store the pre-spike 
                somatic potential when calling forward() the last time. 
                Defaults to False.
        """
        super().__init__(
            dend, soma, soma_shape, step_mode, 
            store_v_dend_seq, store_v_soma_seq, 
            store_v_dend_pre_spike, store_v_soma_pre_spike
        )
        self._forward_strength_learnable = forward_strength_learnable
        self.forward_strength = parameter.Parameter(
            data=torch.tensor(forward_strength), 
            requires_grad=forward_strength_learnable
        )

    @property
    def forward_strength_learnable(self) -> bool:
        return self._forward_strength_learnable

    @forward_strength_learnable.setter
    def forward_strength_learnable(self, v: bool):
        self._forward_strength_learnable = v
        self.forward_strength.requires_grad = v

    @property
    def store_v_dend_pre_spike(self) -> bool:
        return self._store_v_dend_pre_spike

    @store_v_dend_pre_spike.setter
    def store_v_dend_pre_spike(self, val: bool):
        # For VForwardDendNeuron, `v_dend_pre_spike == v_dend` is always true.
        # Hence, use store_v_dend_seq to implement store_v_dend_pre_spike in
        # multi-step mode. 
        self._store_v_dend_pre_spike = val
        self.store_v_dend_seq = val or self.store_v_dend_seq
        if val and (not hasattr(self, "v_dend_pre_spike")):
            self.register_memory("v_dend_pre_spike", None)

    @abc.abstractmethod
    def get_input2soma(
        self, v_dend_output: torch.Tensor, v_soma: torch.Tensor
    ) -> torch.Tensor:
        """Map dendritic and somatic voltage to the somatic input signal.

        The input tensor fed to the soma is determined mainly by the voltage of 
        the output dendritic compartments, and moderated by the somatic voltage.
        To create a child class of VForwardDendNeuron, implement this method!

        Args:
            v_dend_output (torch.Tensor): the voltage of the output dendritic
                compartments with shape [N, *soma_shape, n_output] or [T, N,
                *soma_shape, n_output].
            v_soma (torch.Tensor): the somatic voltage with shape [N, 
                *soma_shape] or [T, N, *soma_shape].

        Returns:
            torch.Tensor: somatic input with shape [N, *soma_shape] or [T, N,
                *soma_shape]. If v_soma or v_dend_output has a prepending T
                dimension, the return value will also have a prepending T 
                dimension.
        """
        pass

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reshape_input(x, "s") # x.shape = [N, soma_shape, n_input]
        self.dend.step_mode = "s"
        v_dend_output = self.dend(x)

        # v_dend_output.shape = [N, *self.soma_shape, self.dend.wiring.n_output]
        self.v_soma_float2tensor_by_shape_ints(
            x.device, *v_dend_output.shape[:-1]
        )
        v_soma = self.soma.v

        # v_soma.shape = [N, *self.soma_shape]
        input2soma = self.get_input2soma(v_dend_output, v_soma)

        # input2soma.shape = [N, *self.soma_shape]
        self.soma.step_mode = "s"
        soma_spike = self.soma(input2soma)
        if self.store_v_dend_pre_spike:
            self.v_dend_pre_spike = self.v_dend
        if self.store_v_soma_pre_spike:
            self.v_soma_pre_spike = self.soma.v_pre_spike
        return soma_spike


class VDiffForwardDendNeuron(VForwardDendNeuron):
    """A layer of neurons that transmit (v_dend - v_soma) forward to the soma.

    The input to the soma is determined by the voltage difference between the 
    soma and the output compartments. The input to soma a in this layer is:
        J_{a} = \sum_i phi_{ia}(v_i - v_a)
    where i can be all output dendritic compartments in neuron a, and phi_{ia} 
    is the feed forward coupling strength.

    Attributes:
        dend, v_dend, soma, v_soma, soma_shape, n_soma, step_mode,
        store_v_dend_seq, store_v_soma_seq, store_v_dend_pre_spike, 
        store_v_soma_pre_spike: 
            see BaseDendNeuron.
        forward_strength, forward_strength_learnable: see VForwardDendNeuron.
    """

    def __init__(
        self, dend: dendrite.BaseDend, soma: soma.BaseSoma,
        soma_shape: List[int],
        forward_strength: Union[float, torch.Tensor] = 1.,
        forward_strength_learnable: bool = False, step_mode: str = "s",
        store_v_dend_seq: bool = False, store_v_soma_seq: bool = False, 
        store_v_dend_pre_spike: bool = False,
        store_v_soma_pre_spike: bool = False
    ):
        """The constructor of VDiffForwardDendNeuron.

        Args:
            dend (BaseDend): dendrite model for a single neuron.
            soma (BaseSoma): somatic model considering all the neurons.
            soma_shape (List[int]): the shape of the somatic voltage tensor in
                this module. The spatial structure of the neurons in this layer.
                The batch dimension should not be included in this list.
            forward_strength (Union[float, torch.Tensor], optional): 
                the coupling strength of the feed-forward connections from the
                output compartments to the soma (defined for a single neuron). 
                If it's a torch.Tensor, its shape should be 
                [dend.wiring.n_output]. If it's a float, the same strength
                will be applied to all the connections (even if new weights are
                learned). Defaults to 1. .
            forward_strength_learnable (bool): whether to compute the grad of
                forward_strength or not. Defaults to None.
            step_mode (str, optional): "s" for single-step mode, and "m" for 
                multi-step mode. Defaults to "s".
            store_v_dend_seq (bool, optional): whether to store the dendritic 
                compartmental potentials at every time step when using 
                multi-step mode. Defaults to False.
            store_v_soma_seq (bool, optional): whether to store the somatic
                potential at every time step when using multi-step mode. 
                Defaults to False.
            store_v_dend_pre_spike (bool): whether to store the dendritic 
                potential before spike emission when forward() is called the 
                last time. Defaults to False.
            store_v_soma_pre_spike (bool): whether to store the pre-spike 
                somatic potential when calling forward() the last time. 
                Defaults to False.
        """
        super().__init__(
            dend, soma, soma_shape, 
            forward_strength, forward_strength_learnable, step_mode,
            store_v_dend_seq, store_v_soma_seq, 
            store_v_dend_pre_spike, store_v_soma_pre_spike
        )

    def get_input2soma(
        self, v_dend_output: torch.Tensor, v_soma: torch.Tensor
    ) -> torch.Tensor:
        input2soma = (v_dend_output-v_soma.unsqueeze(-1))
        #w = self.forward_strength * torch.ones([self.dend.wiring.n_output])
        #return input2soma @ w
        input2soma = input2soma * self.forward_strength
        return input2soma.sum(dim=-1)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """This implementation is equivalent but more efficient than default.
        """
        T= x_seq.shape[0]
        x_seq = self.reshape_input(x_seq, "m") # shape=[T, N, n_soma, n_input]
        self.dend.step_mode = "m"
        v_dend_output_seq = self.dend(x_seq)

        # v2soma_seq.shape = [T, N, *self.soma_shape, self.dend.wiring.n_output]
        self.v_soma_float2tensor_by_shape_ints(
            x_seq.device, *v_dend_output_seq[0].shape[:-1]
        )

        soma_spike_seq = []
        self.soma.step_mode = "s"
        if self.store_v_soma_seq:
            v_soma_seq = []
        if self.store_v_soma_pre_spike:
            v_soma_pre_spike = []
        for t in range(T):
            v_soma = self.soma.v
            v_dend_output = v_dend_output_seq[t]
            input2soma = self.get_input2soma(v_dend_output, v_soma)
            # input2soma.shape = [N, *self.soma_shape]
            soma_spike = self.soma(input2soma)
            soma_spike_seq.append(soma_spike)
            if self.store_v_soma_seq:
                v_soma_seq.append(self.v_soma)
            if self.store_v_soma_pre_spike:
                v_soma_pre_spike.append(self.soma.v_pre_spike)

        if self.store_v_dend_seq:
            self.v_dend_seq = self.dend.compartment.v_seq
        if self.store_v_soma_seq:
            self.v_soma_seq = torch.stack(v_soma_seq)
        if self.store_v_dend_seq:
            # For VForwardDendNeuron, v_dend_pre_spike == v_dend_seq in 
            # multi-step mode
            self.v_dend_pre_spike = self.dend.compartment.v_seq
        if self.store_v_soma_pre_spike:
            self.v_soma_pre_spike = torch.stack(v_soma_pre_spike)

        return torch.stack(soma_spike_seq)


class VActivationForwardDendNeuron(VForwardDendNeuron):
    """A layer of neurons that transmit activated v_dend forward to the soma.

    The input to the soma is determined by applying the dendritic activation 
    function on the voltage of the output dendritic compartments. The input to 
    soma a in this layer is:
        J_{a} = \sum_i phi_{ia}f(v_i)
    where i can be all output dendritic compartments in neuron a, and phi_{ia} 
    is the feed forward coupling strength.

    Attributes:
        dend, v_dend, soma, v_soma, soma_shape, n_soma, step_mode,
        store_v_dend_seq, store_v_soma_seq, store_v_dend_pre_spike, 
        store_v_soma_pre_spike: 
            see BaseDendNeuron.
        f_da: the dendritic activation function applied on all the output 
            dendritic compartmental voltages.
        forward_strength, forward_strength_learnable: see VForwardDendNeuron.
    """

    def __init__(
        self, dend: dendrite.BaseDend, soma: soma.BaseSoma,
        soma_shape: List[int], f_da: Callable,
        forward_strength: Union[float, torch.Tensor] = 1.,
        forward_strength_learnable: bool = False, step_mode: str = "s",
        store_v_dend_seq: bool = False, store_v_soma_seq: bool = False,
        store_v_dend_pre_spike: bool = False,
        store_v_soma_pre_spike: bool = False
    ):
        """The constructor of VActivationForwardDendNeuron.

        Args:
            dend (BaseDend): dendrite model for a single neuron.
            soma (BaseSoma): somatic model considering all the neurons.
            soma_shape (List[int]): the shape of the somatic voltage tensor in
                this module. The spatial structure of the neurons in this layer.
                The batch dimension should not be included in this list.
            f_da (Callable): the dendritic activation function applied on all 
                the output dendritic compartmental voltages.
            forward_strength (Union[float, torch.Tensor], optional): 
                the coupling strength of the feed-forward connections from the
                output compartments to the soma (defined for a single neuron). 
                If it's a torch.Tensor, its shape should be 
                [dend.wiring.n_output]. If it's a float, the same strength
                will be applied to all the connections (even if new weights are
                learned). Defaults to 1. .
            forward_strength_learnable (bool): whether to compute the grad of
                forward_strength or not. Defaults to None.
            step_mode (str, optional): "s" for single-step mode, and "m" for 
                multi-step mode. Defaults to "s".
            store_v_dend_seq (bool, optional): whether to store the dendritic 
                compartmental potentials at every time step when using 
                multi-step mode. Defaults to False.
            store_v_soma_seq (bool, optional): whether to store the somatic
                potential at every time step when using multi-step mode. 
                Defaults to False.
            store_v_dend_pre_spike (bool): whether to store the dendritic 
                potential before spike emission when forward() is called the 
                last time. Defaults to False.
            store_v_soma_pre_spike (bool): whether to store the pre-spike 
                somatic potential when calling forward() the last time. 
                Defaults to False.
        """
        super().__init__(
            dend, soma, soma_shape, 
            forward_strength, forward_strength_learnable, step_mode,
            store_v_dend_seq, store_v_soma_seq, 
            store_v_dend_pre_spike, store_v_soma_pre_spike
        )
        self.f_da = f_da

    def get_input2soma(
        self, v_dend_output: torch.Tensor, v_soma: torch.Tensor
    ) -> torch.Tensor:
        input2soma = self.f_da(v_dend_output)
        input2soma = input2soma * self.forward_strength
        return input2soma.sum(dim=-1)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """This implementation is equivalent but more efficient than default.
        """
        x_seq = self.reshape_input(x_seq, "m") # shape=[T, N, n_soma, n_input]
        self.dend.step_mode = "m"
        v_dend_output_seq = self.dend(x_seq)

        # v2soma_seq.shape = [T, N, *self.soma_shape, self.dend.wiring.n_output]
        self.v_soma_float2tensor_by_shape_ints(
            x_seq.device, *v_dend_output_seq[0].shape[:-1]
        )

        input2soma_seq = self.get_input2soma(v_dend_output_seq, self.soma.v)
        #input2soma_seq.shape = [T, N, *self.soma_shape]
        self.soma.step_mode = "m"
        soma_spike_seq = self.soma(input2soma_seq)
        if self.store_v_dend_seq:
            self.v_dend_seq = self.dend.compartment.v_seq
        if self.store_v_soma_seq:
            self.v_soma_seq = self.soma.v_seq
        if self.store_v_dend_pre_spike:
            # For VForwardDendNeuron, v_dend_pre_spike == v_dend_seq in 
            # multi-step mode
            self.v_dend_pre_spike = self.dend.compartment.v_seq
        if self.store_v_soma_pre_spike:
            self.v_soma_pre_spike = self.soma.v_pre_spike
        return soma_spike_seq


class VForwardSBackwardDendNeuron(BaseDendNeuron, abc.ABC):
    """Base neuron class transmitting v_dend forward and s_soma backward.

    The input to the soma is a function of the dendritic voltage and is 
    moderated by the somatic voltage, while the somatic spikes will also 
    influence the dendritic voltages in a backward manner. See the docstrings
    of get_input2soma() and bp_soma_spike().

    Attributes:
        dend, v_dend, soma, v_soma, soma_shape, n_soma, step_mode,
        store_v_dend_seq, store_v_soma_seq, store_v_dend_pre_spike, 
        store_v_soma_pre_spike: 
            see BaseDendNeuron.
        forward_strength: the feed forward coupling strength phi from the output
            compartments to the soma, defined for a single neuron, which can be
            a Tensor with shape [dend.wiring.n_output] or a float (can be 
            broadcasted into a Tensor).
        forward_strength_learnable: whether to compute the grad of
            forward_strength or not.
        backward_strength: the feedback coupling strength from the soma to the
            output compartments, defined for a single neuron, which can be
            a Tensor with shape [dend.wiring.n_output] or a float (can be 
            broadcasted into a Tensor).
        backward_strength_learnable: whether to compute the grad of 
            backward_strength or not.
    """

    def __init__(
        self, dend: dendrite.BaseDend, soma: soma.BaseSoma, 
        soma_shape: List[int], 
        forward_strength: Union[float, torch.Tensor] = 1.,
        forward_strength_learnable: bool = False,
        backward_strength: Union[float, torch.Tensor] = 1.,
        backward_strength_learnable: bool = False,
        step_mode: str = "s",
        store_v_dend_seq: bool = False, store_v_soma_seq: bool = False,
        store_v_dend_pre_spike: bool = False,
        store_v_soma_pre_spike: bool = False
    ):
        """The constructor of VForwardSBackwardDendNeuron.

        Args:
            dend (BaseDend): dendrite model for a single neuron.
            soma (BaseSoma): somatic model considering all the neurons.
            soma_shape (List[int]): the shape of the somatic voltage tensor in
                this module. The spatial structure of the neurons in this layer.
            forward_strength (Union[float, torch.Tensor], optional): 
                the coupling strength of the feed-forward connections from the
                output compartments to the soma (defined for a single neuron). 
                If it's a torch.Tensor, its shape should be 
                [dend.wiring.n_output]. If it's a float, the same strength
                will be applied to all the connections (even if new weights are
                learned). Defaults to 1..
            forward_strength_learnable (bool): whether to compute the grad of
                forward_strength or not. Defaults to None.
            backward_strength (Union[float, torch.Tensor], optional):
                the coupling strength of the feedback connections from the
                soma to the output compartments (defined for a single neuron). 
                If it's a torch.Tensor, its shape should be 
                [dend.wiring.n_output]. If it's a float, the same strength
                will be applied to all the connections (even if new weights are
                learned). Defaults to 1..
            backward_strength_learnable (bool): whether to compute the grad of
                backward_strength or not. Defaults to None.
            step_mode (str, optional): "s" for single-step mode, and "m" for 
                multi-step mode. Defaults to "s".
            store_v_dend_seq (bool, optional): whether to store the dendritic 
                compartmental potentials at every time step when using 
                multi-step mode. Defaults to False.
            store_v_soma_seq (bool, optional): whether to store the somatic
                potential at every time step when using multi-step mode. 
                Defaults to False.
            store_v_dend_pre_spike (bool): whether to store the dendritic 
                potential before spike emission when forward() is called the 
                last time. Defaults to False.
            store_v_soma_pre_spike (bool): whether to store the pre-spike 
                somatic potential when calling forward() the last time. 
                Defaults to False.
        """
        super().__init__(
            dend, soma, soma_shape, step_mode, 
            store_v_dend_seq, store_v_soma_seq, 
            store_v_dend_pre_spike, store_v_soma_pre_spike
        )
        self._forward_strength_learnable = forward_strength_learnable
        self.forward_strength = parameter.Parameter(
            data=torch.tensor(forward_strength), 
            requires_grad=forward_strength_learnable
        )
        self._backward_strength_learnable = backward_strength_learnable
        self.backward_strength = parameter.Parameter(
            data=torch.tensor(backward_strength), 
            requires_grad=backward_strength_learnable
        )

    @property
    def forward_strength_learnable(self) -> bool:
        return self._forward_strength_learnable

    @forward_strength_learnable.setter
    def forward_strength_learnable(self, v: bool):
        self._forward_strength_learnable = v
        self.forward_strength.requires_grad = v

    @property
    def backward_strength_learnable(self) -> bool:
        return self._backward_strength_learnable

    @backward_strength_learnable.setter
    def backward_strength_learnable(self, v: bool):
        self._backward_strength_learnable = v
        self.backward_strength.requires_grad = v

    @abc.abstractmethod
    def get_input2soma(
        self, v_dend_output: torch.Tensor, v_soma: torch.Tensor
    ) -> torch.Tensor:
        """Map dendritic and somatic voltage to the somatic input signal.

        The input tensor fed to the soma is determined mainly by the voltage of 
        the output dendritic compartments, and moderated by the somatic voltage.
        To create a child class of VForwardSBackwardDendNeuron, implement this 
        method!

        Args:
            v_dend_output (torch.Tensor): the voltage of the output dendritic
                compartments with shape [N, *soma_shape, n_output] or [T, N,
                *soma_shape, n_output].
            v_soma (torch.Tensor): the somatic voltage with shape [N, 
                *soma_shape] or [T, N, *soma_shape].

        Returns:
            torch.Tensor: somatic input with shape [N, *soma_shape] or [T, N,
                *soma_shape]. If v_soma or v_dend_output has a prepending T
                dimension, the return value will also have a prepending T 
                dimension.
        """
        pass

    @abc.abstractmethod
    def bp_soma_spike(self, soma_spike: torch.Tensor) -> None:
        """Change the dendritic voltages given the somatic spike.

        This method has no return value, and it changes the state of self.dend 
        directly.

        Args:
            soma_spike (torch.Tensor): the somatic spike with shape 
                [N, *soma_shape]
        """
        pass

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reshape_input(x, "s") # shape=[T, N, n_soma, n_input]
        self.dend.step_mode = "s"
        v_dend_output = self.dend(x) 
        if self.store_v_dend_pre_spike:
            self.v_dend_pre_spike = self.v_dend

        # v_dend_output.shape = [N, *self.soma_shape, self.dend.wiring.n_output]
        self.v_soma_float2tensor_by_shape_ints(
            x.device, *v_dend_output.shape[:-1]
        )
        v_soma = self.soma.v

        # v_soma.shape = [N, *self.soma_shape]
        input2soma = self.get_input2soma(v_dend_output, v_soma)

        # input2soma.shape = [N, *self.soma_shape]
        self.soma.step_mode = "s"
        soma_spike = self.soma(input2soma)
        self.bp_soma_spike(soma_spike)

        if self.store_v_soma_pre_spike:
            self.v_soma_pre_spike = self.soma.v_pre_spike
        return soma_spike


class VDiffForwardSBackwardDendNeuron(VForwardSBackwardDendNeuron):
    """Neurons transmitting (v_dend-v_soma) forward and s_soma backward.

    Besides the delta_v-driven feed forward signal transmission from dendrite to
    soma defined in VDiffForwardDendNeuron, this module introduces a 
    somatic-spike-driven feedback mechanism from soma to dendrite. If a spike is
    generated in the soma, a back-propagated voltage will be added to all the 
    output compartments.

    Args:
        dend, v_dend, soma, v_soma, soma_shape, n_soma, step_mode,
        store_v_dend_seq, store_v_soma_seq, store_v_dend_pre_spike, 
        store_v_soma_pre_spike: 
            see BaseDendNeuron.
        forward_strength, forward_strength_learnable, backward_strength,
            backward_strength_learnable: see VForwardSBackwardNeuron.
    """

    def __init__(
        self, dend: dendrite.BaseDend, soma: soma.BaseSoma, 
        soma_shape: List[int],
        forward_strength: Union[float, torch.Tensor] = 1.,
        forward_strength_learnable: bool = False,
        backward_strength: Union[float, torch.Tensor] = 1.,
        backward_strength_learnable: bool = False,
        step_mode: str = "s",
        store_v_dend_seq: bool = False, store_v_soma_seq: bool = False, 
        store_v_dend_pre_spike: bool = False,
        store_v_soma_pre_spike: bool = False
    ):
        """The constructor of VDiffForwardSBackwardDendNeuron.

        Args:
            dend (BaseDend): dendrite model for a single neuron.
            soma (BaseSoma): somatic model considering all the neurons.
            soma_shape (List[int]): the shape of the somatic voltage tensor in
                this module. The spatial structure of the neurons in this layer.
            forward_strength (Union[float, torch.Tensor], optional): 
                the coupling strength of the feed-forward connections from the
                output compartments to the soma (defined for a single neuron). 
                If it's a torch.Tensor, its shape should be 
                [dend.wiring.n_output]. If it's a float, the same strength
                will be applied to all the connections (even if new weights are
                learned). Defaults to 1..
            forward_strength_learnable (bool): whether to compute the grad of
                forward_strength or not. Defaults to None.
            backward_strength (Union[float, torch.Tensor], optional):
                the coupling strength of the feedback connections from the
                soma to the output compartments (defined for a single neuron). 
                If it's a torch.Tensor, its shape should be 
                [dend.wiring.n_output]. If it's a float, the same strength
                will be applied to all the connections (even if new weights are
                learned). Defaults to 1..
            backward_strength_learnable (bool): whether to compute the grad of
                backward_strength or not. Defaults to None.
            step_mode (str, optional): "s" for single-step mode, and "m" for 
                multi-step mode. Defaults to "s".
            store_v_dend_seq (bool, optional): whether to store the dendritic 
                compartmental potentials at every time step when using 
                multi-step mode. Defaults to False.
            store_v_soma_seq (bool, optional): whether to store the somatic
                potential at every time step when using multi-step mode. 
                Defaults to False.
            store_v_dend_pre_spike (bool): whether to store the dendritic 
                potential before spike emission when forward() is called the 
                last time. Defaults to False.
            store_v_soma_pre_spike (bool): whether to store the pre-spike 
                somatic potential when calling forward() the last time. 
                Defaults to False.
        """
        super().__init__(
            dend, soma, soma_shape, 
            forward_strength, forward_strength_learnable,
            backward_strength, backward_strength_learnable,
            step_mode, store_v_dend_seq, store_v_soma_seq, 
            store_v_dend_pre_spike, store_v_soma_pre_spike
        )

    def get_input2soma(
        self, v_dend_output: torch.Tensor, v_soma: torch.Tensor
    ) -> torch.Tensor:
        input2soma = (v_dend_output-v_soma.unsqueeze(-1))
        input2soma = input2soma * self.forward_strength
        return input2soma.sum(dim=-1)

    def bp_soma_spike(self, soma_spike: torch.Tensor):
        # soma_spike.shape = [N, *self.soma_shape]
        w = (self.backward_strength *
            torch.ones(size = [1, self.dend.wiring.n_output]))
        bp_v = soma_spike.unsqueeze(-1) @ w
        # bp_v.shape = [N, *self.soma_shape, self.dend.wiring.n_output]
        self.dend.compartment.v[..., self.dend.wiring.output_index] += bp_v


class VActivationForwardSBackwardDendNeuron(VForwardSBackwardDendNeuron):
    """Neurons transmitting activated v_dend forward and s_soma backward.

    Besides the activated voltage of the output dendritic compartments from 
    dendrite to soma defined in VDiffForwardDendNeuron, this module introduces a 
    somatic-spike-driven feedback mechanism from soma to dendrite. If a spike is
    generated in the soma, a back-propagated voltage will be added to all the 
    output compartments.

    Args:
        dend, v_dend, soma, v_soma, soma_shape, n_soma, step_mode,
        store_v_dend_seq, store_v_soma_seq, store_v_dend_pre_spike, 
        store_v_soma_pre_spike: 
            see BaseDendNeuron.
        f_da: the dendritic activation function applied on all the output 
            dendritic compartmental voltages. 
        forward_strength, forward_strength_learnable, backward_strength,
            backward_strength_learnable: see VForwardSBackwardNeuron.
    """

    def __init__(
        self, dend: dendrite.BaseDend, soma: soma.BaseSoma, 
        soma_shape: List[int], f_da: Callable,
        forward_strength: Union[float, torch.Tensor] = 1.,
        forward_strength_learnable: bool = False,
        backward_strength: Union[float, torch.Tensor] = 1.,
        backward_strength_learnable: bool = False,
        step_mode: str = "s",
        store_v_dend_seq: bool = False, store_v_soma_seq: bool = False,
        store_v_dend_pre_spike: bool = False,
        store_v_soma_pre_spike: bool = False
    ):
        """The constructor of VActivationForwardSBackwardDendNeuron.

        Args:
            dend (BaseDend): dendrite model for a single neuron.
            soma (BaseSoma): somatic model considering all the neurons.
            soma_shape (List[int]): the shape of the somatic voltage tensor in
                this module. The spatial structure of the neurons in this layer.
            f_da (Callable): the dendritic activation function applied on all 
                the output dendritic compartmental voltages. 
            forward_strength (Union[float, torch.Tensor], optional): 
                the coupling strength of the feed-forward connections from the
                output compartments to the soma (defined for a single neuron). 
                If it's a torch.Tensor, its shape should be 
                [dend.wiring.n_output]. If it's a float, the same strength
                will be applied to all the connections (even if new weights are
                learned). Defaults to 1..
            forward_strength_learnable (bool): whether to compute the grad of
                forward_strength or not. Defaults to None.
            backward_strength (Union[float, torch.Tensor], optional):
                the coupling strength of the feedback connections from the
                soma to the output compartments (defined for a single neuron). 
                If it's a torch.Tensor, its shape should be 
                [dend.wiring.n_output]. If it's a float, the same strength
                will be applied to all the connections (even if new weights are
                learned). Defaults to 1..
            backward_strength_learnable (bool): whether to compute the grad of
                backward_strength or not. Defaults to None.
            step_mode (str, optional): "s" for single-step mode, and "m" for 
                multi-step mode. Defaults to "s".
            store_v_dend_seq (bool, optional): whether to store the dendritic 
                compartmental potentials at every time step when using 
                multi-step mode. Defaults to False.
            store_v_soma_seq (bool, optional): whether to store the somatic
                potential at every time step when using multi-step mode. 
                Defaults to False.
            store_v_dend_pre_spike (bool): whether to store the dendritic 
                potential before spike emission when forward() is called the 
                last time. Defaults to False.
            store_v_soma_pre_spike (bool): whether to store the pre-spike 
                somatic potential when calling forward() the last time. 
                Defaults to False.
        """
        super().__init__(
            dend, soma, soma_shape, 
            forward_strength, forward_strength_learnable,
            backward_strength, backward_strength_learnable,
            step_mode, store_v_dend_seq, store_v_soma_seq, 
            store_v_dend_pre_spike, store_v_soma_pre_spike
        )
        self.f_da = f_da

    def get_input2soma(
        self, v_dend_output: torch.Tensor, v_soma: torch.Tensor
    ) -> torch.Tensor:
        input2soma = self.f_da(v_dend_output)
        input2soma = input2soma * self.forward_strength
        return input2soma.sum(dim=-1)

    def soma_spike_backprop(self, soma_spike: torch.Tensor):
        # soma_spike.shape = [N, *self.soma_shape]
        w = (self.backward_strength *
            torch.ones(size = [1, self.dend.wiring.n_output]))
        bp_v = soma_spike.unsqueeze(-1) @ w
        # bp_v.shape = [N, *self.soma_shape, self.dend.wiring.n_output]
        self.dend.compartment.v[..., self.dend.wiring.output_index] += bp_v
