"""
This module defines the connection diagram of the dendritic compartments in one
dendritic neuron so that we can build dendrites with complex morphologies.
The three key components of a "compartment wiring" are:
1. the adjacency matrix within a set of compartments
2. a list specifying which compartments receive synaptic inputs (a.k.a. input
    compartments)
3. a list specifying which compartments are connected to the soma (a.k.a. output
    compartments)

Notice that neither the synaptic connection mask from the presynaptic neurons
to the input compartments nor the connection from the output compartments to
the soma is defined here.
"""

import abc
from typing import List, Optional

import torch


class BaseWiring(abc.ABC):

    def __init__(
        self, n_compartment: int, n_input: int, output_index: List[int],
        bidirection: bool, *args, **kwargs
    ):
        self._n_compartment = n_compartment
        self.adjacency_matrix = torch.zeros(
            size = [n_compartment, n_compartment], dtype = torch.int32
        )
        self._n_input = n_input
        self._output_index = output_index
        self._bidirection = bidirection
        if not self.validation_check():
            raise ValueError(
                f"Invalid wiring:\n"
                f"  n_input = {n_input}\n"
                f"  output_index = {output_index}\n"
            )
        self.build(*args, **kwargs)

    # bidirection: read-only
    @property
    def bidirection(self) -> int:
        return self._bidirection

    # n_compartment: read-only
    @property
    def n_compartment(self) -> int:
        return self._n_compartment

    # input_index: read-only
    @property
    def input_index(self) -> List[int]:
        return list(range(self._n_input))

    # output_index: readable & writable
    @property
    def output_index(self) -> List[int]:
        return self._output_index

    @output_index.setter
    def output_index(self, idx):
        self._output_index = idx
        if not self.validation_check():
            raise ValueError(
                f"invalid output_index of wiring: "
                f"output_index = {idx}"
            )

    # n_input: readable & writable
    @property
    def n_input(self) -> int:
        return self._n_input

    @n_input.setter
    def n_input(self, n):
        self._n_input = n
        self.validation_check()
        if not self.validation_check():
            raise ValueError(
                f"invalid n_input of wiring: "
                f"n_input = {n}"
            )

    # n_output: read-only
    @property
    def n_output(self) -> int:
        return len(self._output_index)

    def validation_check(self) -> bool:
        # check the shape of the adjacency matrix
        ad_shape = self.adjacency_matrix.shape
        if not ad_shape == torch.Size([self.n_compartment, self.n_compartment]):
            return False

        # check the validity of n_input and output_index
        if (self.n_input < 0) or (self.n_input > self.n_compartment):
            return False
        if ((min(self.output_index) < 0)
            or (max(self.output_index) >= self.n_compartment)):
            return False

        # pass all the tests
        return True

    def _add_compartment_single_connection(self, src: int, dest: int):
        if (src < 0 or src >= self.n_compartment 
            or dest < 0 or dest >= self.n_compartment):
            raise ValueError(
                f"invalid dendritic compartment connection: "
                f"src = {src}, dest = {dest}"
            )
        self.adjacency_matrix[src, dest] = 1

    def _add_compartment_double_connection(self, n1: int, n2: int):
        self._add_compartment_single_connection(n1, n2)
        self._add_compartment_single_connection(n2, n1)

    def add_compartment_connection(self, n1: int, n2: int):
        if self.bidirection:
            self._add_compartment_double_connection(n1, n2)
        else:
            self._add_compartment_single_connection(n1, n2)

    @abc.abstractmethod
    def build(self, *args, **kwargs):
        pass


class SegregatedDendWiring(BaseWiring):

    def __init__(self, n_compartment: int):
        """
        The wiring diagram of segregated dendritic compartments of a single 
        neuron where n_compartment = n_input = n_output and the adjacency matrix
        is empty.

        Args:
            n_compartment (int):
        """
        super().__init__(
            n_compartment = n_compartment, n_input = n_compartment,
            output_index = list(range(n_compartment)), bidirection = False
        )

    def build(self):
        pass


class Kto1DendWirng(BaseWiring):

    def __init__(
        self, k: int, n_output: int,
        n_input: Optional[int] = None, bidirection: bool = False
    ):
        """
        The wiring diagram of dendritic compartments of a single neuron
        connected in a k-to-1 manner. There are `n_output` output compartments,
        and each of them receive information from `k` input compartments. Hence,
        there are (k+1) * n_output compartments in total.

        Args:
            k (int)
            n_output (int)
            n_input (Optional[int], optional): Defaults to None. If not None,
                should be equal to k * n_output
            bidirection (bool, optional): Defaults to False.

        Raises:
            ValueError: n_input is not None and does not equal to k * n_output .
        """
        if n_input is None:
            n_input = k * n_output
        elif n_output * k != n_input:
            raise ValueError(
                f"n_input should be equal to k * n_output, "
                f"but n_input = {n_input}, k = {k}, n_output = {n_output}"
            )

        n_compartment = n_input + n_output
        super().__init__(
            n_compartment = n_compartment, n_input = n_input,
            output_index = list(range(n_input, n_compartment)),
            bidirection = bidirection, k = k
        )
        self.k = k

    def build(self, k: int):
        for i, dest in enumerate(self.output_index):
            for j in range(k):
                src = i * k + j
                self.add_compartment_connection(src, dest)
