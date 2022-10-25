"""
This module defines the connection diagram of the dendritic compartments so that
we can build dendrites with complex morphologies.
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
from typing import List

import torch


class BaseWiring(abc.ABC):

    def __init__(
        self, n_compartment: int, n_input: int, output_index: List[int],
    ):
        self._n_compartment = n_compartment
        self.adjacency_matrix = torch.zeros(
            size = [n_compartment, n_compartment], dtype = torch.int32
        )
        self._n_input = n_input
        self._output_index = output_index
        if not self.validation_check():
            raise ValueError(
                f"Invalid wiring:\n"
                f"  n_input = {n_input}\n"
                f"  output_index = {output_index}\n"
            )

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

    def add_compartment_connection(self, src: int, dest: int):
        if (src < 0 or src >= self.n_compartment 
            or dest < 0 or dest >= self.n_compartment):
            raise ValueError(
                f"invalid dendritic compartment connection: "
                f"src = {src}, dest = {dest}"
            )
        self.adjacency_matrix[src, dest] = 1

    def add_compartment_double_connection(self, n1: int, n2: int):
        self.add_compartment_connection(n1, n2)
        self.add_compartment_connection(n2, n1)

    @abc.abstractmethod
    def build(self):
        pass


class SingleDendLayerWiring(BaseWiring):

    def __init__(self, n_compartment: int):
        super().__init__(
            n_compartment = n_compartment, n_input = n_compartment,
            output_index = list(range(n_compartment))
        )

    def build(self):
        pass
