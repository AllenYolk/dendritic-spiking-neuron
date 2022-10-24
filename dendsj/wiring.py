import abc
from typing import List

import torch


class BaseWiring(abc.ABC):

    def __init__(
        self, n_compartment: int, n_input: int, output_index: List[int],
    ):
        self.n_compartment = n_compartment
        self.adjacency_matrix = torch.zeros([n_compartment, n_compartment])
        self._n_input = n_input
        self._output_index = output_index
        if not self.validation_check():
            raise ValueError(
                f"Invalid wiring:\n"
                f"  n_input = {n_input}\n"
                f"  output_index = {output_index}\n"
            )

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

    @abc.abstractmethod
    def build(self):
        pass
