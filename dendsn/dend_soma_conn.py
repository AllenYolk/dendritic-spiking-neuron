import abc
from typing import Optional

import torch


class BaseDendSomaConn(abc.ABC):

    def __init__(
        self, n_output_compartment: int, n_soma: int, 
        enable_backward: bool, *args, **kwargs
    ):
        self._n_output_compartment = n_output_compartment
        self._n_soma = n_soma
        self._enable_backward = enable_backward
        self.forward_adjacency_matrix = torch.zeros(
            size = [n_output_compartment, n_soma], dtype = torch.int32
        )
        self.backward_adjacency_matrix = torch.zeros(
            size = [n_soma, n_output_compartment], dtype = torch.int32
        )
        self.build(*args, **kwargs)

    # n_output_compartment: read-only
    @property
    def n_output_compartment(self) -> int:
        return self._n_output_compartment

    # n_soma: read-only
    @property
    def n_soma(self) -> int:
        return self._n_soma

    # enable_backward: read-only
    @property
    def enable_backward(self) -> bool:
        return self._enable_backward

    def add_dend2soma_connection(self, src: int, dest: int):
        if (src < 0 or src >= self.n_output_compartment
            or dest < 0 or dest >= self.n_soma):
            raise ValueError(
                f"invalid dendrite-to-soma connection: "
                f"src = {src}, dest = {dest}"
            )
        self.forward_adjacency_matrix[src, dest] = 1

    def add_soma2dend_connection(self, src: int, dest: int):
        if (dest < 0 or dest >= self.n_output_compartment
            or src < 0 or src >= self.n_soma):
            raise ValueError(
                f"invalid soma-to-dendrite connection: "
                f"src = {src}, dest = {dest}"
            )
        self.backward_adjacency_matrix[src, dest] = 1 

    @abc.abstractmethod
    def build(self, *args, **kwargs):
        pass


class Kto1DendSomaConn(BaseDendSomaConn):

    def __init__(
        self, k: int, n_soma: int, n_output_compartment: Optional[int] = None,
        enable_backward: bool = False
    ):
        if n_output_compartment is None:
            n_output_compartment = k * n_soma
        elif n_output_compartment != k * n_soma:
            raise ValueError(
                f"n_output_compartment should be equal to k * n_soma, "
                f"but n_output_compartment = {n_output_compartment}, "
                f"k = {k}, n_soma = {n_soma}"
            )

        super().__init__(
            n_output_compartment = n_output_compartment, n_soma = n_soma,
            enable_backward = enable_backward, k = k
        )
        self.k = k

    def build(self, k: int):
        for dest in range(self.n_soma):
            for i in range(k):
                src = dest * k + i
                self.add_dend2soma_connection(src, dest)
        if self.enable_backward:
            for src in range(self.n_soma):
                for i in range(k):
                    dest = src * k + i
                    self.add_soma2dend_connection(src, dest)