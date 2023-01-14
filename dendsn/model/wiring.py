"""The wiring diagram of dendritic compartments in a single neuron.

This module defines the complex morphology of the dendritic tree in a single
neuron. The three key components of a "compartment wiring" are:
1. the adjacency matrix within a set of compartments
2. a list specifying which compartments receive synaptic inputs (a.k.a. input
    compartments)
3. a list specifying which compartments are connected to the soma (a.k.a. output
    compartments)
Notice that the synaptic connection mask from the presynaptic neurons
to the input compartments is not defined here!
"""

import abc
from typing import List, Optional

import torch


class BaseWiring(abc.ABC):
    """Base class for all dendritic wiring.

    The design pattern here is: Builder. build() is the "Director", while 
    add_compartment_connection() is the "BuildPart() function". The wiring 
    is built up step by step, and can be accessed through adjacency_matrix. To
    define a subclass, just implement build() method.

    Attributes:
        adjacency_matrix (torch.Tensor): the connection diagram of the 
            compartments, which is built after calling build() method in the
            constructor. Read only (but actually can be modified).
        bidirection (int): whether to enable back-propagating current. 
            Read only.
        n_compartment (int): the number of dendritic compartment in a single
            neuron. Ready only.
        input_index, output_index (List[int]): a list of the input / output
            compartments' indices. Read only. 
            Notice that input_index = [0, 1, ..., n_input-1]
        n_input, n_output (int): the number of input / ouptut compartments.
            Read only.
    """

    def __init__(
        self, n_compartment: int, n_input: int, output_index: List[int],
        bidirection: bool, *args, **kwargs
    ):
        """The constructor of BaseWiring.

        Args:
            n_compartment (int): the number of dendritic compartment in a single
                neuron.
            n_input (int): the number of input compartments. Notice that
                input_index = [0, 1, ..., n_input-1]
            output_index (List[int]): the number of output compartments.
            bidirection (bool): whether to enable back-propagating current.

        Raises:
            ValueError: invalid wiring.
        """
        self._n_compartment = n_compartment
        self._adjacency_matrix = torch.zeros(
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

    # adjacency_matrix: read-only
    @property
    def adjacency_matrix(self) -> torch.Tensor:
        return self._adjacency_matrix

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

    # output_index: read-only
    @property
    def output_index(self) -> List[int]:
        return self._output_index

    # n_input: read-only
    @property
    def n_input(self) -> int:
        return self._n_input

    # n_output: read-only
    @property
    def n_output(self) -> int:
        return len(self._output_index)

    def validation_check(self) -> bool:
        """Whether the attributes of this wiring diagram are valid.
        """
        # check the shape of the adjacency matrix
        ad_shape = self._adjacency_matrix.shape
        if not ad_shape == torch.Size([self.n_compartment, self.n_compartment]):
            return False

        # check the validity of n_input and output_index
        if (self.n_input < 0) or (self.n_input > self.n_compartment):
            return False
        if ((min(self.output_index) < 0)
            or (max(self.output_index) >= self.n_compartment)):
            return False

        # if all the tests are passed
        return True

    def _add_compartment_single_connection(self, src: int, dest: int):
        if (src < 0 or src >= self.n_compartment 
            or dest < 0 or dest >= self.n_compartment):
            raise ValueError(
                f"invalid dendritic compartment connection: "
                f"src = {src}, dest = {dest}"
            )
        self._adjacency_matrix[src, dest] = 1

    def _add_compartment_double_connection(self, n1: int, n2: int):
        self._add_compartment_single_connection(n1, n2)
        self._add_compartment_single_connection(n2, n1)

    def add_compartment_connection(self, n1: int, n2: int):
        """Set a connection between two compartments.

        This method acts as the "BuildPart() function" in the "Builder" design
        pattern, which implement a single step in the construction process of
        adjacency_matrix. If self.bidirection is True, two edges (forward and
        backward) will be added; otherwise, only the forward edge will be added.

        Args:
            n1 (int): from which compartment? (an index)
            n2 (int): to which compartment? (an index)
        """
        if self.bidirection:
            self._add_compartment_double_connection(n1, n2)
        else:
            self._add_compartment_single_connection(n1, n2)

    @abc.abstractmethod
    def build(self, *args, **kwargs):
        """Construct the wiring diagam step by step.

        This method acts as the "Director" in the "Builder" design pattern,
        which builds adjacency_matrix by calling add_compartment_connection()
        multiple times. Implement this method while defining a new subclass.
        """
        pass


class SegregatedDendWiring(BaseWiring):
    """The wiring of segregated dendritic compartments in a single neuron.

    In this model, all the dendritic compartments receive synaptic input,
    and all of them are connected to the soma. There are no interconnections
    between them. In other words, n_compartment = n_input = n_output, 
    and the adjacency matrix is empty.

    Attributes:
        See base class: BaseWiring.
    """

    def __init__(self, n_compartment: int):
        """The constructor of SegregatedDendWiring.

        Args:
            n_compartment (int): the number of dendritic compartments in a
                single neuron.
        """
        super().__init__(
            n_compartment = n_compartment, n_input = n_compartment,
            output_index = list(range(n_compartment)), bidirection = False
        )

    def build(self):
        pass


class Kto1DendWirng(BaseWiring):
    """The wiring diagram of a k-to-1 dendritic tree.

    In this model, there are two "layers" of dendritic comparments. There 
    are n_output output compartments, and each of them receive information 
    from k input compartments. Hence, there are (k+1) * n_output 
    dendritic compartments in total.

    Attributes:
        See base class: BaseWiring.
    """

    def __init__(
        self, k: int, n_output: int,
        n_input: Optional[int] = None, bidirection: bool = False
    ):
        """The constructor of Kto1DendWiring.

        Args:
            k (int): how many input compartments are wired to an output one?
            n_output (int): the number of output compartments.
            n_input (Optional[int], optional): the number of input compartments.
                Defaults to None. If not None, should be equal to k * n_output .
            bidirection (bool, optional): whether to enable back-propagating
                current. Defaults to False.

        Raises:
            ValueError: n_input is not None and does not equal k * n_output .
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
