import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron as sj_neuron

from dendsn import dendrite


class DendNeuron(nn.Module):

    def __init__(
        self, dend: dendrite.BaseDend, soma: sj_neuron.BaseNode
    ):
        super().__init__()
        self.dend = dend
        self.soma = soma