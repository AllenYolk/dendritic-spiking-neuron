import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron as sj_neuron

from dendsj import dendrite


class DendNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.soma = sj_neuron.LIFNode()
        self.dend = dendrite.DendCompartment()

