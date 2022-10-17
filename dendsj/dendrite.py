import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base


class DendCompartment(base.MemoryModule):

    def __init__(self, tau: float = 2):
        super().__init__()
        self.register_memory("v", 0)
        self.tau = tau