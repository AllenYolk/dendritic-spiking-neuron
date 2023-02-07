import abc

from spikingjelly.activation_based import base


class BaseLearner(base.MemoryModule, abc.ABC):

    def __init__(self, step_mode: str = "s"):
        super().__init__()
        self.step_mode = step_mode

    def reset(self):
        super().reset()

    @abc.abstractmethod
    def enable(self):
        pass

    @abc.abstractmethod
    def disable(self):
        pass

    @abc.abstractmethod
    def step(self, *args, **kwargs):
        pass