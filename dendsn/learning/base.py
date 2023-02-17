import abc
from typing import Optional, Iterable

import torch.nn as nn
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


class LearnerList(nn.ModuleList):

    def __init__(self, learners: Optional[Iterable[BaseLearner]] = None):
        super().__init__(learners)

    def reset(self):
        for learner in self:
            learner.reset()

    def enable(self):
        for learner in self:
            learner.enable()

    def disable(self):
        for learner in self:
            learner.disable()

    def step(self, *args, **kwargs):
        for learner in self:
            learner.step(*args, **kwargs)