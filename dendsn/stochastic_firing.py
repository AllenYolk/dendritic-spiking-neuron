import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class BaseStochasticFiring(nn.Module, abc.ABC):

    def __init__(self, spiking: bool = True):
        super().__init__()
        self._spiking = spiking

    @property
    def spiking(self) -> bool:
        return self._spiking

    @spiking.setter
    def spiking(self, spk: bool):
        self._spiking = spk

    @abc.abstractmethod
    def spiking_function(self, v: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def rate_function(self, v: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        if self.spiking:
            return self.spiking_function(v)
        else:
            return self.rate_function(v)

    def plot_firing_rate(self, ax = None):
        v = torch.arange(-1.25, 1.25, 0.025)
        r = self.rate_function(v)
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(v, r)
        ax.set(xlabel = "v", ylabel = "firing_rate")


@torch.jit.script
def _stochastic_firing(firing_rate: torch.Tensor):
    comp = torch.rand_like(firing_rate) # [0, 1)
    return (firing_rate >= comp).to(firing_rate)


@torch.jit.script
def tri_param_logistic(x: torch.Tensor, phi: float, beta: float, theta: float):
    return phi * ((beta * (x - theta)).sigmoid())


@torch.jit.script
def logistic_stochastic_firing_backward(
    grad_output: torch.Tensor, x: torch.Tensor,
    phi: float, beta: float, theta: float
):
    sg = (beta * (x - theta)).sigmoid()
    return grad_output * phi * (1. - sg) * sg * beta


class logistic_stochastic_firing(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, f_max: float, beta: float, theta: float):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.f_max = f_max
            ctx.beta = beta
            ctx.theta = theta
        firing_rate = tri_param_logistic(x, f_max, beta, theta).to(x)
        return _stochastic_firing(firing_rate)

    @staticmethod
    def backward(ctx, grad_output):
        return logistic_stochastic_firing_backward(
            grad_output, ctx.saved_tensors[0],
            ctx.f_max, ctx.beta, ctx.theta
        ), None, None, None


class LogisticStochasticFiring(BaseStochasticFiring):

    def __init__(
        self, f_thres: float, beta: float, theta: float = 0.,
        spiking: bool = True
    ):
        """
        A wrapper for logistic stochastic firing function.
        The firing rate can be computed as:
            freq(v) = 2*f_thres / (1 + exp(-beta * (x - theta)))
        Notice that if the function is used as the spiking function for
        spikingjelly.activation_based.BaseNode (i.e. the `surrogate_function`
        argument), theta should be set to 0 (since threshold has already been 
        subtracted from v in spikingjelly).

        Args:
            f_thres (float): f_thres = 0.5 * f_max
            beta (float)
            theta (float, optional): Defaults to 0. .
            spiking (bool, optional): whether returns spikes or firing rates. 
                Defaults to True (returns spikes).
        """
        super().__init__(spiking)
        self.f_thres = f_thres
        self.beta = beta
        self.theta = theta

    def spiking_function(self, v: torch.Tensor) -> torch.Tensor:
        return logistic_stochastic_firing.apply(
            v, 2*self.f_thres, self.beta, self.theta
        )

    def rate_function(self, v: torch.Tensor) -> torch.Tensor:
        return tri_param_logistic(v, 2*self.f_thres, self.beta, self.theta)


@torch.jit.script
def tri_param_exp(x: torch.Tensor, phi: float, beta: float, theta: float):
    return phi * torch.exp(beta * (x - theta))


@torch.jit.script
def exponential_stochastic_firing_backward(
    grad_output: torch.Tensor, x: torch.Tensor,
    phi: float, beta: float, theta: float
):
    expx = torch.exp(beta * (x - theta))
    return grad_output * phi * expx * beta


class exponential_stochastic_firing(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, x: torch.Tensor,
        f_thres: float, beta: float, theta: float
    ):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.f_thres = f_thres
            ctx.beta = beta
            ctx.theta = theta
        firing_rate = tri_param_exp(x, f_thres, beta, theta).to(x)
        return _stochastic_firing(firing_rate)

    @staticmethod
    def backward(ctx, grad_output):
        return exponential_stochastic_firing_backward(
            grad_output, ctx.saved_tensors[0],
            ctx.f_thres, ctx.beta, ctx.theta
        ), None, None, None


class ExpStochasticFiring(BaseStochasticFiring):

    def __init__(
        self, f_thres: float, beta: float, theta: float = 0.,
        spiking: bool = True
    ):
        """
        A wrapper for exponential stochastic firing function.
        The firing rate can be computed as:
            freq(v) = f_thres * exp(beta * (v - theta))
        Notice that if the function is used as the spiking function for
        spikingjelly.activation_based.BaseNode (i.e. the `surrogate_function`
        argument), theta should be set to 0. (since threshold has already been
        subtracted from v in spikingjelly).

        Args:
            f_thres (float)
            beta (float)
            theta (float, optional): Defaults to 0..
            spiking (bool, optional): whether returns spikes or firing rates. 
                Defaults to True (returns spikes).
        """
        super().__init__(spiking)
        self.f_thres = f_thres
        self.beta = beta
        self.theta = theta

    def spiking_function(self, v: torch.Tensor) -> torch.Tensor:
        return exponential_stochastic_firing.apply(
            v, self.f_thres, self.beta, self.theta
        )

    def rate_function(self, v: torch.Tensor) -> torch.Tensor:
        return tri_param_exp(v, self.f_thres, self.beta, self.theta)