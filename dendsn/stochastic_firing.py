"""Stochastic firing modules.

A stochastic firing module maps a membrane potential to the probability of
emitting a spike in a time step (a.k.a. firing rate), and then generates a spike
(or not) according to the probability. 
"""
import abc

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class BaseStochasticFiring(nn.Module, abc.ABC):
    """Base class for all stochastic firing modules.

    Attributes:
        spiking (bool): if true, the output of forward() is a spike tensor;
            otherwise, the output is the firing probability (or firing rate).
    """

    def __init__(self, spiking: bool = True):
        """The constructor of BaseStochasticFiring.

        To define a subclass, just implement 2 methods: spiking_function() and 
        rate_function().

        Args:
            spiking (bool, optional): if true, the output of forward() is a 
            spike tensor; otherwise, the output is the firing probability (or 
            firing rate). Defaults to True.
        """
        super().__init__()
        self.spiking = spiking

    @abc.abstractmethod
    def spiking_function(self, v: torch.Tensor) -> torch.Tensor:
        """Map membrane potential to spike according to certain firing rate.

        This function must support gradient back propagation. Typically, a 
        spiking function is implemented by directly calling a customized 
        subclass of torch.autograd.Function.

        Args:
            v (torch.Tensor): membrane potential tensor.

        Returns:
            torch.Tensor: spike tensor that hasa the same shape as v.
        """
        pass

    @abc.abstractmethod
    def rate_function(self, v: torch.Tensor) -> torch.Tensor:
        """Map membrane potential to firing probability (or firing rate).

        Args:
            v (torch.Tensor): membrane potential tensor.

        Returns:
            torch.Tensor: a tensor of firing probability (firing rate) that has
                the same shape as v.
        """
        pass

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        if self.spiking:
            return self.spiking_function(v)
        else:
            return self.rate_function(v)

    def plot_firing_rate(self, f=None, ax=None):
        """Plot the firing rate function.

        Args:
            f, ax (optional): the matplotlib figure and axes to place the 
                picture. If one of them is None, a new pair of f, ax will be 
                created inside the function.
        Returns:
            f, ax: the matplotlib figure and axes that contain the picture.
        """
        v = torch.arange(-1.25, 1.25, 0.025)
        r = self.rate_function(v)
        if (f is None) or (ax is None):
            f, ax = plt.subplots()
        ax.plot(v, r)
        ax.set(xlabel = "v", ylabel = "firing_rate")
        return f, ax


@torch.jit.script
def _stochastic_firing(firing_rate: torch.Tensor) -> torch.Tensor:
    comp = torch.rand_like(firing_rate) # [0, 1)
    return (firing_rate >= comp).to(firing_rate)


@torch.jit.script
def _tri_param_logistic(
    x: torch.Tensor, phi: float, beta: float, theta: float
) -> torch.Tensor:
    return phi * ((beta * (x - theta)).sigmoid())


@torch.jit.script
def _logistic_stochastic_firing_backward(
    grad_output: torch.Tensor, x: torch.Tensor,
    phi: float, beta: float, theta: float
) -> torch.Tensor:
    sg = (beta * (x - theta)).sigmoid()
    return grad_output * phi * (1. - sg) * sg * beta


class logistic_stochastic_firing(torch.autograd.Function):
    """The autograd function for 3-param-logistic stochastic spiking module.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, f_max: float, beta: float, theta: float):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.f_max = f_max
            ctx.beta = beta
            ctx.theta = theta
        firing_rate = _tri_param_logistic(x, f_max, beta, theta).to(x)
        return _stochastic_firing(firing_rate)

    @staticmethod
    def backward(ctx, grad_output):
        return _logistic_stochastic_firing_backward(
            grad_output, ctx.saved_tensors[0],
            ctx.f_max, ctx.beta, ctx.theta
        ), None, None, None


class LogisticStochasticFiring(BaseStochasticFiring):
    """3-param-logistic stochastic spiking module.

    The firing rate is:
        freq(v) = 2*f_thres / (1 + exp(-beta * (x - theta)))
    """

    def __init__(
        self, f_thres: float, beta: float, theta: float = 0.,
        spiking: bool = True
    ):
        """The constructor of LogisticStochasticFiring.

        Notice: if the function is used as the spiking function for
        spikingjelly.activation_based.BaseNode (i.e. the `surrogate_function`
        argument) or dendsn.model.soma.BaseSoma, theta should be set to 0 (since 
        threshold has already been subtracted from v).

        Args:
            f_thres (float): the firing rate at v=theta. f_thres = 0.5 * f_max
            beta (float): the steepness of the probability function curve.
            theta (float, optional): adjust the probability function
                horizontally. Defaults to 0. .
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
        return _tri_param_logistic(v, 2*self.f_thres, self.beta, self.theta)


@torch.jit.script
def _tri_param_exp(
    x: torch.Tensor, phi: float, beta: float, theta: float
) -> torch.Tensor:
    return phi * torch.exp(beta * (x - theta))


@torch.jit.script
def _exponential_stochastic_firing_backward(
    grad_output: torch.Tensor, x: torch.Tensor,
    phi: float, beta: float, theta: float
) -> torch.Tensor:
    expx = torch.exp(beta * (x - theta))
    return grad_output * phi * expx * beta


class exponential_stochastic_firing(torch.autograd.Function):
    """The autograd function for exponential stochastic spiking module.
    """

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
        firing_rate = _tri_param_exp(x, f_thres, beta, theta).to(x)
        return _stochastic_firing(firing_rate)

    @staticmethod
    def backward(ctx, grad_output):
        return _exponential_stochastic_firing_backward(
            grad_output, ctx.saved_tensors[0],
            ctx.f_thres, ctx.beta, ctx.theta
        ), None, None, None


class ExpStochasticFiring(BaseStochasticFiring):
    """Exponential stochastic spiking module.

    The firing rate is:
        freq(v) = f_thres * exp(beta * (v - theta))
    """

    def __init__(
        self, f_thres: float, beta: float, theta: float = 0.,
        spiking: bool = True
    ):
        """The constructor of ExpStochasticFiring.

        Notice: if the function is used as the spiking function for
        spikingjelly.activation_based.BaseNode (i.e. the `surrogate_function`
        argument) or dendsn.model.soma.BaseSoma, theta should be set to 0 (since 
        threshold has already been subtracted from v).

        Args:
            f_thres (float): the firing rate at v=theta.
            beta (float): the steepness of the probability function curve.
            theta (float, optional): adjust the probability function
                horizontally. Defaults to 0. .
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
        return _tri_param_exp(v, self.f_thres, self.beta, self.theta)