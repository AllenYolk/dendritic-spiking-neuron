import torch
import torch.nn.functional as F


@torch.jit.script
def _stochastic_firing(firing_rate: torch.Tensor):
    comp = torch.rand_like(firing_rate) # [0, 1)
    return (firing_rate >= comp).to(firing_rate)


@torch.jit.script
def tri_param_logistic(x: torch.Tensor, phi: float, beta: float, theta: float):
    return phi * F.sigmoid(beta * (x - theta))


@torch.jit.script
def logistic_stochastic_firing_backward(
    grad_output: torch.Tensor, x: torch.Tensor,
    phi: float, beta: float, theta: float
):
    sg = F.sigmoid(beta * (x - theta))
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
