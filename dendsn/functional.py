"""Tool functions.

This module defines a series of tool functions that are widely used throughout
`dendsn` package. These functions are "abstract" mathematical or logical 
operations and are not closely related neuron models or learning algorithms, so
it's reasonable to place them here.
"""

from typing import Union, Sequence, Callable

import torch
import torch.nn as nn


@torch.jit.script
def _diff_mask_mult_sum_factor_tensor(
    x1: torch.Tensor, x2: torch.Tensor,
    mask: torch.Tensor, factor: torch.Tensor
) -> torch.Tensor:
    # x1.shape = [..., n1], x2.shape = [..., n2]
    x_gap = x1.unsqueeze(-1) - x2.unsqueeze(-2) # x_gap.shape = [..., n1, n2]
    x_gap = x_gap * mask
    x_output = (x_gap * factor).sum(dim = -2)
    return x_output


@torch.jit.script
def _diff_mask_mult_sum_factor_float(
    x1: torch.Tensor, x2: torch.Tensor,
    mask: torch.Tensor, factor: float
) -> torch.Tensor:
    # x1.shape = [..., n1], x2.shape = [..., n2]
    x_gap = x1.unsqueeze(-1) - x2.unsqueeze(-2) # x_gap.shape = [..., n1, n2]
    x_gap = x_gap * mask
    x_output = (x_gap * factor).sum(dim = -2)
    return x_output


def diff_mask_mult_sum(
    x1: torch.Tensor, x2: torch.Tensor,
    mask: torch.Tensor, factor: Union[float, torch.Tensor]
) -> torch.Tensor:
    """Used when computing `input_internal` in dendrites.

    First, compute the tensor x_gap:
        x_grap[..., i, j] = x1[..., i] - x2[..., j]
    Then, apply the mask and the factor:
        x_grap = x_grap * mask * factor
    Finally, do summation along dim = -2
        x_output = x_grap.sum(dim = -2)
    Hence, x_output.shape = x2.shape

    Args:
        x1 (torch.Tensor)
        x2 (torch.Tensor)
        mask (torch.Tensor)
        factor (Union[float, torch.Tensor])

    Returns:
        A torch.Tensor with the same shape as x2.
    """
    if isinstance(factor, float):
        return _diff_mask_mult_sum_factor_float(x1, x2, mask, factor)
    elif isinstance(factor, torch.Tensor):
        return _diff_mask_mult_sum_factor_tensor(x1, x2, mask, factor)
    else: 
        raise TypeError(
            f"factor in diff_mask_mult_sum should be float or torch.Tensor, "
            f"but instead get {factor}."
        )


def unfold_forward_fold(
    x_seq: torch.Tensor, 
    stateless_module: Union[Sequence, nn.Module, nn.Sequential, Callable]
) -> torch.Tensor:
    """Used in synaptic connection modules (wrappers of nn.Conv2d, ...).

    Merge the time and batch dimensions of `x_seq` into one dimension ('fold'),
    conduct the computations defined in `stateless_module`,
    and resume the time and batch dimensions of the outcome ('unfold').
    The shape of the tensor: [T, N, *input_shape] -> [T*N, *input_shape]
    -> [T, N, *input_shape]

    Args:
        x_seq (torch.Tensor): with shape [T, N, *input_shape]
        stateless_module (Union[Sequence, nn.Module, nn.Sequential, Callable])

    Returns:
        torch.Tensor: with shape [T, N, *output_shape]
    """
    y_shape = [x_seq.shape[0], x_seq.shape[1]]
    y = x_seq.flatten(start_dim = 0, end_dim = 1)
    if isinstance(stateless_module, (list, tuple, nn.Sequential)):
        for m in stateless_module:
            y = m(y)
    else:
        y = stateless_module(y)
    y_shape.extend(y.shape[1:])
    return y.view(y_shape)