from typing import Union

import torch


@torch.jit.script
def diff_mask_mult_sum_factor_tensor(
    x1: torch.Tensor, x2: torch.Tensor,
    mask: torch.Tensor, factor: torch.Tensor
) -> torch.Tensor:
    # x1.shape = [..., n1], x2.shape = [..., n2]
    x_gap = x1.unsqueeze(-1) - x2.unsqueeze(-2) # x_gap.shape = [..., n1, n2]
    x_gap = x_gap * mask
    x_output = (x_gap * factor).sum(dim = -2)
    return x_output


@torch.jit.script
def diff_mask_mult_sum_factor_float(
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
    if isinstance(factor, float):
        return diff_mask_mult_sum_factor_float(x1, x2, mask, factor)
    elif isinstance(factor, torch.Tensor):
        return diff_mask_mult_sum_factor_tensor(x1, x2, mask, factor)
    else: 
        raise TypeError(
            f"factor in diff_mask_mult_sum should be float or torch.Tensor, "
            f"but instead get {factor}."
        )