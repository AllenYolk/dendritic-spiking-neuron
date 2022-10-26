from typing import Union

import torch


@torch.jit.script
def diff_mask_mult_sum(
    x1: torch.Tensor, x2: torch.Tensor,
    mask: torch.Tensor, factor: Union[torch.Tensor, float]
) -> torch.Tensor:
    # x1.shape = [..., n1], x2.shape = [..., n2]
    x_gap = x1.unsqueeze(-1) - x2.unsqueeze(-2) # x_gap.shape = [..., n1, n2]
    x_gap = x_gap * mask
    x_output = (x_gap * factor).sum(dim = -2)
    return x_output