from typing import Optional, Tuple

import torch
from torch import Tensor


def masked_mean_std(
    x: Tensor, n: Optional[Tensor] = None, mask: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    `x`: [days, stocks], input data
    `n`: [days], should be `(~mask).sum(dim=1)`, provide this to avoid unnecessary computations
    `mask`: [days, stocks], data masked as `True` will not participate in the computation, \
    defaults to `torch.isnan(x)`
    """
    if mask is None:
        mask = torch.isnan(x)
    if n is None:
        n = (~mask).sum(dim=1)
    x = x.clone()
    x[mask] = 0.0
    mean = x.sum(dim=1) / n
    std = ((((x - mean[:, None]) * ~mask) ** 2).sum(dim=1) / n).sqrt()

    return mean, std


def normalize_by_day(value: Tensor) -> Tensor:
    "The shape of the input and the output is (days, stocks)"
    mean, std = masked_mean_std(value)
    value = (value - mean[:, None]) / std[:, None]
    # nan_mask = torch.isnan(value)
    # value[nan_mask] = 0.
    return value


def nanstd(tensor: torch.Tensor, dim: int, keepdim: bool = False):
    """Compute the standard deviation manually while ignoring NaNs."""
    # Mask NaN values
    valid_mask = ~torch.isnan(tensor)
    count = valid_mask.sum(dim=dim, keepdim=True).float()

    # Compute the mean, while ignoring NaNs (ensure broadcasting along the correct dimension)
    mean = torch.nansum(tensor, dim=dim, keepdim=True) / count.clamp(min=1)

    # Compute variance: sum of squared differences from the mean, divided by (count - 1)
    variance = torch.nansum((tensor - mean) ** 2, dim=dim, keepdim=True) / (
        count - 1
    ).clamp(min=1)

    # Standard deviation is the square root of variance
    std = torch.sqrt(variance)

    if not keepdim:
        std = std.squeeze(dim=dim)
        count = count.squeeze(dim=dim)

    # replace the place where the whole dim are nans
    std[count == 0] = torch.nan

    return std


def nanvar(tensor: torch.Tensor, dim: int, keepdim: bool = False):
    """Compute the standard deviation manually while ignoring NaNs."""
    # Mask NaN values
    valid_mask = ~torch.isnan(tensor)
    count = valid_mask.sum(dim=dim, keepdim=True).float()

    # Compute the mean, while ignoring NaNs (ensure broadcasting along the correct dimension)
    mean = torch.nansum(tensor, dim=dim, keepdim=True) / count.clamp(min=1)

    # Compute variance: sum of squared differences from the mean, divided by (count - 1)
    variance = torch.nansum((tensor - mean) ** 2, dim=dim, keepdim=True) / count.clamp(
        min=1
    )

    if not keepdim:
        variance = variance.squeeze(dim=dim)
        count = count.squeeze(dim=dim)

    # replace the place where the whole dim are nans
    variance[count == 0] = torch.nan

    return variance
