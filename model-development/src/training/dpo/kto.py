"""Small, testable pieces of the Kahneman--Tversky Optimization objective.

The trainer keeps the model forwards and chat-trajectory rendering local to the
training script.  This module contains only the scalar loss, which makes the
sign convention and class weighting independently testable.
"""

from __future__ import annotations

import torch


def kto_loss_from_logratios(
    desirable_logratios: torch.Tensor,
    undesirable_logratios: torch.Tensor,
    *,
    reference_point: torch.Tensor,
    beta: float,
    desirable_weight: float,
    undesirable_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return standard KTO loss and its desirable/undesirable components.

    ``reference_point`` is deliberately detached by the caller.  KTO's
    reference-point estimator controls saturation but is not differentiated
    through, following Ethayarajh et al. (2024).
    """
    if desirable_logratios.numel() == 0 or undesirable_logratios.numel() == 0:
        raise ValueError("KTO requires at least one desirable and one undesirable trajectory per local batch.")
    if beta <= 0:
        raise ValueError("KTO beta must be positive.")
    if desirable_weight <= 0 or undesirable_weight <= 0:
        raise ValueError("KTO desirable/undesirable weights must be positive.")

    desirable_losses = 1.0 - torch.sigmoid(beta * (desirable_logratios - reference_point))
    undesirable_losses = 1.0 - torch.sigmoid(beta * (reference_point - undesirable_logratios))
    loss = torch.cat(
        (desirable_weight * desirable_losses, undesirable_weight * undesirable_losses), dim=0
    ).mean()
    return loss, desirable_losses.mean(), undesirable_losses.mean()
