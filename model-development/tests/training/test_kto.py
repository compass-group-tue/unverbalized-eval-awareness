"""Unit tests for the scalar KTO objective."""

import torch

from training.dpo.kto import kto_loss_from_logratios


def loss(chosen, rejected, *, beta=0.1):
    return kto_loss_from_logratios(
        torch.tensor(chosen, dtype=torch.float32),
        torch.tensor(rejected, dtype=torch.float32),
        reference_point=torch.tensor(0.0),
        beta=beta,
        desirable_weight=1.0,
        undesirable_weight=1.0,
    )


def test_zero_logratios_have_half_loss_per_class():
    total, desirable, undesirable = loss([0.0], [0.0])
    assert torch.allclose(total, torch.tensor(0.5))
    assert torch.allclose(desirable, torch.tensor(0.5))
    assert torch.allclose(undesirable, torch.tensor(0.5))


def test_increasing_desirable_and_decreasing_undesirable_reduce_kto_loss():
    baseline, _, _ = loss([0.0], [0.0], beta=1.0)
    improved, _, _ = loss([2.0], [-2.0], beta=1.0)
    assert improved < baseline


def test_class_weights_scale_their_respective_losses_before_batch_mean():
    total, desirable, undesirable = kto_loss_from_logratios(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        reference_point=torch.tensor(0.0),
        beta=1.0,
        desirable_weight=2.0,
        undesirable_weight=1.0,
    )
    assert torch.allclose(desirable, torch.tensor(0.5))
    assert torch.allclose(undesirable, torch.tensor(0.5))
    assert torch.allclose(total, torch.tensor(0.75))
