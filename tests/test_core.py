import math

import torch

from rl_scaniqa.advantages import generalized_advantage_estimation
from rl_scaniqa.losses import LossConfig, compute_total_loss, pairwise_rank_loss
from rl_scaniqa.policy import ScanPolicyGRU
from rl_scaniqa.ppo import LinearScheduler
from rl_scaniqa.rewards import compute_r_div_from_actions, grayscale_entropy, ssim_simple
from rl_scaniqa.viewport_discretization import (
    erp_to_all_viewports,
    index_to_yaw_pitch,
    make_uniform_viewport_centers,
)


def test_viewport_centers_and_projection_shapes():
    centers = make_uniform_viewport_centers(num_yaw=8, num_pitch=4)
    assert centers.shape == (32, 2)
    assert torch.all(centers[:, 0] >= -math.pi)
    assert torch.all(centers[:, 0] < math.pi)
    assert torch.all(centers[:, 1] > -math.pi / 2)
    assert torch.all(centers[:, 1] < math.pi / 2)

    yaw, pitch = index_to_yaw_pitch(0, num_yaw=8, num_pitch=4)
    assert math.isclose(yaw, float(centers[0, 0]))
    assert math.isclose(pitch, float(centers[0, 1]))

    erp = torch.linspace(0, 1, 2 * 4).reshape(1, 1, 2, 4).repeat(1, 3, 1, 1)
    viewports = erp_to_all_viewports(
        erp,
        num_yaw=2,
        num_pitch=1,
        fov_deg=(90.0, 90.0),
        out_hw=(8, 8),
    )
    assert viewports.shape == (1, 2, 3, 8, 8)
    assert torch.isfinite(viewports).all()


def test_policy_probabilities_and_mask():
    torch.manual_seed(0)
    policy = ScanPolicyGRU(d_feat=16, d_hidden=8, d_z=12, num_layers=1)
    global_feature = torch.randn(2, 16)
    candidates = torch.randn(2, 5, 16)
    hidden = policy.init_hidden(global_feature)
    mask = torch.zeros(2, 5)
    mask[:, 0] = -1e9

    logits, probabilities = policy(global_feature, candidates, hidden, mask=mask)
    assert logits.shape == (2, 5)
    assert probabilities.shape == (2, 5)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.all(probabilities[:, 0] < 1e-7)


def test_rewards_have_expected_reference_values():
    constant = torch.full((2, 1, 8, 8), 0.5)
    assert torch.allclose(grayscale_entropy(constant), torch.zeros(2), atol=1e-6)
    assert torch.allclose(ssim_simple(constant, constant), torch.ones(2), atol=1e-5)

    actions = torch.tensor([[0, 0], [1, 2]])
    diversity, group_ids = compute_r_div_from_actions(
        actions,
        group_index=torch.tensor([0, 0]),
        X=4,
    )
    expected = 0.2 * (3 / 4) - 0.2 * (1 / 3)
    assert torch.allclose(diversity, torch.tensor([expected]), atol=1e-6)
    assert torch.equal(group_ids, torch.tensor([0]))


def test_gae_keeps_parallel_environments_separate():
    rewards = torch.tensor([[1.0, 10.0], [1.0, 10.0]])
    values = torch.zeros_like(rewards)
    dones = torch.tensor([[False, False], [True, True]])
    next_value = torch.zeros(2)
    advantages, returns = generalized_advantage_estimation(
        rewards,
        values,
        dones,
        next_value,
        gamma=1.0,
        lam=1.0,
    )
    assert torch.equal(advantages, torch.tensor([[2.0, 20.0], [1.0, 10.0]]))
    assert torch.equal(returns, advantages)


def test_losses_are_finite_for_large_differences():
    loss = pairwise_rank_loss(
        torch.tensor([-1_000.0]),
        torch.tensor([1_000.0]),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
    )
    assert torch.isfinite(loss)

    predictions = {
        "pair_pred1": torch.tensor([0.8], requires_grad=True),
        "pair_pred2": torch.tensor([0.2], requires_grad=True),
        "clean": torch.tensor([0.8], requires_grad=True),
        "weak": torch.tensor([0.79], requires_grad=True),
        "mild": torch.tensor([0.7], requires_grad=True),
        "strong": torch.tensor([0.6], requires_grad=True),
        "A_aug": torch.tensor([0.75], requires_grad=True),
        "B_aug": torch.tensor([0.25], requires_grad=True),
    }
    ground_truth = {
        "pair_gt1": torch.tensor([0.9]),
        "pair_gt2": torch.tensor([0.1]),
        "gt_A": torch.tensor([0.9]),
        "gt_B": torch.tensor([0.1]),
    }
    total, details = compute_total_loss(predictions, ground_truth, LossConfig())
    total.backward()
    assert torch.isfinite(total)
    assert set(details) == {"L_mse", "L_rank", "L_cons", "L_triplet", "L_cross", "L_total"}


def test_linear_scheduler_handles_single_update():
    schedule = LinearScheduler(0.2, 0.1, total_steps=1)
    assert schedule.at(0) == 0.2
    assert schedule.at(1) == 0.1
