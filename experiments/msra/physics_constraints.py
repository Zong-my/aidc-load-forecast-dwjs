"""
Module D: Physical Constraints and Mixture Output.

Enforces physically-grounded constraints on predictions:
  - Power bounds: P_min ≤ ŷ ≤ P_max
  - Slew rate: |dŷ/dt| ≤ ΔP_max per timestep
  - Quantile monotonicity: q_low ≤ q_mid ≤ q_high

Combines expert predictions weighted by regime transition probabilities
into a final probabilistic forecast.

General-purpose: bounds and rates are configurable parameters.
"""

import torch
import torch.nn as nn


class PhysicsConstraintLayer(nn.Module):
    """Physical constraint enforcement for power predictions.

    Hard constraints applied at inference; soft penalties during training.
    """

    def __init__(self, p_min=0.0, p_max=50.0, max_slew_rate=None,
                 lambda_slew=0.05, lambda_bound=0.1):
        super().__init__()
        self.register_buffer("p_min", torch.tensor(p_min, dtype=torch.float32))
        self.register_buffer("p_max", torch.tensor(p_max, dtype=torch.float32))
        self.lambda_slew = lambda_slew
        self.lambda_bound = lambda_bound

        if max_slew_rate is not None:
            self.register_buffer("max_slew",
                                 torch.tensor(max_slew_rate, dtype=torch.float32))
        else:
            self.max_slew = None

    def apply_hard_constraints(self, pred):
        """Apply hard physical constraints (inference time).

        Args:
            pred: (batch, horizon, n_quantiles) in original scale (kW)

        Returns:
            Constrained predictions, same shape.
        """
        # Bound clipping
        pred = torch.clamp(pred, self.p_min.item(), self.p_max.item())

        # Slew rate enforcement (sequential, step by step)
        if self.max_slew is not None:
            sr = self.max_slew.item()
            for h in range(1, pred.shape[1]):
                delta = pred[:, h, :] - pred[:, h - 1, :]
                delta_clipped = torch.clamp(delta, -sr, sr)
                pred[:, h, :] = pred[:, h - 1, :] + delta_clipped

            # Re-clip after slew enforcement
            pred = torch.clamp(pred, self.p_min.item(), self.p_max.item())

        return pred

    def compute_penalty(self, pred):
        """Compute soft constraint violation penalty (training time).

        Args:
            pred: (batch, horizon, n_quantiles) in original scale (kW)

        Returns:
            Scalar penalty loss
        """
        penalty = torch.tensor(0.0, device=pred.device)

        # Bound violation
        bound_viol = (torch.relu(pred - self.p_max) +
                      torch.relu(self.p_min - pred))
        penalty = penalty + self.lambda_bound * bound_viol.mean()

        # Slew rate violation
        if self.max_slew is not None:
            diffs = torch.abs(pred[:, 1:, :] - pred[:, :-1, :])
            slew_viol = torch.relu(diffs - self.max_slew)
            penalty = penalty + self.lambda_slew * slew_viol.mean()

        return penalty


class MixtureOutput(nn.Module):
    """Combine K expert predictions weighted by regime transition probabilities.

    ŷ(t+h) = Σ_k π_k(t+h) × f_k(X(t))

    Handles quantile crossing correction.
    """

    def __init__(self, quantile_levels=None):
        super().__init__()
        if quantile_levels is None:
            quantile_levels = [0.1, 0.5, 0.9]
        self.register_buffer(
            "quantile_levels",
            torch.tensor(quantile_levels, dtype=torch.float32))

    def forward(self, expert_preds, regime_weights):
        """Compute weighted mixture of expert predictions.

        Args:
            expert_preds: (batch, K, horizon, n_quantiles)
            regime_weights: (batch, horizon, K)

        Returns:
            (batch, horizon, n_quantiles) weighted mixture
        """
        # regime_weights: (batch, horizon, K) → (batch, K, horizon, 1)
        w = regime_weights.permute(0, 2, 1).unsqueeze(-1)

        # Weighted sum: (batch, K, horizon, n_q) * (batch, K, horizon, 1) → sum over K
        mixed = (expert_preds * w).sum(dim=1)  # (batch, horizon, n_quantiles)

        # Fix quantile crossing: sort along quantile dimension
        mixed = self.fix_quantile_crossing(mixed)

        return mixed

    def fix_quantile_crossing(self, pred):
        """Ensure quantile monotonicity: q_low ≤ q_mid ≤ q_high.

        Args:
            pred: (batch, horizon, n_quantiles)

        Returns:
            Sorted predictions, same shape
        """
        return torch.sort(pred, dim=-1).values


def pinball_loss(pred, target, quantile_levels):
    """Quantile (pinball) loss for probabilistic forecasting.

    Args:
        pred: (batch, horizon, n_quantiles) predicted quantiles
        target: (batch, horizon) or (batch, horizon, 1) ground truth
        quantile_levels: tensor of shape (n_quantiles,), e.g. [0.1, 0.5, 0.9]

    Returns:
        Scalar loss
    """
    if target.dim() == 2:
        target = target.unsqueeze(-1)  # (batch, horizon, 1)

    # quantile_levels shape: (1, 1, n_quantiles)
    tau = quantile_levels.view(1, 1, -1).to(pred.device)

    errors = target - pred  # (batch, horizon, n_quantiles)
    loss = torch.maximum(tau * errors, (tau - 1) * errors)
    return loss.mean()
