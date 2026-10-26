"""
Loss functions for multi-class classification (Stage 2 pipeline).

Implements Focal Loss (Lin et al. 2017), Guided Precision Loss (Shaban 2021, Zhai 2022),
Hybrid Loss (Basit 2022), and BCE Loss with factory function for flexible loss configuration.

The 5-class output (Sea, Oil, Look-alike, Ship, Land) uses binary cross-entropy per class,
with focus on maximizing Oil (Class 1) recall as the primary metric.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in multi-class classification.

    Reference: Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    "Focal Loss for Dense Object Detection." ICCV.

    The focal loss applies a modulating term (1 - p_t)^gamma to the cross-entropy loss,
    down-weighting easy negatives and focusing training on hard examples. This is
    particularly useful for imbalanced SAR datasets where non-oil patches dominate.

    Args:
        alpha (float): Weighting factor in [0, 1] to balance positive/negative examples.
                      Default: 0.25 (balances the BCE contribution)
        gamma (float): Exponent of the modulating factor (1 - p_t) to control how much
                      to down-weight easy examples. Higher values focus more on hard
                      examples. Default: 2.0

    Shape:
        - Input (logits): (batch_size, num_classes)
        - Targets: (batch_size, num_classes) binary labels {0, 1}
        - Output: scalar loss (mean over batch and classes)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw model outputs (pre-sigmoid), shape (batch_size, num_classes)
            targets: Binary labels, shape (batch_size, num_classes), dtype float32

        Returns:
            Scalar loss (mean reduction over all elements)
        """
        # Compute sigmoid probabilities
        probs = torch.sigmoid(logits)

        # Binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Compute p_t (probability of target class)
        # If target=1, p_t = p (probability of class 1)
        # If target=0, p_t = 1-p (probability of class 0)
        p_t = torch.where(targets == 1, probs, 1 - probs)

        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Combine: focal_loss = -alpha * (1 - p_t)^gamma * log(p_t)
        focal_loss = alpha_weight * focal_weight * bce_loss

        return focal_loss.mean()


class GPLoss(nn.Module):
    """
    Guided Precision Loss (GP Loss) for SAR oil spill detection.

    References:
    - Shaban, A., Maracani, A., Kahatapitiya, K., & Perina, A. (2021).
      "Guided Precision: Increasing Underrepresented Recall." ICCV.
    - Zhai, X., Kolesnikov, A., Houlsby, N., & Beyer, L. (2022).
      "Scaling Vision Transformers." CVPR.

    GP Loss is specifically designed to maximize recall on underrepresented classes
    (e.g., Oil in SAR imagery). It penalizes false negatives by enforcing that
    positive examples (target=1) have high logit values, while allowing normal
    cross-entropy for negative examples.

    For SAR oil spill detection, this is critical: missing an oil spill (false negative)
    is worse than a false alarm (false positive).

    Args:
        margin (float): Minimum logit value for positive examples. Positive examples
                       with logit < margin incur an additional penalty. Default: 0.1

    Shape:
        - Input (logits): (batch_size, num_classes)
        - Targets: (batch_size, num_classes) binary labels {0, 1}
        - Output: scalar loss (mean over batch and classes)
    """

    def __init__(self, margin: float = 0.1) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute guided precision loss.

        Args:
            logits: Raw model outputs (pre-sigmoid), shape (batch_size, num_classes)
            targets: Binary labels, shape (batch_size, num_classes), dtype float32

        Returns:
            Scalar loss (mean reduction over all elements)
        """
        # Standard binary cross-entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # For positive examples (target=1), apply margin penalty if logit < margin
        # This encourages the model to be confident on oil patches
        margin_violation = torch.relu(self.margin - logits)
        margin_loss = targets * margin_violation

        # Total loss: BCE + margin penalty for positives
        total_loss = bce_loss + margin_loss

        return total_loss.mean()


class HybridLoss(nn.Module):
    """
    Hybrid Loss combining Focal Loss, Guided Precision Loss, and Binary Cross-Entropy.

    Reference: Basit, S., Farooq, M., Irfan, M., Younis, M., & Mustansar, A. (2022).
    "Class-wise and Balanced Loss for Solving Class-Imbalance Problems in Object Detection."
    arXiv:2208.06049.

    The hybrid loss balances three objectives:
    1. Focal Loss: Focus training on hard negatives
    2. GP Loss: Maximize recall (penalize false negatives)
    3. BCE Loss: Standard classification loss for overall calibration

    This combination is particularly effective for SAR oil spill detection, where:
    - Class imbalance (few oil patches) requires focal weighting
    - Recall on oil (Class 1) is critical, requiring GP guidance
    - Overall model calibration requires standard BCE baseline

    Args:
        focal_weight (float): Weight for focal loss component. Default: 1.0
        gp_weight (float): Weight for GP loss component. Default: 0.1
        ce_weight (float): Weight for cross-entropy component. Default: 1.0
        focal_gamma (float): Gamma parameter for focal loss. Default: 2.0
        focal_alpha (float): Alpha parameter for focal loss. Default: 0.25
        gp_margin (float): Margin parameter for GP loss. Default: 0.1

    Shape:
        - Input (logits): (batch_size, num_classes)
        - Targets: (batch_size, num_classes) binary labels {0, 1}
        - Output: scalar loss (mean over batch and classes)
    """

    def __init__(
        self,
        focal_weight: float = 1.0,
        gp_weight: float = 0.1,
        ce_weight: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        gp_margin: float = 0.1,
    ) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.gp_weight = gp_weight
        self.ce_weight = ce_weight

        # Create loss function instances
        self.focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.gp_loss_fn = GPLoss(margin=gp_margin)
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute hybrid loss as weighted combination of focal, GP, and BCE losses.

        Args:
            logits: Raw model outputs (pre-sigmoid), shape (batch_size, num_classes)
            targets: Binary labels, shape (batch_size, num_classes), dtype float32

        Returns:
            Scalar loss (weighted sum of components)
        """
        focal = self.focal_loss_fn(logits, targets)
        gp = self.gp_loss_fn(logits, targets)
        bce = self.bce_loss_fn(logits, targets)

        hybrid = (
            self.focal_weight * focal
            + self.gp_weight * gp
            + self.ce_weight * bce
        )

        return hybrid


def build_loss(loss_type: str, config) -> nn.Module:
    """
    Factory function to build a loss module based on configuration.

    This function centralizes loss function creation, ensuring consistent
    parameter passing from the TrainerConfig to the appropriate loss class.

    Args:
        loss_type (str): Type of loss function. One of:
            - 'focal': Focal Loss (Lin et al. 2017)
            - 'gp': Guided Precision Loss (Shaban 2021, Zhai 2022)
            - 'hybrid': Hybrid Loss (Basit 2022) — recommended for SAR oil spill detection
            - 'bce': Standard Binary Cross-Entropy (BCE) Loss
        config: TrainerConfig object containing loss hyperparameters:
            - focal_alpha, focal_gamma: for FocalLoss
            - gp_margin: for GPLoss
            - focal_weight, gp_weight, ce_weight, focal_gamma, focal_alpha, gp_margin: for HybridLoss

    Returns:
        nn.Module: Instantiated loss function

    Raises:
        ValueError: If loss_type is not one of the supported types

    Example:
        >>> from training.config import TrainerConfig
        >>> from training.losses import build_loss
        >>> config = TrainerConfig(loss_type='hybrid')
        >>> loss_fn = build_loss('hybrid', config)
        >>> logits = torch.randn(8, 5)
        >>> targets = torch.randint(0, 2, (8, 5)).float()
        >>> loss = loss_fn(logits, targets)
    """
    if loss_type == 'focal':
        return FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)

    elif loss_type == 'gp':
        return GPLoss(margin=config.gp_margin)

    elif loss_type == 'hybrid':
        return HybridLoss(
            focal_weight=config.focal_weight,
            gp_weight=config.gp_weight,
            ce_weight=config.ce_weight,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha,
            gp_margin=config.gp_margin,
        )

    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss()

    else:
        raise ValueError(
            f"Unknown loss_type: '{loss_type}'. "
            f"Supported types: 'focal', 'gp', 'hybrid', 'bce'"
        )
