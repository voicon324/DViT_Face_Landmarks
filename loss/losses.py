import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.heatmap_utils import soft_argmax # Assuming utils package accessible

# AwingLoss and TotalLoss classes remain the same as in the notebook
class AwingLoss(nn.Module):
    """
    Implementation of Adaptive Wing Loss from:
    "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"
    """
    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5, reduction='mean'):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)
        self.reduction = reduction

        # Precompute constants that only depend on hyperparameters
        # Note: alpha - target_heatmaps makes A and C depend on the ground truth heatmap values.
        # This seems computationally intensive and potentially problematic if target_heatmap values are zero.
        # Let's re-read the paper's formula carefully.
        # Eq. 4 shows AWing(y, y^) where y and y^ are single values.
        # Eq. 5 applies it pixel-wise: L_heatmap = 1/(WH) * sum_{i,j} AWing(H_ij, H^_ij)
        # The formula for A and C inside AWing(y, y^) depends on 'y' (target value).
        # This means A and C need to be computed per pixel based on the target heatmap value H_ij.

        # Let's implement the per-pixel calculation within forward().

    def forward(self, pred_heatmaps, target_heatmaps):
        # pred_heatmaps, target_heatmaps: (B, N_lmk, H, W)
        delta = torch.abs(pred_heatmaps - target_heatmaps)

        # Calculate A and C per pixel based on target_heatmaps values
        # alpha_minus_y = self.alpha - target_heatmaps # Shape: (B, N_lmk, H, W)
        # Need to handle potential division by zero or invalid powers if alpha_minus_y <= 1?
        # Paper implies alpha > 1. If target_heatmap can be 1, alpha_minus_y can be small/zero.
        # Let's clamp target_heatmaps slightly below alpha for stability? Or check paper constraints.
        # Assuming target_heatmaps are between [0, 1]. Alpha is 2.1. alpha_minus_y > 1.1. Seems safe.

        # Factor C1 = (theta/epsilon)^(alpha - y - 1)
        # Factor C2 = 1 / (1 + (theta/epsilon)^(alpha - y))
        # A = omega * C2 * (alpha - y) * C1 * (1/epsilon)
        # C = theta * A - omega * log(1 + (theta/epsilon)^(alpha - y))

        # For numerical stability, calculate powers carefully
        theta_eps = self.theta / self.epsilon
        alpha_minus_target = self.alpha - target_heatmaps

        # Use log-sum-exp trick ideas? Or just compute directly and rely on clamp.
        # Clamp exponent to avoid overflow/underflow in pow
        exponent_val = alpha_minus_target * torch.log(torch.clamp(theta_eps, min=1e-6)) # log(theta/epsilon)
        # Clamp exponent_val range? Paper doesn't mention issues.
        pow_theta_eps = torch.exp(torch.clamp(exponent_val, max=50)) # (theta/epsilon)^(alpha-y)

        pow_theta_eps_minus_1 = torch.exp(torch.clamp(exponent_val - torch.log(torch.clamp(theta_eps, min=1e-6)), max=50)) # (theta/epsilon)^(alpha-y-1)
        # pow_theta_eps_minus_1 = pow_theta_eps / theta_eps # Simpler?

        A = self.omega * (1 / (1 + pow_theta_eps)) * alpha_minus_target * pow_theta_eps_minus_1 * (1 / self.epsilon)
        # Add small epsilon to log argument for stability
        C = self.theta * A - self.omega * torch.log(1 + pow_theta_eps + 1e-9)


        # Calculate loss based on delta threshold
        loss = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + torch.abs(delta / self.epsilon)**(alpha_minus_target) + 1e-9),
            A * delta - C
        )

        # Handle reduction
        if self.reduction == 'mean':
            # Mean over all dimensions (pixels, landmarks, batch)
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none' or invalid
            return loss

class TotalLoss(nn.Module):
    """
    Combined loss for intermediate supervision: SmoothL1 on coordinates + Awing on heatmaps.
    """
    def __init__(self, num_blocks, beta=1.0, weight_intermediate=1.2, aw_alpha=2.1, aw_omega=14, smooth_l1_beta=1.0):
        super().__init__()
        if num_blocks <= 0:
             raise ValueError("num_blocks must be positive for TotalLoss")
        self.num_blocks = num_blocks
        self.beta = beta # Weight for heatmap loss relative to coordinate loss (Eq 5)
        self.w = weight_intermediate # Base weight for intermediate supervision (Eq 6)

        # Coordinate Loss (on soft-argmax outputs)
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean', beta=smooth_l1_beta)

        # Heatmap Loss
        self.aw_loss = AwingLoss(alpha=aw_alpha, omega=aw_omega, reduction='mean')

    def forward(self, intermediate_heatmaps, gt_coords_feat_scale, gt_heatmaps):
        """
        Args:
            intermediate_heatmaps (list): List of heatmap tensors from each block (B, N_lmk, H, W).
            gt_coords_feat_scale (tensor): Ground truth landmark coordinates scaled to feature map size (B, N_lmk, 2).
            gt_heatmaps (tensor): Ground truth heatmaps (B, N_lmk, H, W).
        """
        total_loss = 0.0
        num_stages = len(intermediate_heatmaps)
        if num_stages != self.num_blocks:
             print(f"Warning: Expected {self.num_blocks} heatmap stages, but got {num_stages}.")
             # Option: Use num_stages instead of self.num_blocks for weighting?
             # Or raise an error? Let's use num_stages for flexibility.
             # raise ValueError(f"Number of intermediate heatmaps ({num_stages}) does not match num_blocks ({self.num_blocks})")

        for j in range(num_stages): # j from 0 to num_stages-1
            pred_heatmaps_j = intermediate_heatmaps[j] # (B, N_lmk, H, W)

            # 1. Coordinate Loss (d1 in paper)
            # Use soft-argmax to get coordinates from predicted heatmap
            pred_coords_j = soft_argmax(pred_heatmaps_j) # (B, N_lmk, 2) - scaled to feature map size
            coord_loss = self.smooth_l1_loss(pred_coords_j, gt_coords_feat_scale)

            # 2. Heatmap Loss (d2 in paper)
            heatmap_loss = self.aw_loss(pred_heatmaps_j, gt_heatmaps)

            # 3. Combine losses for stage j (Lj in paper)
            stage_loss = coord_loss + self.beta * heatmap_loss

            # 4. Apply intermediate supervision weight (Eq 6: w^(j-B+1))
            # Note: j is 0-indexed, B is num_blocks (1-indexed count)
            # Stage 1 (j=0): weight = w^(0 - num_stages + 1) = w^(1-num_stages)
            # Final Stage (j=num_stages-1): weight = w^((num_stages-1) - num_stages + 1) = w^0 = 1
            weight = self.w ** (j - num_stages + 1)
            total_loss += weight * stage_loss

            # Debug print (optional)
            # print(f"Stage {j+1}/{num_stages}: Weight={weight:.4f}, CoordL={coord_loss:.4f}, HeatmapL={heatmap_loss:.4f}, StageL={stage_loss:.4f}")


        # Average loss over stages? Or just sum weighted losses?
        # Paper's Eq 6 is a sum: L_total = sum_{j=1 to B} [ w^(j-B) * Lj ] -> assuming Lj uses B=num_blocks
        # Let's stick to the sum of weighted stage losses.
        return total_loss