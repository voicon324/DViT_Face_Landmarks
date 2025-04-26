import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.heatmap_utils import soft_argmax # Assuming utils package accessible

class AwingLoss(nn.Module):
    """
    Triển khai Awing Loss từ bài báo:
    "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"
    """
    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5, reduction='mean'):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)
        self.reduction = reduction

    def forward(self, pred_heatmaps, target_heatmaps):
        # pred_heatmaps, target_heatmaps: (B, N_lmk, H, W)
        delta = torch.abs(pred_heatmaps - target_heatmaps)

        # Tính A và C
        A = self.omega * (1 / (1 + (self.theta / self.epsilon)**(self.alpha - target_heatmaps))) * \
            (self.alpha - target_heatmaps) * ((self.theta / self.epsilon)**(self.alpha - target_heatmaps - 1)) * \
            (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + (self.theta / self.epsilon)**(self.alpha - target_heatmaps))

        # Tính loss dựa trên delta
        loss = torch.where(delta < self.theta,
                           self.omega * torch.log(1 + torch.abs(delta / self.epsilon)**(self.alpha - target_heatmaps)),
                           A * delta - C)

        if self.reduction == 'mean':
            # Trung bình loss trên tất cả các pixel, landmark và batch
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class TotalLoss(nn.Module):
    """
    Hàm mất mát tổng hợp cho giám sát trung gian.
    Kết hợp Smooth L1 cho tọa độ (từ soft-argmax) và Awing Loss cho heatmap.
    """
    def __init__(self, num_blocks, beta=1.0, weight_intermediate=1.2, aw_alpha=2.1, aw_omega=14):
        super().__init__()
        self.num_blocks = num_blocks
        self.beta = beta # Trọng số cân bằng giữa loss tọa độ và heatmap (Eq 5)
        self.w = weight_intermediate # Trọng số w cho giám sát trung gian (Eq 6)

        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean', beta=1.0) # beta của SmoothL1
        self.aw_loss = AwingLoss(alpha=aw_alpha, omega=aw_omega, reduction='mean')

    def forward(self, intermediate_heatmaps, gt_coords, gt_heatmaps):
        """
        Args:
            intermediate_heatmaps (list): List các tensor heatmap từ các khối (mỗi tensor: B, N_lmk, H, W)
            gt_coords (tensor): Tọa độ landmark ground truth (B, N_lmk, 2)
            gt_heatmaps (tensor): Heatmap ground truth (B, N_lmk, H, W)
        """
        total_loss = 0.0
        num_stages = len(intermediate_heatmaps)
        assert num_stages == self.num_blocks

        for j in range(num_stages): # j từ 0 đến B-1
            pred_heatmaps_j = intermediate_heatmaps[j] # (B, N_lmk, H, W)

            # 1. Tính loss tọa độ (d1 trong paper)
            pred_coords_j = soft_argmax(pred_heatmaps_j) # (B, N_lmk, 2)
            coord_loss = self.smooth_l1_loss(pred_coords_j, gt_coords)

            # 2. Tính loss heatmap (d2 trong paper)
            heatmap_loss = self.aw_loss(pred_heatmaps_j, gt_heatmaps)

            # 3. Tính loss cho stage j (Lj trong paper)
            stage_loss = coord_loss + self.beta * heatmap_loss

            # 4. Áp dụng trọng số giám sát trung gian (Eq 6: w^(j-B))
            # stage_idx = j + 1 (vì j bắt đầu từ 0, B là num_blocks)
            weight = self.w ** (j - self.num_blocks + 1) # j=0 -> w^(1-B), j=B-1 -> w^0=1
            total_loss += weight * stage_loss

        return total_loss