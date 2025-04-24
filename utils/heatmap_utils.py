import torch
import torch.nn.functional as F
import numpy as np # Needed for numpy usage if any

# generate_gaussian_heatmap, generate_ground_truth_heatmaps, soft_argmax
# functions remain the same as in the notebook
def generate_gaussian_heatmap(center_x, center_y, heatmap_size, sigma=1.0):
    """Generates a 2D Gaussian heatmap."""
    H, W = heatmap_size
    # Create coordinate grids
    x = torch.arange(0, W, 1, dtype=torch.float32, device=center_x.device if isinstance(center_x, torch.Tensor) else None)
    y = torch.arange(0, H, 1, dtype=torch.float32, device=center_y.device if isinstance(center_y, torch.Tensor) else None)
    # y = y.unsqueeze(1) # Shape (H, 1) - Old way
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij') # More robust way, HxW grid

    # Calculate Gaussian, ensure consistent device
    center_x = torch.as_tensor(center_x, device=x_grid.device)
    center_y = torch.as_tensor(center_y, device=y_grid.device)

    dist_sq = (x_grid - center_x)**2 + (y_grid - center_y)**2
    exponent = dist_sq / (2 * sigma**2)
    # Clamp exponent to prevent exp from generating inf/NaN
    heatmap = torch.exp(-torch.clamp(exponent, max=50)) # Clamp large positive exponents

    return heatmap

def generate_ground_truth_heatmaps(landmarks, heatmap_size, sigma=1.0):
    """Generates ground truth heatmaps for all landmarks."""
    # landmarks: Tensor (N_lmk, 2) with (x, y) coordinates scaled to heatmap size
    # heatmap_size: tuple (H, W)
    num_landmarks = landmarks.shape[0]
    H, W = heatmap_size
    heatmaps = torch.zeros((num_landmarks, H, W), dtype=torch.float32, device=landmarks.device)

    for i in range(num_landmarks):
        center_x, center_y = landmarks[i]
        # Check if landmark is within bounds (important!)
        # Use >= 0 and < Size
        if 0 <= center_x < W and 0 <= center_y < H:
            heatmaps[i] = generate_gaussian_heatmap(center_x, center_y, heatmap_size, sigma)
        # else: # Optional: print warning if landmark is out of bounds
            # print(f"Warning: Landmark {i} ({center_x:.1f}, {center_y:.1f}) is outside heatmap bounds ({W}x{H}). Heatmap will be zero.")

    return heatmaps


def soft_argmax(heatmaps):
    """
    Calculates soft-argmax coordinates from heatmaps.
    Input: heatmaps (B, N_lmk, H, W)
    Output: coords (B, N_lmk, 2) - coordinates (x, y) relative to heatmap grid
    """
    B, N_lmk, H, W = heatmaps.shape
    device = heatmaps.device
    dtype = heatmaps.dtype

    # Normalize heatmaps to probability distributions using softmax
    # Apply softmax over the spatial dimensions (H, W) for each landmark independently
    heatmaps_flat = heatmaps.view(B, N_lmk, -1) # Flatten spatial dims: (B, N_lmk, H*W)
    # Ensure numerical stability for softmax
    heatmaps_flat = heatmaps_flat - torch.max(heatmaps_flat, dim=-1, keepdim=True)[0]
    softmax_heatmaps = F.softmax(heatmaps_flat, dim=2) # Apply softmax along the H*W dimension
    softmax_heatmaps = softmax_heatmaps.view(B, N_lmk, H, W) # Reshape back: (B, N_lmk, H, W)

    # Create coordinate grids matching heatmap dimensions
    # y coordinates (rows): 0 to H-1
    coord_y = torch.arange(H, device=device, dtype=dtype).float()
    # x coordinates (cols): 0 to W-1
    coord_x = torch.arange(W, device=device, dtype=dtype).float()

    # Calculate expected coordinates
    # Sum over H and W dimensions
    # Expected y = sum_{h,w} ( softmax[h,w] * h )
    # Expected x = sum_{h,w} ( softmax[h,w] * w )
    # Need to broadcast coord_y/coord_x correctly
    expected_y = torch.sum(softmax_heatmaps * coord_y.view(1, 1, H, 1), dim=(2, 3)) # Sum over H, W
    expected_x = torch.sum(softmax_heatmaps * coord_x.view(1, 1, 1, W), dim=(2, 3)) # Sum over H, W

    # Stack coordinates into (x, y) format
    coords = torch.stack((expected_x, expected_y), dim=2) # Shape: (B, N_lmk, 2)

    return coords