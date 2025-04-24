import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from .heatmap_utils import soft_argmax # Assuming utils package accessible
# Need to import dataset class if used directly here, or pass data samples
# from data.dataset_300w import FaceLandmark300WDataset
import albumentations as A # For potential inverse transforms if needed

# Function to draw landmarks (can be reused)
def draw_landmarks(image, landmarks, color=(0, 255, 0), radius=2):
    """Draws landmarks on a copy of the image."""
    img_copy = image.copy()
    # Ensure image is BGR for cv2 drawing if it came from matplotlib/RGB
    if len(img_copy.shape) == 3 and img_copy.shape[2] == 3:
       # Heuristic: If max value > 1, assume 0-255 range. If not, maybe 0-1.
       # This is tricky. Assume input 'image' is uint8 RGB [0, 255] for consistency.
       pass # Assume input is uint8 RGB for drawing visualization

    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.detach().cpu().numpy()

    landmarks_int = landmarks.astype(np.int32)

    for (x, y) in landmarks_int:
        # Check bounds before drawing
        h, w = img_copy.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img_copy, (x, y), radius, color, -1) # Use BGR color for OpenCV
    return img_copy

# Visualization function adapted from notebook
# Takes model, dataset, config parameters needed
def visualize_ Hpredictions_heatmaps(
    model,
    vis_dataset, # Instance of the dataset (e.g., FaceLandmark300WDataset)
    config, # Dictionary or config object with IMG_SIZE, FEATURE_W, FEATURE_H, NUM_LANDMARKS etc.
    num_samples_to_show=5,
    device='cpu',
    landmark_indices_to_show=None # Optional list of landmark indices for heatmaps
    ):
    """Visualizes predictions and selected heatmaps for random samples."""

    print("Starting visualization...")
    if landmark_indices_to_show is None:
         # Default indices if not provided (example for 68 landmarks)
         landmark_indices_to_show = [8, 30, 36, 45, 48, 54]
         if config['NUM_LANDMARKS'] == 98: # Example for WFLW
             landmark_indices_to_show = [33, 51, 60, 72, 76, 82] # Indices for WFLW nose, eyes, mouth corners
         print(f"Using default landmark indices for heatmaps: {landmark_indices_to_show}")


    num_heatmaps_to_show = len(landmark_indices_to_show)
    num_cols = 1 + num_heatmaps_to_show # 1 for image + N for heatmaps

    if len(vis_dataset) == 0:
        print("Error: Dataset is empty. Cannot visualize.")
        return
    if len(vis_dataset) < num_samples_to_show:
        print(f"Warning: Requested {num_samples_to_show} samples, but dataset only has {len(vis_dataset)}. Showing all.")
        num_samples_to_show = len(vis_dataset)
    if num_samples_to_show <= 0:
        print("No samples requested for visualization.")
        return

    indices_to_show = random.sample(range(len(vis_dataset)), num_samples_to_show)

    fig, axes = plt.subplots(num_samples_to_show, num_cols, figsize=(3 * num_cols, 3.5 * num_samples_to_show))
    if num_samples_to_show == 1: # Handle single row case for axes indexing
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(f"DViT Predictions (GT=Green, Pred=Red) & Heatmaps (Lmk Indices: {landmark_indices_to_show})", fontsize=12)

    model.eval() # Ensure model is in evaluation mode
    with torch.no_grad():
        for i, data_idx in enumerate(indices_to_show):
            img_name_display = f"Idx {data_idx}"
            ax_img = axes[i, 0] # First column for the image+landmarks

            try:
                # Get data: image_tensor, _, _, gt_landmarks_img_scale, basename
                # The dataset __getitem__ should return these 5 elements
                sample = vis_dataset[data_idx]
                if len(sample) != 5:
                     raise ValueError(f"Dataset __getitem__ returned {len(sample)} elements, expected 5.")

                image_tensor, _, _, gt_landmarks_img_scale, basename = sample
                img_name_display = basename if isinstance(basename, str) else f"Idx {data_idx}"

                # --- Prepare Original Image for Display ---
                # Need to load the *original* image and resize it without augmentations/normalization
                # The dataset might not easily provide this, so we may need to reload it.
                # Let's try reloading based on the info available in the dataset object.
                if hasattr(vis_dataset, 'image_files') and data_idx < len(vis_dataset.image_files):
                    img_path = vis_dataset.image_files[data_idx]
                    original_image = cv2.imread(img_path)
                    if original_image is None: raise IOError(f"Failed to reload image {img_path}")
                    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                    # Resize to the standard input size for consistent display
                    image_display = cv2.resize(original_image_rgb, (config['IMG_SIZE'], config['IMG_SIZE']), interpolation=cv2.INTER_LINEAR)
                else:
                    # Fallback: Try to reconstruct from tensor (loses original color/quality)
                    print(f"Warning: Could not reload original image for {img_name_display}. Reconstructing from tensor.")
                    img_display_tensor = image_tensor.cpu().numpy().transpose(1, 2, 0)
                    # Inverse normalization (approximate)
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_display_tensor = std * img_display_tensor + mean
                    img_display_tensor = np.clip(img_display_tensor, 0, 1)
                    image_display = (img_display_tensor * 255).astype(np.uint8)


                # Ground truth landmarks (already at image scale)
                landmarks_display_gt = gt_landmarks_img_scale.cpu().numpy()

                # --- Get Model Prediction ---
                input_tensor = image_tensor.unsqueeze(0).to(device)
                intermediate_heatmaps = model(input_tensor)
                final_heatmap = intermediate_heatmaps[-1] # Shape: (1, Num_lmk, H_feat, W_feat)

                # --- Get Predicted Coordinates ---
                pred_coords_feat_scale = soft_argmax(final_heatmap).squeeze(0) # Shape: (Num_lmk, 2)
                # Scale coordinates back to image size
                scale_back_x = config['IMG_SIZE'] / config['FEATURE_W']
                scale_back_y = config['IMG_SIZE'] / config['FEATURE_H']
                scale_tensor = torch.tensor([scale_back_x, scale_back_y], device=device, dtype=pred_coords_feat_scale.dtype)
                pred_coords_display = pred_coords_feat_scale * scale_tensor
                pred_coords_display_np = pred_coords_display.cpu().numpy()

                # --- Draw Image with Landmarks ---
                # Draw GT (Green), then Pred (Red) on top
                image_with_gt = draw_landmarks(image_display, landmarks_display_gt, color=(0, 255, 0), radius=1) # Green (BGR for cv2)
                image_with_pred = draw_landmarks(image_with_gt, pred_coords_display_np, color=(0, 0, 255), radius=1) # Red (BGR for cv2)

                ax_img.imshow(image_with_pred) # Show the final image with both sets of landmarks
                ax_img.set_title(f"{img_name_display}\nGT(G)/Pred(R)", fontsize=8)
                ax_img.axis('off')

                # --- Draw Heatmaps ---
                heatmaps_to_plot = final_heatmap[0].cpu().numpy() # Shape: (Num_lmk, H_feat, W_feat)

                for j, lmk_idx in enumerate(landmark_indices_to_show):
                    ax_heatmap = axes[i, 1 + j] # Get the correct subplot column
                    if 0 <= lmk_idx < config['NUM_LANDMARKS']:
                         heatmap_data = heatmaps_to_plot[lmk_idx]
                         # Find peak location for visualization aid
                         peak_y, peak_x = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)

                         im = ax_heatmap.imshow(heatmap_data, cmap='viridis', aspect='equal') # 'viridis' is good, 'equal' aspect preserves shape
                         ax_heatmap.plot(peak_x, peak_y, 'rx', markersize=5) # Mark the peak
                         ax_heatmap.set_title(f"Lmk {lmk_idx}", fontsize=8)
                         # Optional colorbar - can make plot crowded
                         # fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
                    else:
                         ax_heatmap.text(0.5, 0.5, f'Invalid Lmk\nIndex {lmk_idx}', ha='center', va='center', fontsize=8)
                    ax_heatmap.axis('off')

            except Exception as e:
                print(f"ERROR processing/plotting index {data_idx} ({img_name_display}): {e}")
                import traceback
                traceback.print_exc() # Print detailed traceback for debugging
                # Display error message on the plot
                ax_img.text(0.5, 0.5, f'Error\n{img_name_display}', ha='center', va='center', color='red', fontsize=8, wrap=True)
                ax_img.axis('off')
                # Turn off remaining heatmap axes for this row
                for j in range(1, num_cols):
                    if i < axes.shape[0] and j < axes.shape[1]: # Check bounds
                         axes[i, j].axis('off')
                         axes[i, j].text(0.5, 0.5, 'Error', ha='center', va='center', fontsize=8)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    # Save the figure automatically
    save_path = os.path.join(config.get('SAVE_DIR', '.'), "prediction_visualization.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Visualization saved to {save_path}")
    plt.show() # Also display it interactively