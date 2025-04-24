import numpy as np
import torch
from sklearn.metrics import auc
from tqdm import tqdm
import matplotlib.pyplot as plt
from .heatmap_utils import soft_argmax # Assuming utils package accessible
import os

# --- Constants for Interocular Distance ---
# Should match the landmark indices used in your specific dataset annotation standard
INDICES_INTEROCULAR_300W = (36, 45) # Left eye corner, Right eye corner (0-based index for 68 landmarks)
# Add others if needed:
# INDICES_INTEROCULAR_WFLW = (60, 72) # WFLW specific indices if using 98 landmarks

# calculate_nme, calculate_fr_auc, evaluate_model, plot_ced_curve
# functions remain the same as in the notebook
# Ensure soft_argmax is imported correctly.

def calculate_nme(predictions, ground_truth, normalization_distances, sample_basenames=None):
    """
    Calculates Normalized Mean Error (NME).

    Args:
        predictions (np.ndarray): Predicted landmarks (N, num_landmarks, 2).
        ground_truth (np.ndarray): Ground truth landmarks (N, num_landmarks, 2).
        normalization_distances (np.ndarray): Normalization distance for each sample (N,).
                                            Usually interocular distance.
        sample_basenames (list, optional): List of sample identifiers for warning messages.

    Returns:
        tuple: (mean_nme, sample_nmes)
            - mean_nme (float): Average NME over valid samples.
            - sample_nmes (np.ndarray): NME for each sample (N,). Contains NaN for invalid samples.
    """
    num_samples = predictions.shape[0]
    if num_samples == 0:
        print("Warning: calculate_nme called with zero samples.")
        return 0.0, np.array([])
    if predictions.shape != ground_truth.shape:
        raise ValueError(f"Prediction shape {predictions.shape} mismatch ground truth shape {ground_truth.shape}")
    if normalization_distances.shape[0] != num_samples:
         raise ValueError(f"Normalization distances shape {normalization_distances.shape} mismatch number of samples {num_samples}")

    # Calculate point-to-point Euclidean errors for all landmarks in all samples
    errors = np.linalg.norm(predictions - ground_truth, axis=2) # Shape: (N, num_landmarks)

    # Calculate mean error per sample (average over landmarks)
    mean_error_per_sample = np.mean(errors, axis=1) # Shape: (N,)

    # Normalize mean error by the normalization distance
    # Handle potential zero or very small normalization distances to avoid division by zero / inf
    valid_norm_mask = normalization_distances > 1e-6
    sample_nmes = np.full(num_samples, np.nan, dtype=np.float64) # Initialize with NaN

    if np.any(valid_norm_mask):
        sample_nmes[valid_norm_mask] = mean_error_per_sample[valid_norm_mask] / normalization_distances[valid_norm_mask]

    # Report samples with invalid normalization distance
    invalid_indices = np.where(~valid_norm_mask)[0]
    if len(invalid_indices) > 0:
        # print(f"Warning: {len(invalid_indices)} samples had near-zero normalization distance. Their NME is NaN.")
        # if sample_basenames:
        #     for idx in invalid_indices[:5]: # Print first few problematic samples
        #         print(f"  - Sample: {sample_basenames[idx]}, NormDist: {normalization_distances[idx]:.4f}")
        pass # Reduce verbosity

    # Calculate overall mean NME, ignoring NaN values
    valid_nmes = sample_nmes[~np.isnan(sample_nmes)]
    mean_nme = np.mean(valid_nmes) if len(valid_nmes) > 0 else 0.0

    return mean_nme, sample_nmes

# --- Calculate Failure Rate (FR) and Area Under Curve (AUC) for CED ---
def calculate_fr_auc(sample_nmes, failure_threshold=0.10, num_bins=1000):
    """
    Calculates Failure Rate (FR) and Area Under the Curve (AUC) for the
    Cumulative Error Distribution (CED).

    Args:
        sample_nmes (np.ndarray): NME for each sample (N,). Can contain NaNs.
        failure_threshold (float): NME threshold to define a failure.
        num_bins (int): Number of bins for discretizing the error thresholds for AUC calculation.

    Returns:
        tuple: (failure_rate, auc_value, ced_thresholds, ced_values)
            - failure_rate (float): Proportion of valid samples with NME > failure_threshold.
            - auc_value (float): Area under the CED curve up to the failure_threshold.
            - ced_thresholds (np.ndarray): Error thresholds used for CED.
            - ced_values (np.ndarray): Proportion of samples with NME <= each threshold.
    """
    # Filter out NaN values first
    valid_sample_nmes = sample_nmes[~np.isnan(sample_nmes)]
    num_valid_samples = len(valid_sample_nmes)

    if num_valid_samples == 0:
        print("Warning: calculate_fr_auc called with no valid NME samples.")
        return 0.0, 0.0, np.array([0.0]), np.array([0.0]) # Return minimal arrays

    # 1. Calculate Failure Rate (FR)
    num_failures = np.sum(valid_sample_nmes > failure_threshold)
    failure_rate = num_failures / num_valid_samples if num_valid_samples > 0 else 0.0

    # 2. Calculate Cumulative Error Distribution (CED) values
    # Create error thresholds from 0 up to failure_threshold
    ced_thresholds = np.linspace(0, failure_threshold, num_bins + 1)
    # Calculate the proportion of samples whose NME is less than or equal to each threshold
    ced_values = [np.sum(valid_sample_nmes <= th) / num_valid_samples for th in ced_thresholds]
    ced_values = np.array(ced_values)

    # 3. Calculate Area Under the Curve (AUC) using the trapezoidal rule (via sklearn.metrics.auc)
    # AUC is calculated up to the failure_threshold and normalized by the threshold
    # to give a value between 0 and 1.
    auc_value = auc(ced_thresholds, ced_values) / failure_threshold if failure_threshold > 0 else 0.0

    return failure_rate, auc_value, ced_thresholds, ced_values


# --- Main Evaluation Function ---
def evaluate_model(model, dataloader, device, dataset_name, num_landmarks, img_size, feature_map_size, failure_threshold=0.10):
    """
    Runs quantitative evaluation on a given model and dataloader.

    Args:
        model (nn.Module): The trained model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): CPU or CUDA device.
        dataset_name (str): Name of the dataset (e.g., '300w', 'wflw') for choosing norm indices.
        num_landmarks (int): Number of landmarks the model predicts.
        img_size (int): Input image size the model expects.
        feature_map_size (tuple): Spatial size (H, W) of the model's output heatmaps.
        failure_threshold (float): Threshold for calculating FR and AUC.

    Returns:
        dict: A dictionary containing evaluation results (NME, FR, AUC, CED data).
              Returns None if evaluation fails.
    """
    model.eval() # Set model to evaluation mode
    all_predictions_img_scale = []
    all_ground_truths_img_scale = []
    all_norm_distances = []
    all_basenames = []

    # --- Determine Normalization Indices ---
    if dataset_name.lower() == '300w':
        if num_landmarks != 68: print(f"Warning: Evaluating 300W but num_landmarks is {num_landmarks} (expected 68). Check norm indices.");
        norm_indices = INDICES_INTEROCULAR_300W
    # Add elif for other datasets like WFLW if needed
    # elif dataset_name.lower() == 'wflw':
    #     if num_landmarks != 98: print(f"Warning: Evaluating WFLW but num_landmarks is {num_landmarks} (expected 98). Check norm indices.");
    #     norm_indices = INDICES_INTEROCULAR_WFLW
    else:
        print(f"Error: Unknown dataset name '{dataset_name}' for selecting normalization indices.")
        # Fallback or default? Let's default to 300W if landmarks match, otherwise error.
        if num_landmarks == 68:
            print("Warning: Unknown dataset, defaulting to 300W normalization indices as landmarks=68.")
            norm_indices = INDICES_INTEROCULAR_300W
        else:
             print("Cannot determine normalization indices.")
             return None # Cannot proceed without normalization method

    # Check if indices are valid for the number of landmarks
    if not (0 <= norm_indices[0] < num_landmarks and 0 <= norm_indices[1] < num_landmarks):
        print(f"Error: Normalization indices {norm_indices} are out of bounds for {num_landmarks} landmarks.")
        return None
    print(f"Using normalization landmarks (0-based): {norm_indices} for {dataset_name} evaluation.")

    # --- Collect Predictions and Ground Truth ---
    eval_pbar = tqdm(dataloader, desc=f"Evaluating on {dataset_name}", leave=False)
    with torch.no_grad():
        for batch in eval_pbar:
            # Expecting dataset to yield: image, _, _, gt_landmarks_img_scale, basename
            if len(batch) != 5:
                print(f"Error: Dataloader yielded batch with {len(batch)} elements, expected 5. Skipping batch.")
                continue
            images, _, _, gt_landmarks_img_scale, basenames_batch = batch

            # Ensure batch items are lists/tensors as expected
            if not isinstance(images, torch.Tensor) or not isinstance(gt_landmarks_img_scale, torch.Tensor):
                 print(f"Error: Invalid data types in batch. Img: {type(images)}, GT: {type(gt_landmarks_img_scale)}. Skipping.")
                 continue

            images = images.to(device)
            gt_landmarks_img_scale = gt_landmarks_img_scale.cpu().numpy() # GT to numpy for distance calc

            # --- Get Model Predictions ---
            intermediate_heatmaps = model(images)
            final_heatmap = intermediate_heatmaps[-1] # Use the final stage heatmap (B, N_lmk, H_feat, W_feat)

            # --- Convert Heatmaps to Coordinates (Image Scale) ---
            pred_coords_feat_scale = soft_argmax(final_heatmap) # (B, N_lmk, 2) - feature map scale
            # Scale coordinates back to image size
            scale_back_x = img_size / feature_map_size[1]
            scale_back_y = img_size / feature_map_size[0]
            # Use float32 for scale tensor
            scale_tensor = torch.tensor([scale_back_x, scale_back_y], device=device, dtype=torch.float32).view(1, 1, 2)
            pred_coords_img_scale = pred_coords_feat_scale * scale_tensor # (B, N_lmk, 2) - image scale
            preds_np = pred_coords_img_scale.cpu().numpy() # Predictions to numpy

            # --- Calculate Normalization Distance for each sample ---
            batch_norm_distances = []
            valid_indices_in_batch = [] # Track samples with valid norm distance
            for i in range(gt_landmarks_img_scale.shape[0]):
                gt_sample = gt_landmarks_img_scale[i] # Landmarks for sample i
                p1 = gt_sample[norm_indices[0]] # Coords of first norm landmark
                p2 = gt_sample[norm_indices[1]] # Coords of second norm landmark
                dist = np.linalg.norm(p1 - p2)
                # Store distance if valid, mark index
                if dist >= 1e-6:
                    batch_norm_distances.append(dist)
                    valid_indices_in_batch.append(i)
                # else: # Optional: Warn about invalid distance
                #     basename = basenames_batch[i] if isinstance(basenames_batch, (list, tuple)) and i < len(basenames_batch) else f"Sample {i}"
                #     print(f"Warning: Normalization distance near zero ({dist:.4f}) for {basename}. Skipping sample in NME calc.")


            # --- Store Valid Data ---
            if valid_indices_in_batch:
                 # Select only the predictions, ground truths, and basenames corresponding to valid norm distances
                 valid_preds = preds_np[valid_indices_in_batch]
                 valid_gts = gt_landmarks_img_scale[valid_indices_in_batch]
                 all_predictions_img_scale.append(valid_preds)
                 all_ground_truths_img_scale.append(valid_gts)
                 all_norm_distances.extend(batch_norm_distances) # Already filtered

                 # Handle basenames (might be tuple or list)
                 if isinstance(basenames_batch, (list, tuple)):
                      valid_basenames = [basenames_batch[i] for i in valid_indices_in_batch]
                      all_basenames.extend(valid_basenames)
                 # else: # If basenames aren't provided correctly, generate placeholders
                 #     all_basenames.extend([f"batch{eval_pbar.n}_sample{i}" for i in valid_indices_in_batch])


    # --- Consolidate Collected Data ---
    if not all_predictions_img_scale:
        print("Error: No valid predictions collected during evaluation.")
        return None

    try:
        all_predictions = np.concatenate(all_predictions_img_scale, axis=0)
        all_ground_truths = np.concatenate(all_ground_truths_img_scale, axis=0)
        all_norm_distances = np.array(all_norm_distances)
        print(f"\nEvaluation data collected: Preds={all_predictions.shape}, GTs={all_ground_truths.shape}, NormDists={all_norm_distances.shape}, Basenames={len(all_basenames)}")
        if not (all_predictions.shape[0] == all_ground_truths.shape[0] == all_norm_distances.shape[0] == len(all_basenames)):
             print("ERROR: Mismatch in collected data lengths!")
             # Print lengths for debugging
             print(f"Preds: {all_predictions.shape[0]}, GTs: {all_ground_truths.shape[0]}, NormD: {all_norm_distances.shape[0]}, Names: {len(all_basenames)}")
             # Attempt recovery or return error
             min_len = min(all_predictions.shape[0], all_ground_truths.shape[0], all_norm_distances.shape[0], len(all_basenames))
             if min_len > 0 :
                 print(f"Attempting to use the minimum length: {min_len}")
                 all_predictions = all_predictions[:min_len]
                 all_ground_truths = all_ground_truths[:min_len]
                 all_norm_distances = all_norm_distances[:min_len]
                 all_basenames = all_basenames[:min_len]
             else:
                 return None

    except ValueError as e:
         print(f"Error concatenating evaluation results: {e}")
         return None


    # --- Calculate Metrics ---
    results = {}
    print("\nCalculating NME...")
    # Calculate NME overall and per sample (sample_nmes_full contains NaNs for invalid norms)
    nme_overall, sample_nmes_full = calculate_nme(all_predictions, all_ground_truths, all_norm_distances, all_basenames)
    print(f"[{dataset_name}] Overall NME calculated: {nme_overall:.6f} ({np.sum(~np.isnan(sample_nmes_full))} valid samples)")

    print(f"Calculating FR@{failure_threshold:.2f} and AUC@{failure_threshold:.2f}...")
    # Calculate FR and AUC based on the specified failure threshold
    fr_value, auc_value, ced_x, ced_y = calculate_fr_auc(sample_nmes_full, failure_threshold=failure_threshold)

    results['NME_Overall'] = nme_overall # Overall NME using specified norm
    results[f'FR@{failure_threshold:.2f}'] = fr_value
    results[f'AUC@{failure_threshold:.2f}'] = auc_value
    results['CED_x'] = ced_x
    results['CED_y'] = ced_y
    results['FailureThreshold'] = failure_threshold # Store threshold used

    print(f"[{dataset_name}] FR@{failure_threshold:.2f}: {fr_value:.6f}")
    print(f"[{dataset_name}] AUC@{failure_threshold:.2f}: {auc_value:.6f}")


    # --- Special Handling for 300W Subsets (Common, Challenging, Full) ---
    if dataset_name.lower() == '300w':
        print("\nCalculating NME for 300W subsets...")
        common_indices = [i for i, bn in enumerate(all_basenames) if isinstance(bn, str) and (bn.startswith('lfpw/testset/') or bn.startswith('helen/testset/'))]
        challenging_indices = [i for i, bn in enumerate(all_basenames) if isinstance(bn, str) and bn.startswith('ibug/')]
        full_indices = list(range(len(all_basenames))) # All valid indices collected

        print(f"300W Split Indices: Full={len(full_indices)}, Common={len(common_indices)}, Challenging={len(challenging_indices)}")

        # Use the NME calculated earlier for the 'Full' set
        results['NME_Full'] = nme_overall # Already calculated

        # Calculate NME for Common subset
        if common_indices:
            nme_common, _ = calculate_nme(all_predictions[common_indices], all_ground_truths[common_indices], all_norm_distances[common_indices])
            results['NME_Common'] = nme_common
            print(f"[300W] NME Common ({len(common_indices)} samples): {nme_common:.6f}")
        else: results['NME_Common'] = float('nan'); print("[300W] No Common samples found.")

        # Calculate NME for Challenging subset
        if challenging_indices:
            nme_challenging, _ = calculate_nme(all_predictions[challenging_indices], all_ground_truths[challenging_indices], all_norm_distances[challenging_indices])
            results['NME_Challenging'] = nme_challenging
            print(f"[300W] NME Challenging ({len(challenging_indices)} samples): {nme_challenging:.6f}")
        else: results['NME_Challenging'] = float('nan'); print("[300W] No Challenging samples found.")

        # Remove the generic 'NME_Overall' key for 300W as we have NME_Full
        if 'NME_Overall' in results: del results['NME_Overall']
        # FR and AUC are typically reported on the Full set for 300W.

    return results

# --- Function to Plot CED Curve ---
def plot_ced_curve(ced_x, ced_y, dataset_name, failure_threshold=0.10, save_path=None):
    """Plots the Cumulative Error Distribution (CED) curve."""
    if len(ced_x) < 2 or len(ced_y) < 2: # Need at least two points to plot
        print(f"Warning: Cannot plot CED for {dataset_name} - insufficient data points ({len(ced_x)}).")
        return

    plt.figure(figsize=(8, 6))
    plt.plot(ced_x, ced_y, lw=2, marker='.', markersize=3) # Add markers for visibility
    plt.title(f'CED Curve for {dataset_name} (up to NME={failure_threshold:.2f})', fontsize=14)
    plt.xlabel('Normalized Mean Error (NME)', fontsize=12)
    plt.ylabel('Proportion of Test Images', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Set reasonable limits, ensuring the failure threshold is visible
    plt.xlim(0, failure_threshold * 1.05) # Extend x-axis slightly beyond threshold
    plt.ylim(0, 1.05) # Extend y-axis slightly beyond 1

    # Add vertical line at failure threshold
    plt.axvline(x=failure_threshold, color='r', linestyle='--', label=f'Failure Threshold = {failure_threshold:.2f}')

    # Add AUC value to the plot text
    auc_value = auc(ced_x, ced_y) / failure_threshold if failure_threshold > 0 else 0.0
    plt.text(failure_threshold * 0.1, 0.1, f'AUC@{failure_threshold:.2f} = {auc_value:.4f}', fontsize=12, color='darkblue')


    plt.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"CED curve saved to {save_path}")
        except Exception as e:
            print(f"Error saving CED curve to {save_path}: {e}")

    plt.show() # Display the plot