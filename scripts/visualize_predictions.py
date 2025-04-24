import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Project specific imports
from config import config # Import default config
from models.cascaded_dvit import CascadedDViT
from data.dataset_300w import FaceLandmark300WDataset # Assuming 300W vis
from utils.visualization import visualize_predictions_heatmaps # Use the refactored function

def main():
    parser = argparse.ArgumentParser(description='Visualize DViT model predictions.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint (.pth file).')
    parser.add_argument('--dataset_root', type=str, default=config.ROOT_DATA_DIR, help='Path to the root dataset directory.')
    parser.add_argument('--annotation_file', type=str, default=config.TEST_LIST_FILE, help='Path to the annotation file list (e.g., test list).')
    parser.add_argument('--dataset_name', type=str, default='300w', choices=['300w', 'wflw'], help='Name of the dataset used for visualization.')
    parser.add_argument('--num_samples', type=int, default=config.NUM_VISUALIZE_SAMPLES, help='Number of samples to visualize.')
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(config.SAVE_DIR), help='Directory to save visualization results.')

    args = parser.parse_args()

    # --- Device Setup ---
    device = config.PRIMARY_DEVICE
    print(f"Using device: {device}")

    # Determine NUM_LANDMARKS based on dataset name
    if args.dataset_name.lower() == '300w':
        num_landmarks = 68
        # Example landmark indices for 300W heatmap visualization
        landmark_indices_to_show = [8, 30, 36, 45, 48, 54]
    elif args.dataset_name.lower() == 'wflw':
        num_landmarks = 98
        # Example landmark indices for WFLW heatmap visualization
        landmark_indices_to_show = [33, 51, 60, 72, 76, 82]
    else:
        print(f"Error: Unsupported dataset name '{args.dataset_name}'.")
        return

    # --- Load Model ---
    print(f"Loading model architecture (Landmarks: {num_landmarks})...")
    vis_model = CascadedDViT(
        img_size=config.IMG_SIZE,
        num_blocks=config.NUM_BLOCKS,
        num_landmarks=num_landmarks,
        backbone_out_chans_target=config.BACKBONE_CHANNELS,
        dvit_chans=config.DVIT_INTERNAL_CHANNELS,
        embed_dim=config.VIT_EMBED_DIM,
        depth=config.VIT_DEPTH,
        num_heads=config.VIT_HEADS,
        feature_size=(config.FEATURE_H, config.FEATURE_W)
    )

    print(f"Loading model weights from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
    try:
        # Load state dict, handling potential DataParallel wrapping
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        is_parallel = any(key.startswith('module.') for key in state_dict.keys())
        if is_parallel:
            for k, v in state_dict.items(): new_state_dict[k[7:]] = v
        else: new_state_dict = state_dict
        vis_model.load_state_dict(new_state_dict)
        vis_model.to(device)
        vis_model.eval() # Set to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Load Dataset ---
    print(f"Loading dataset for visualization: {args.dataset_name}")
    if args.dataset_name.lower() == '300w':
         DatasetClass = FaceLandmark300WDataset
    # Add elif for WFLW etc.
    else:
         print(f"Error: Dataset class for '{args.dataset_name}' not implemented.")
         return

    # Minimal transform for visualization (usually just normalize + tensor)
    vis_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', label_fields=[], remove_invisible=False))

    try:
        vis_dataset = DatasetClass(
            root_dir=args.dataset_root,
            file_list_path=args.annotation_file,
            img_size=config.IMG_SIZE,
            feature_map_size=(config.FEATURE_H, config.FEATURE_W),
            sigma=config.SIGMA,
            transform=vis_transform,
            num_landmarks=num_landmarks
        )
    except (FileNotFoundError, ValueError) as e:
         print(f"Error initializing visualization dataset: {e}")
         return

    # --- Perform Visualization ---
    # Prepare config dict needed by the visualization function
    vis_config = {
        'IMG_SIZE': config.IMG_SIZE,
        'FEATURE_W': config.FEATURE_W,
        'FEATURE_H': config.FEATURE_H,
        'NUM_LANDMARKS': num_landmarks,
        'SAVE_DIR': args.save_dir # Pass save dir for plot
    }

    visualize_predictions_heatmaps(
        model=vis_model,
        vis_dataset=vis_dataset,
        config=vis_config,
        num_samples_to_show=args.num_samples,
        device=device,
        landmark_indices_to_show=landmark_indices_to_show
    )

    print("\nVisualization script finished.")

if __name__ == '__main__':
    main()