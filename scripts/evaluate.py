import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Project specific imports
from config import config # Import default config
from models.cascaded_dvit import CascadedDViT
from data.dataset_300w import FaceLandmark300WDataset # Assuming 300W evaluation
from utils.evaluation_metrics import evaluate_model, plot_ced_curve

def main():
    parser = argparse.ArgumentParser(description='Evaluate DViT model on a dataset.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint (.pth file).')
    parser.add_argument('--dataset_root', type=str, default=config.ROOT_DATA_DIR, help='Path to the root dataset directory.')
    parser.add_argument('--annotation_file', type=str, default=config.TEST_LIST_FILE, help='Path to the annotation file list (e.g., test list).')
    parser.add_argument('--dataset_name', type=str, default='300w', choices=['300w', 'wflw'], help='Name of the dataset being evaluated.')
    parser.add_argument('--batch_size', type=int, default=config.EVAL_BATCH_SIZE, help='Batch size for evaluation.')
    parser.add_argument('--num_workers', type=int, default=config.NUM_WORKERS, help='Number of workers for DataLoader.')
    parser.add_argument('--failure_threshold', type=float, default=config.EVAL_FAILURE_THRESHOLD, help='NME threshold for FR/AUC calculation.')
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(config.SAVE_DIR), help='Directory to save evaluation results (plots).') # Save in parent of train save dir

    args = parser.parse_args()

    # --- Device Setup ---
    device = config.PRIMARY_DEVICE # Use device from config
    print(f"Using device: {device}")

    # Determine NUM_LANDMARKS based on dataset name
    if args.dataset_name.lower() == '300w':
        num_landmarks = 68
    elif args.dataset_name.lower() == 'wflw':
        num_landmarks = 98
    else:
        print(f"Error: Unsupported dataset name '{args.dataset_name}' for landmark count.")
        return

    # --- Load Model ---
    print(f"Loading model architecture (Landmarks: {num_landmarks})...")
    eval_model = CascadedDViT(
        img_size=config.IMG_SIZE,
        num_blocks=config.NUM_BLOCKS,
        num_landmarks=num_landmarks, # Use determined landmark count
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
        # Create new state_dict without 'module.' prefix if necessary
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        is_parallel = any(key.startswith('module.') for key in state_dict.keys())
        if is_parallel:
            print("Checkpoint appears to be from DataParallel, removing 'module.' prefix.")
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
        else:
            new_state_dict = state_dict

        eval_model.load_state_dict(new_state_dict)
        eval_model.to(device)
        eval_model.eval() # Set to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Prepare Evaluation Dataset and Dataloader ---
    print(f"Loading evaluation dataset: {args.dataset_name}")
    # Use the appropriate dataset class based on args.dataset_name if needed
    if args.dataset_name.lower() == '300w':
         DatasetClass = FaceLandmark300WDataset
    # Add elif for WFLW etc. if implementing those datasets
    # elif args.dataset_name.lower() == 'wflw':
    #     DatasetClass = FaceLandmarkWFLWDataset # Assumes you have this class
    else:
         print(f"Error: Dataset class for '{args.dataset_name}' not implemented.")
         return

    # Use minimal transform for evaluation
    eval_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', label_fields=[], remove_invisible=False))

    try:
        evaluation_dataset = DatasetClass(
            root_dir=args.dataset_root,
            file_list_path=args.annotation_file,
            img_size=config.IMG_SIZE,
            feature_map_size=(config.FEATURE_H, config.FEATURE_W),
            sigma=config.SIGMA, # Sigma doesn't affect eval, but needed by class init
            transform=eval_transform,
            num_landmarks=num_landmarks
        )
    except (FileNotFoundError, ValueError) as e:
         print(f"Error initializing evaluation dataset: {e}")
         print("Please ensure dataset paths and file lists are correct.")
         return

    evaluation_dataloader = DataLoader(
        evaluation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=config.PIN_MEMORY
    )

    print(f"Evaluation samples: {len(evaluation_dataset)}, Batches: {len(evaluation_dataloader)}")

    # --- Run Evaluation ---
    print(f"\nRunning evaluation for {args.dataset_name.upper()}...")
    evaluation_results = evaluate_model(
        model=eval_model,
        dataloader=evaluation_dataloader,
        device=device,
        dataset_name=args.dataset_name,
        num_landmarks=num_landmarks,
        img_size=config.IMG_SIZE,
        feature_map_size=(config.FEATURE_H, config.FEATURE_W),
        failure_threshold=args.failure_threshold
    )

    # --- Display Results ---
    if evaluation_results:
        print("\n--- Evaluation Summary ---")
        ft_key = f'@{args.failure_threshold:.2f}' # Key suffix for FR/AUC

        # Print metrics based on dataset type
        if args.dataset_name.lower() == '300w':
            print(f"NME Full:       {evaluation_results.get('NME_Full', float('nan')):.4f}")
            print(f"NME Common:     {evaluation_results.get('NME_Common', float('nan')):.4f}")
            print(f"NME Challenging:{evaluation_results.get('NME_Challenging', float('nan')):.4f}")
        else: # For other datasets like WFLW
            print(f"NME Overall:    {evaluation_results.get('NME_Overall', float('nan')):.4f}")

        # Print FR and AUC (keys include the threshold)
        print(f"FR{ft_key}:   {evaluation_results.get(f'FR{ft_key}', float('nan')):.4f}")
        print(f"AUC{ft_key}:   {evaluation_results.get(f'AUC{ft_key}', float('nan')):.4f}")


        # --- Plot CED Curve ---
        if 'CED_x' in evaluation_results and 'CED_y' in evaluation_results:
            print("\nPlotting CED curve...")
            # Create results directory if it doesn't exist
            os.makedirs(args.save_dir, exist_ok=True)
            ced_save_path = os.path.join(args.save_dir, f"ced_curve_{args.dataset_name}.png")
            plot_ced_curve(
                evaluation_results['CED_x'],
                evaluation_results['CED_y'],
                dataset_name=args.dataset_name.upper(),
                failure_threshold=args.failure_threshold,
                save_path=ced_save_path
            )
    else:
        print("Evaluation failed or returned no results.")

    print("\nEvaluation script finished.")

if __name__ == '__main__':
    main()