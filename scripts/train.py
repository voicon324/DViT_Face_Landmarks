import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 # Needed by albumentations for some transforms
import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Project specific imports
from config import config
from models.cascaded_dvit import CascadedDViT
from data.dataset_300w import FaceLandmark300WDataset
from loss.losses import TotalLoss
from utils.visualization import visualize_predictions_heatmaps # Renamed function

def main():
    # Create save directory if it doesn't exist
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    print(f"Results will be saved to: {config.SAVE_DIR}")

    # --- Device Setup ---
    device = config.PRIMARY_DEVICE
    use_dp = config.USE_DATAPARALLEL
    gpu_ids = config.GPU_IDS
    print(f"Using primary device: {device}")
    if use_dp:
        print(f"Using DataParallel on GPUs: {gpu_ids}")

    # --- Model Initialization ---
    print("Initializing model...")
    model = CascadedDViT(
        img_size=config.IMG_SIZE,
        num_blocks=config.NUM_BLOCKS,
        num_landmarks=config.NUM_LANDMARKS,
        backbone_out_chans_target=config.BACKBONE_CHANNELS, # Target output from backbone stage
        dvit_chans=config.DVIT_INTERNAL_CHANNELS,
        embed_dim=config.VIT_EMBED_DIM,
        depth=config.VIT_DEPTH,
        num_heads=config.VIT_HEADS,
        feature_size=(config.FEATURE_H, config.FEATURE_W)
    )
    model.to(device)

    # Wrap model with DataParallel if using multiple GPUs
    if use_dp:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"Model wrapped with DataParallel for devices {gpu_ids}")


    # --- Data Augmentation and Datasets ---
    print("Setting up data loaders...")
    # Training Augmentations
    train_transform = A.Compose([
        A.HorizontalFlip(p=config.AUG_HFLIP_PROB),
        A.ShiftScaleRotate(
            shift_limit=config.AUG_SHIFT_SCALE_LIMIT,
            scale_limit=config.AUG_SHIFT_SCALE_LIMIT,
            rotate_limit=config.AUG_ROTATION_LIMIT,
            p=0.8,
            border_mode=cv2.BORDER_CONSTANT, value=0
        ),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=config.AUG_COLOR_JITTER_PROB),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.5),
        ], p=config.AUG_BLUR_PROB),
        A.CoarseDropout(
            max_holes=8, max_height=int(config.IMG_SIZE*0.1), max_width=int(config.IMG_SIZE*0.1),
            min_holes=1, min_height=int(config.IMG_SIZE*0.05), min_width=int(config.IMG_SIZE*0.05),
            fill_value=0, p=config.AUG_COARSE_DROPOUT_PROB
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', label_fields=[], remove_invisible=False)) # Crucial: remove_invisible=False

    # Validation Transform (minimal augmentation)
    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', label_fields=[], remove_invisible=False))

    # Datasets
    try:
        train_dataset = FaceLandmark300WDataset(
            root_dir=config.ROOT_DATA_DIR,
            file_list_path=config.TRAIN_LIST_FILE,
            img_size=config.IMG_SIZE,
            feature_map_size=(config.FEATURE_H, config.FEATURE_W),
            sigma=config.SIGMA,
            transform=train_transform,
            num_landmarks=config.NUM_LANDMARKS
        )
        val_dataset = FaceLandmark300WDataset(
            root_dir=config.ROOT_DATA_DIR,
            file_list_path=config.TEST_LIST_FILE, # Use test list for validation
            img_size=config.IMG_SIZE,
            feature_map_size=(config.FEATURE_H, config.FEATURE_W),
            sigma=config.SIGMA,
            transform=val_transform,
            num_landmarks=config.NUM_LANDMARKS
        )
    except (FileNotFoundError, ValueError) as e:
         print(f"Error initializing datasets: {e}")
         print("Please ensure dataset paths and file lists are correct in config.py and file lists exist.")
         return

    # DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=config.DROP_LAST_BATCH # Important for consistent batch sizes, esp. with BatchNorm
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")

    # --- Loss Function, Optimizer, Scheduler ---
    print("Setting up loss, optimizer, and scheduler...")
    criterion = TotalLoss(
        num_blocks=config.NUM_BLOCKS,
        beta=config.LOSS_HEATMAP_BETA,
        weight_intermediate=config.LOSS_INTERMEDIATE_WEIGHT_W,
        aw_alpha=config.AWING_ALPHA,
        aw_omega=config.AWING_OMEGA,
        smooth_l1_beta=config.LOSS_COORD_SMOOTH_L1_BETA
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    if config.LR_SCHEDULER_ENABLE:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.LR_DECAY_EPOCHS,
            gamma=config.LR_DECAY_FACTOR
        )
        print(f"Using StepLR scheduler: step_size={config.LR_DECAY_EPOCHS}, gamma={config.LR_DECAY_FACTOR}")
    else:
        lr_scheduler = None
        print("LR scheduler disabled.")


    # --- Training Loop ---
    print("Starting training...")
    best_val_loss = float('inf')
    start_epoch = 0 # TODO: Add checkpoint loading logic to resume training

    # Store losses for plotting
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start_time = time.time()

        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]", leave=False)

        for batch_idx, batch in enumerate(train_pbar):
            # Expecting: image_tensor, landmarks_for_coord_loss, gt_heatmaps, landmarks_gt_img_scale_tensor, basename
            if len(batch) != 5: print(f"Warning: Train batch has {len(batch)} elements, expected 5."); continue
            images, gt_coords_feat, gt_heatmaps, _, _ = batch # Unpack, ignore last two for loss calc

            # Check for dummy data (indicating an error in dataset __getitem__)
            if isinstance(batch[-1], list) and any("error_" in b for b in batch[-1]):
                 print(f"Warning: Skipping batch {batch_idx+1} due to data loading errors.")
                 continue
            if isinstance(batch[-1], str) and "error_" in batch[-1]: # Handle single error case
                 print(f"Warning: Skipping batch {batch_idx+1} due to data loading error.")
                 continue


            images = images.to(device)
            gt_coords_feat = gt_coords_feat.to(device)
            gt_heatmaps = gt_heatmaps.to(device)

            optimizer.zero_grad()

            # Forward pass
            intermediate_heatmaps = model(images)

            # Loss calculation
            loss = criterion(intermediate_heatmaps, gt_coords_feat, gt_heatmaps)

            # Backward pass and optimization
            loss.backward()
            # Gradient clipping (optional but recommended)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Log batch loss periodically
            # if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            #      print(f"  Batch [{batch_idx+1}/{len(train_dataloader)}], Batch Loss: {loss.item():.4f}")

        avg_epoch_train_loss = running_loss / len(train_dataloader)
        train_losses.append(avg_epoch_train_loss)
        print(f"Epoch {epoch+1} Training Summary: Avg Loss = {avg_epoch_train_loss:.4f}")


        # --- Validation Phase ---
        if (epoch + 1) % config.VALIDATION_INTERVAL == 0:
            model.eval()
            val_running_loss = 0.0
            val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]", leave=False)

            with torch.no_grad():
                for batch in val_pbar:
                    if len(batch) != 5: print(f"Warning: Val batch has {len(batch)} elements, expected 5."); continue
                    images_val, gt_coords_feat_val, gt_heatmaps_val, _, _ = batch
                    if isinstance(batch[-1], list) and any("error_" in b for b in batch[-1]): continue # Skip error samples
                    if isinstance(batch[-1], str) and "error_" in batch[-1]: continue

                    images_val = images_val.to(device)
                    gt_coords_feat_val = gt_coords_feat_val.to(device)
                    gt_heatmaps_val = gt_heatmaps_val.to(device)

                    intermediate_heatmaps_val = model(images_val)
                    loss_val = criterion(intermediate_heatmaps_val, gt_coords_feat_val, gt_heatmaps_val)
                    val_running_loss += loss_val.item()
                    val_pbar.set_postfix(loss=f"{loss_val.item():.4f}")


            avg_epoch_val_loss = val_running_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0.0
            val_losses.append(avg_epoch_val_loss)
            print(f"Epoch {epoch+1} Validation Summary: Avg Loss = {avg_epoch_val_loss:.4f}")

            # --- Save Best Model ---
            if avg_epoch_val_loss < best_val_loss:
                best_val_loss = avg_epoch_val_loss
                # Save the model state dict (unwrap if using DataParallel)
                model_to_save = model.module if use_dp else model
                checkpoint_path = os.path.join(config.SAVE_DIR, "best_model.pth")
                try:
                    torch.save(model_to_save.state_dict(), checkpoint_path)
                    print(f"** Validation loss improved to {best_val_loss:.4f}. Saved best model to {checkpoint_path} **")
                except Exception as e:
                    print(f"Error saving best model: {e}")
            else:
                 print(f"Validation loss did not improve from {best_val_loss:.4f}.")

            # --- Optional Periodic Visualization ---
            if (epoch + 1) % config.VISUALIZATION_INTERVAL == 0:
                print("Visualizing validation samples...")
                try:
                    # Prepare config dict needed by visualization function
                    vis_config = {
                         'IMG_SIZE': config.IMG_SIZE,
                         'FEATURE_W': config.FEATURE_W,
                         'FEATURE_H': config.FEATURE_H,
                         'NUM_LANDMARKS': config.NUM_LANDMARKS,
                         'SAVE_DIR': config.SAVE_DIR # Pass save dir for saving plot
                    }
                    visualize_predictions_heatmaps(
                        model.module if use_dp else model, # Pass the underlying model if DP used
                        val_dataset,
                        vis_config,
                        num_samples_to_show=config.NUM_VISUALIZE_SAMPLES,
                        device=device
                    )
                except Exception as e:
                    print(f"Error during visualization: {e}")


        # --- End of Epoch ---
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s. Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Update Learning Rate
        if lr_scheduler:
            lr_scheduler.step()

        # --- Save Periodic Checkpoint (Optional) ---
        if not config.SAVE_BEST_MODEL_ONLY and (epoch + 1) % config.SAVE_CHECKPOINT_EPOCHS == 0:
            model_to_save = model.module if use_dp else model
            checkpoint_path = os.path.join(config.SAVE_DIR, f"model_epoch_{epoch+1}.pth")
            try:
                torch.save(model_to_save.state_dict(), checkpoint_path)
                print(f"Saved periodic checkpoint to {checkpoint_path}")
            except Exception as e:
                 print(f"Error saving periodic checkpoint: {e}")

        # --- Plot Losses ---
        if (epoch + 1) % config.VISUALIZATION_INTERVAL == 0: # Plot losses periodically
             plt.figure(figsize=(10, 5))
             plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
             # Align validation losses (plotted every val interval)
             val_epochs = range(config.VALIDATION_INTERVAL, (len(val_losses) * config.VALIDATION_INTERVAL) + 1, config.VALIDATION_INTERVAL)
             plt.plot(val_epochs, val_losses, label='Validation Loss', marker='o')
             plt.xlabel('Epoch')
             plt.ylabel('Loss')
             plt.title('Training and Validation Loss')
             plt.legend()
             plt.grid(True)
             loss_plot_path = os.path.join(config.SAVE_DIR, 'loss_curve.png')
             plt.savefig(loss_plot_path)
             print(f"Loss curve saved to {loss_plot_path}")
             plt.close() # Close plot to free memory


    # --- End of Training ---
    print("\nTraining finished!")

    # Save the final model
    final_model_path = os.path.join(config.SAVE_DIR, "final_model.pth")
    model_to_save = model.module if use_dp else model
    try:
        torch.save(model_to_save.state_dict(), final_model_path)
        print(f"Saved final model to {final_model_path}")
    except Exception as e:
        print(f"Error saving final model: {e}")

if __name__ == '__main__':
    main()
