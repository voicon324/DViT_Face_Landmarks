import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from .dvit_modules import DViT # Relative import

# PredictionHead class remains the same
class PredictionHead(nn.Module):
    """ Regresses features to heatmaps using a 1x1 Conv """
    def __init__(self, in_chans=256, num_landmarks=98):
        super().__init__()
        # Reduce intermediate channels? Paper doesn't specify head details much.
        # Option 1: Direct 1x1 conv
        self.conv = nn.Conv2d(in_chans, num_landmarks, kernel_size=1)
        # Option 2: Intermediate layer (like some detection heads)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_chans, in_chans // 2, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_chans // 2, num_landmarks, kernel_size=1)
        # )

    def forward(self, x):
        return self.conv(x)

# CascadedDViT class - Check backbone output channels and projection carefully
class CascadedDViT(nn.Module):
    def __init__(self, img_size=256, num_blocks=8, num_landmarks=98,
                 backbone_out_chans_target=128, # Target channels after layer2 of ResNet18
                 dvit_chans=256,
                 embed_dim=512,
                 depth=2, num_heads=8,
                 feature_size=(32, 32) # Target spatial size for DViT blocks
                 ):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_landmarks = num_landmarks
        self.dvit_chans = dvit_chans
        self.feature_size = feature_size # Expected spatial size (H, W) for DViT input/output

        # --- Backbone (ResNet18 up to layer2) ---
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)
        # Remove layers after layer2
        self.backbone.layer3 = nn.Identity()
        self.backbone.layer4 = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        # Explicitly define the layers to use (conv1 to layer2)
        self.backbone_layers = nn.Sequential(
            self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool,
            self.backbone.layer1, # Output channels: 64
            self.backbone.layer2  # Output channels: 128
        )
        # Determine the actual output channels from the backbone part we use
        # For ResNet18, layer2 outputs 128 channels.
        actual_backbone_out_chans = backbone_out_chans_target # Should be 128 for layer2

        # --- Feature Size Adjustment ---
        # Calculate expected output size from backbone_layers
        # Input: img_size (e.g., 256)
        # Conv1 (s=2, p=3): ceil((256-7+2*3)/2)+1 = ceil(258/2)+1 = 129+1 = 130 ??? NO
        # Conv1 (k=7, s=2, p=3): floor((256 + 2*3 - 7)/2) + 1 = floor(255/2)+1 = 127+1 = 128
        # MaxPool (k=3, s=2, p=1): floor((128 + 2*1 - 3)/2) + 1 = floor(127/2)+1 = 63+1 = 64
        # Layer1 (no stride change): 64
        # Layer2 (block 1, s=2 in conv1/shortcut): floor((64 + 2*padding - kernel)/2) + 1
        # Layer2 (first block shortcut s=2, conv s=2): floor((64-1)/2)+1 = floor(63/2)+1 = 31+1 = 32
        # So, expected output H, W is img_size / 8 = 256 / 8 = 32
        # This matches the default feature_size=(32, 32)

        # --- Initial Projection (if backbone output doesn't match DViT input channels) ---
        if actual_backbone_out_chans != dvit_chans:
             print(f"Projecting backbone output {actual_backbone_out_chans} -> {dvit_chans}")
             self.initial_proj = nn.Conv2d(actual_backbone_out_chans, dvit_chans, kernel_size=1)
        else:
             self.initial_proj = nn.Identity()


        # --- Long-Short Context (LSC) Projection ---
        # Input: Concatenated backbone features and previous DViT block output
        lsc_input_chans = actual_backbone_out_chans + dvit_chans
        # Project LSC input down to DViT input channels
        self.lsc_proj = nn.Conv2d(lsc_input_chans, dvit_chans, kernel_size=1)
        print(f"LSC Projection: {lsc_input_chans} -> {dvit_chans}")

        # --- Cascaded DViT Blocks ---
        self.prediction_blocks = nn.ModuleList()
        for i in range(num_blocks):
            print(f"Creating DViT block {i+1}/{num_blocks}")
            # Assuming DViT blocks maintain channel size (in_chans=dvit_chans, out_chans=dvit_chans)
            # And maintain spatial size (upscale_factor=1 inside DViT components)
            self.prediction_blocks.append(
                DViT(in_chans=dvit_chans, out_chans=dvit_chans, embed_dim=embed_dim,
                     depth=depth, num_heads=num_heads, feature_size=feature_size,
                     upscale_factor=1) # Ensure DViT block itself doesn't upscale
            )

        # --- Prediction Heads ---
        self.heads = nn.ModuleList()
        for _ in range(num_blocks):
            self.heads.append(
                PredictionHead(in_chans=dvit_chans, num_landmarks=num_landmarks)
            )

    def forward(self, x):
        # 1. Get Backbone Features
        backbone_features = self.backbone_layers(x) # Expected shape: (B, 128, 32, 32)

        # Check spatial dimensions match target feature_size
        if backbone_features.shape[-2:] != self.feature_size:
             # Add adaptive pooling or interpolation if mismatch occurs
             print(f"Warning: Backbone output size {backbone_features.shape[-2:]} != target {self.feature_size}. Applying AdaptiveAvgPool2d.")
             pool = nn.AdaptiveAvgPool2d(self.feature_size)
             backbone_features = pool(backbone_features)

        # 2. Initial Projection
        current_features = self.initial_proj(backbone_features) # Shape: (B, dvit_chans, 32, 32)

        intermediate_heatmaps = []

        # 3. Iterate through DViT blocks
        for i in range(self.num_blocks):
            # LSC Fusion: Concatenate backbone features and current features
            # Ensure backbone_features spatial size matches current_features (should if pooling applied)
            if backbone_features.shape[-2:] != current_features.shape[-2:]:
                 # This shouldn't happen if pooling is done correctly, but as a safeguard
                 print(f"Warning: Mismatch in LSC shapes! Backbone: {backbone_features.shape}, Current: {current_features.shape}. Resizing backbone.")
                 backbone_features_resized = F.interpolate(backbone_features, size=current_features.shape[-2:], mode='bilinear', align_corners=False)
                 lsc_input = torch.cat((backbone_features_resized, current_features), dim=1)
            else:
                 lsc_input = torch.cat((backbone_features, current_features), dim=1)
            # Shape: (B, actual_backbone_out_chans + dvit_chans, 32, 32)

            # Project LSC features
            combined_features = self.lsc_proj(lsc_input) # Shape: (B, dvit_chans, 32, 32)

            # Pass through DViT block
            refined_features = self.prediction_blocks[i](combined_features) # Shape: (B, dvit_chans, 32, 32)

            # Generate heatmap prediction
            heatmap = self.heads[i](refined_features) # Shape: (B, num_landmarks, 32, 32)
            intermediate_heatmaps.append(heatmap)

            # Update current features for the next block (output of DViT block)
            current_features = refined_features

        # Return list of heatmaps from all stages
        return intermediate_heatmaps