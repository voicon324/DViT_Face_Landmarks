import torch
import torch.nn as nn
import torch.nn.functional as F
from .vit_components import ViTBlock # Relative import

# SpatialViT and ChannelViT classes remain mostly the same as in the notebook
# Need careful checking of channel projections and PixelShuffle logic
class SpatialViT(nn.Module):
    """ ViT operating on spatial patches """
    def __init__(self, in_chans=256, embed_dim=512, depth=2, num_heads=8, mlp_ratio=4.,
                 feature_size=(32, 32), patch_kernel=2, patch_stride=2, norm_layer=nn.LayerNorm, upscale_factor=2):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feature_size = feature_size # e.g. (32, 32)
        self.patch_stride = patch_stride # e.g. 2
        self.upscale_factor = upscale_factor # e.g. 2

        # Patch embedding using Conv2d
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_kernel, stride=patch_stride)
        # Calculate the size of the feature map after patch embedding
        self.new_feature_size = (feature_size[0] // patch_stride, feature_size[1] // patch_stride) # e.g. (16, 16)
        self.num_patches = self.new_feature_size[0] * self.new_feature_size[1] # e.g. 256

        # Positional embedding
        # Requires grad needs to be True if it's learnable (which it usually is)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Projection for PixelShuffle: target channels = C_out * (r^2)
        # C_out is the desired number of channels AFTER PixelShuffle
        # If we want the output feature map to have similar channels to input (e.g., in_chans)
        # after upscaling by factor r, the input to PixelShuffle needs
        # C_in_shuffle = C_out_shuffle * (r^2) channels.
        # Let's aim for C_out_shuffle = embed_dim / (r^2) for simplicity if divisible,
        # otherwise project to a suitable number.
        self.final_channels_after_shuffle = embed_dim // (upscale_factor ** 2) # Target channels after shuffle
        if self.final_channels_after_shuffle * (upscale_factor ** 2) != embed_dim:
             # If not perfectly divisible, adjust target channels and project
             print(f"SpatialViT: Adjusting output channels for PixelShuffle.")
             # Aim for roughly embed_dim / 4 output channels maybe? Or match original in_chans?
             # Let's target original in_chans // (upscale_factor**2) if possible?
             # For flexibility, let's define target output channels, e.g., embed_dim//4 = 128
             # self.final_channels_after_shuffle = embed_dim // 4 # Example: target 128 chans out
             # Or maybe target closer to in_chans after shuffle?
             # Let's try to make it output embed_dim // 4 channels
             self.final_channels_after_shuffle = max(1, embed_dim // 4) # Ensure at least 1 chan
             target_shuffle_in_chans = self.final_channels_after_shuffle * (upscale_factor ** 2)
             print(f"SpatialViT: Projecting {embed_dim} -> {target_shuffle_in_chans} for PixelShuffle (Output Channels: {self.final_channels_after_shuffle})")
             self.proj_for_shuffle = nn.Linear(embed_dim, target_shuffle_in_chans)
             self.shuffle_in_channels = target_shuffle_in_chans
        else:
             # Perfectly divisible
             self.proj_for_shuffle = nn.Identity()
             self.shuffle_in_channels = embed_dim
             print(f"SpatialViT: Output Channels after shuffle: {self.final_channels_after_shuffle}")


        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)


    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.feature_size[0] and W == self.feature_size[1], \
            f"Input spatial dim ({H}x{W}) doesn't match configured feature_size {self.feature_size}"

        # 1. Patch Embedding
        x = self.patch_embed(x) # Shape: (B, embed_dim, H/stride, W/stride) e.g., (B, 512, 16, 16)
        N_H, N_W = x.shape[-2:]

        # 2. Flatten and Transpose for Transformer
        x = x.flatten(2).transpose(1, 2) # Shape: (B, num_patches, embed_dim) e.g., (B, 256, 512)

        # 3. Add Positional Embedding
        # TODO: Add interpolation for pos_embed if input size changes, but here feature_size is fixed.
        x = x + self.pos_embed # Shape: (B, 256, 512)

        # 4. Pass through Transformer Blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) # Shape: (B, 256, 512)

        # 5. Project for PixelShuffle (if needed)
        x = self.proj_for_shuffle(x) # Shape: (B, 256, shuffle_in_channels)

        # 6. Reshape for PixelShuffle
        # Transpose back: (B, shuffle_in_channels, num_patches) -> Reshape: (B, shuffle_in_channels, N_H, N_W)
        x = x.transpose(1, 2).reshape(B, self.shuffle_in_channels, N_H, N_W) # e.g. (B, 512, 16, 16) if no proj

        # 7. PixelShuffle
        x = self.pixel_shuffle(x) # Shape: (B, final_channels_after_shuffle, N_H*r, N_W*r) e.g. (B, 128, 32, 32)

        return x


class ChannelViT(nn.Module):
    """ ViT operating on channel dimension """
    def __init__(self, in_chans=256, embed_dim=512, depth=2, num_heads=8, mlp_ratio=4.,
                 feature_size=(32, 32), spatial_kernel=2, spatial_stride=2, norm_layer=nn.LayerNorm, upscale_factor=2):
        super().__init__()
        self.in_chans = in_chans # e.g., 256
        self.embed_dim = embed_dim # e.g., 512 (dimension for transformer)
        self.num_heads = num_heads
        self.feature_size = feature_size # e.g., (32, 32)
        self.upscale_factor = upscale_factor # e.g., 2

        # 1. Spatial Reduction (e.g., via Conv2d)
        # Use Conv2d to reduce spatial dimensions H, W while keeping C
        self.spatial_conv = nn.Conv2d(in_chans, in_chans, kernel_size=spatial_kernel, stride=spatial_stride, groups=in_chans) # Depthwise? Or regular conv? Paper isn't specific. Let's try regular.
        # self.spatial_conv = nn.Conv2d(in_chans, in_chans, kernel_size=spatial_kernel, stride=spatial_stride) # Regular Conv
        self.spatially_reduced_size = (feature_size[0] // spatial_stride, feature_size[1] // spatial_stride) # e.g., (16, 16)
        # The number of spatial locations after reduction becomes the 'sequence length' for the channel projection
        num_spatial_locations = self.spatially_reduced_size[0] * self.spatially_reduced_size[1] # e.g., 256

        # 2. Projection to Embedding Dimension
        # Each channel's feature vector (length num_spatial_locations) is projected to embed_dim
        self.proj_embed = nn.Linear(num_spatial_locations, embed_dim)
        # The number of 'tokens' is the number of input channels
        self.num_channel_tokens = in_chans # e.g., 256

        # 3. Positional Embedding (for channels)
        # Requires grad needs to be True if it's learnable
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_channel_tokens, embed_dim), requires_grad=True)

        # 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # 5. Projection Back to Spatial Dimension Size
        # Project each channel's transformed embedding back to the original number of spatial locations
        self.proj_back = nn.Linear(embed_dim, num_spatial_locations)

        # 6. Projection for PixelShuffle
        # We need to reshape the output of proj_back into (B, C, H_reduced, W_reduced)
        # Then apply PixelShuffle which requires C_in = C_out * r^2 channels.
        # Here, C_in is self.in_chans. We want C_out after shuffle.
        self.final_channels_after_shuffle = self.in_chans // (upscale_factor ** 2) # Target channels after shuffle
        if self.final_channels_after_shuffle * (upscale_factor ** 2) != self.in_chans:
             # If in_chans is not divisible by r^2, we need to adjust channels *before* PixelShuffle.
             # We can use a Conv2d 1x1 to project channels.
             target_shuffle_in_chans = self.final_channels_after_shuffle * (upscale_factor ** 2)
             print(f"ChannelViT: Projecting channels {self.in_chans} -> {target_shuffle_in_chans} for PixelShuffle (Output Channels: {self.final_channels_after_shuffle})")
             # This projection happens AFTER reshaping back to spatial format
             self.proj_ch_for_shuffle = nn.Conv2d(self.in_chans, target_shuffle_in_chans, kernel_size=1)
             self.shuffle_in_channels = target_shuffle_in_chans
        else:
             # No projection needed before shuffle
             self.proj_ch_for_shuffle = nn.Identity()
             self.shuffle_in_channels = self.in_chans
             print(f"ChannelViT: Output Channels after shuffle: {self.final_channels_after_shuffle}")

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)


    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.in_chans and H == self.feature_size[0] and W == self.feature_size[1], \
            f"Input dim ({B},{C},{H},{W}) doesn't match config ({self.in_chans},{self.feature_size})"

        # 1. Spatial Reduction
        x = self.spatial_conv(x) # Shape: (B, C, H_reduced, W_reduced) e.g., (B, 256, 16, 16)
        H_r, W_r = x.shape[-2:]

        # 2. Flatten spatial dimensions and Permute for Channel Attention
        x = x.flatten(2) # Shape: (B, C, H_reduced * W_reduced) e.g., (B, 256, 256)
        # We want to treat channels as the sequence, so transpose: (B, num_spatial_locations, C)
        # x = x.transpose(1, 2) # Shape: (B, 256, 256) -- NO, transpose needed

        # Flatten: (B, C, N) where N=H_r*W_r
        # Project requires input (B, *, in_features=N)
        # We want tokens to be channels, so should be (B, C, N) input to proj_embed?
        # Let's check proj_embed: Linear(N, embed_dim). Input should be (*, N)
        # So, x is (B, C, N). Pass to linear directly? No, linear acts on last dim.
        # Need to permute: (B, N, C) so linear acts on C? No, we want linear to act on N.
        # Input is (B, C, N). Need to apply proj_embed (Linear(N, embed_dim)) to each channel.
        # This means input should be (B, C, N). OK.

        # 3. Project spatial features to embedding dimension for each channel
        x = self.proj_embed(x) # Shape: (B, C, embed_dim) e.g., (B, 256, 512)

        # 4. Add Positional Embedding (to channel dimension)
        x = x + self.pos_embed # Shape: (B, 256, 512)

        # 5. Pass through Transformer Blocks
        for blk in self.blocks:
            x = blk(x) # Operates on the C dimension tokens
        x = self.norm(x) # Shape: (B, 256, 512)

        # 6. Project Back to Spatial Dimension Size
        x = self.proj_back(x) # Shape: (B, C, num_spatial_locations) e.g., (B, 256, 256)

        # 7. Reshape back to spatial feature map format
        x = x.reshape(B, self.in_chans, H_r, W_r) # e.g., (B, 256, 16, 16)

        # 8. Project Channels for PixelShuffle (if needed)
        x = self.proj_ch_for_shuffle(x) # Apply 1x1 Conv if necessary. Shape: (B, shuffle_in_channels, 16, 16)

        # 9. PixelShuffle
        x = self.pixel_shuffle(x) # Shape: (B, final_channels_after_shuffle, H_r*r, W_r*r) e.g. (B, 64, 32, 32)

        return x


# DViT block class - combining Spatial and Channel ViTs
class DViT(nn.Module):
    """ Dual Vision Transformer Block combining Spatial and Channel ViTs """
    def __init__(self, in_chans=256, out_chans=256, embed_dim=512, depth=2, num_heads=8, mlp_ratio=4.,
                 feature_size=(32, 32), spatial_patch_kernel=2, spatial_patch_stride=2,
                 channel_spatial_kernel=2, channel_spatial_stride=2, upscale_factor=1, # Added upscale_factor
                 norm_layer=nn.LayerNorm, act_layer=nn.ReLU): # Added activation layer parameter
        super().__init__()

        # Upscale factor should ideally be 1 for the main DViT block,
        # unless specifically designed for upsampling within the block.
        # The paper's diagram suggests PixelShuffle happens *after* the ViTs.
        # Let's assume upscale_factor=1 for the ViTs inside DViT block itself.
        # If upscaling is desired *within* the block, it should be set > 1.
        # For now, let's assume upscale_factor=1 meaning no pixel shuffle inside ViTs.

        self.spatial_vit = SpatialViT(in_chans=in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                      mlp_ratio=mlp_ratio, feature_size=feature_size,
                                      patch_kernel=spatial_patch_kernel, patch_stride=spatial_patch_stride,
                                      norm_layer=norm_layer, upscale_factor=upscale_factor)

        self.channel_vit = ChannelViT(in_chans=in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                      mlp_ratio=mlp_ratio, feature_size=feature_size,
                                      spatial_kernel=channel_spatial_kernel, spatial_stride=channel_spatial_stride,
                                      norm_layer=norm_layer, upscale_factor=upscale_factor)

        # Get output channels from each ViT (AFTER potential pixel shuffle if upscale_factor > 1)
        spatial_out_chans = self.spatial_vit.final_channels_after_shuffle
        channel_out_chans = self.channel_vit.final_channels_after_shuffle

        combined_chans = spatial_out_chans + channel_out_chans
        print(f"DViT Block: Spatial ViT Out Chans: {spatial_out_chans}, Channel ViT Out Chans: {channel_out_chans}, Combined: {combined_chans}")


        # Residual connection components
        # Using Conv-BN-ReLU -> Conv-BN structure like ResNet blocks
        self.conv_res = nn.Sequential(
            nn.Conv2d(combined_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            act_layer(inplace=True), # Use specified activation
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans)
        )

        # Shortcut connection to match dimensions if necessary
        # It needs to handle both channel changes and potential spatial changes if upscale_factor > 1
        if in_chans != out_chans or upscale_factor > 1:
            shortcut_layers = []
            if upscale_factor > 1:
                # If ViTs upscale, shortcut needs to upscale too (e.g., via Interpolate or ConvTranspose)
                # Using Interpolate is simpler
                 shortcut_layers.append(nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=False))
            shortcut_layers.append(nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False))
            shortcut_layers.append(nn.BatchNorm2d(out_chans))
            self.shortcut = nn.Sequential(*shortcut_layers)
            print(f"DViT Block: Using projection shortcut (in:{in_chans}, out:{out_chans}, upscale:{upscale_factor})")
        else:
            self.shortcut = nn.Identity()
            print("DViT Block: Using identity shortcut.")

        self.final_act = act_layer(inplace=True) # Final activation after adding shortcut

    def forward(self, x):
        identity = x

        spatial_features = self.spatial_vit(x)
        channel_features = self.channel_vit(x)

        # Concatenate features along the channel dimension
        # Ensure spatial dimensions match before concat if upscale_factor > 1
        # If upscale_factor=1, spatial dims should be H/stride * 1 = H/stride which matches
        # If upscale_factor=2, spatial dims H/stride * 2 = H. Need shortcut to match this.
        combined = torch.cat((spatial_features, channel_features), dim=1)

        # Pass combined features through residual convolutional layers
        residual = self.conv_res(combined)

        # Apply shortcut connection
        shortcut_out = self.shortcut(identity)

        # Add residual and shortcut
        out = self.final_act(residual + shortcut_out)
        return out