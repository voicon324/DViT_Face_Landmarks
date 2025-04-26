import torch
import torch.nn as nn
import torch.nn.functional as F
from .vit_components import ViTBlock # Relative import

class SpatialViT(nn.Module):
    """ ViT operating on spatial patches """
    def __init__(self, in_chans=256, embed_dim=512, depth=2, num_heads=8, mlp_ratio=4.,
                 feature_size=(32, 32), patch_kernel=2, patch_stride=2, norm_layer=nn.LayerNorm, upscale_factor=2):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feature_size = feature_size
        self.patch_stride = patch_stride
        self.upscale_factor = upscale_factor 

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_kernel, stride=patch_stride)
        self.num_patches = (feature_size[0] // patch_stride) * (feature_size[1] // patch_stride)
        self.new_feature_size = (feature_size[0] // patch_stride, feature_size[1] // patch_stride)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
 
        self.blocks = nn.ModuleList([
            ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.final_channels = embed_dim // (upscale_factor ** 2)
        if embed_dim != self.final_channels * (upscale_factor ** 2):
             print(f"SpatialViT: Projecting {embed_dim} to {self.final_channels * (upscale_factor ** 2)} for PixelShuffle")
             self.proj_for_shuffle = nn.Linear(embed_dim, self.final_channels * (upscale_factor ** 2))
        else:
             self.proj_for_shuffle = nn.Identity()

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)


    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.feature_size[0] and W == self.feature_size[1], \
            f"Input spatial dim ({H}x{W}) doesn't match configured feature_size {self.feature_size}"

        x = self.patch_embed(x) 

        x = x.flatten(2).transpose(1, 2) 

        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) 

        x = self.proj_for_shuffle(x) 

        N_H, N_W = self.new_feature_size
        x = x.transpose(1, 2).reshape(B, self.final_channels * (self.upscale_factor**2), N_H, N_W)
        x = self.pixel_shuffle(x)

        return x

class ChannelViT(nn.Module):
    """ ViT operating on channel dimension """
    def __init__(self, in_chans=256, embed_dim=512, depth=2, num_heads=8, mlp_ratio=4.,
                 feature_size=(32, 32), spatial_kernel=2, spatial_stride=2, norm_layer=nn.LayerNorm, upscale_factor=2):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim 
        self.num_heads = num_heads
        self.feature_size = feature_size
        self.upscale_factor = upscale_factor 

        self.spatial_conv = nn.Conv2d(in_chans, in_chans, kernel_size=spatial_kernel, stride=spatial_stride)
        self.spatially_reduced_size = (feature_size[0] // spatial_stride, feature_size[1] // spatial_stride)
        num_spatial_locations = self.spatially_reduced_size[0] * self.spatially_reduced_size[1]

        self.proj_embed = nn.Linear(num_spatial_locations, embed_dim)
        self.num_channel_tokens = in_chans 

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_channel_tokens, embed_dim))

        self.blocks = nn.ModuleList([
            ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.proj_back = nn.Linear(embed_dim, num_spatial_locations)

        self.final_channels = in_chans // (upscale_factor ** 2)
        if in_chans != self.final_channels * (upscale_factor ** 2):
             print(f"ChannelViT: Projecting channels {in_chans} to {self.final_channels * (upscale_factor ** 2)} for PixelShuffle")
             self.proj_ch_for_shuffle = nn.Linear(in_chans, self.final_channels * (upscale_factor ** 2)) # Acts on channel dim B,N,C? -> B,C,N ?
             self.final_channels_shuffle_in = self.final_channels * (upscale_factor ** 2)
        else:
             self.proj_ch_for_shuffle = nn.Identity()
             self.final_channels_shuffle_in = in_chans

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)


    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.in_chans and H == self.feature_size[0] and W == self.feature_size[1], \
            f"Input dim ({B},{C},{H},{W}) doesn't match config ({self.in_chans},{self.feature_size})"

        x = self.spatial_conv(x) 
        H_r, W_r = self.spatially_reduced_size

        x = x.flatten(2)
        x = x.permute(0, 2, 1)
        x = x.permute(0, 2, 1).contiguous()

        x = self.proj_embed(x) 

        x = x + self.pos_embed 

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) 

        x = self.proj_back(x)
        x = x.reshape(B, self.in_chans, H_r, W_r)

        if isinstance(self.proj_ch_for_shuffle, nn.Linear):
             x = x.permute(0, 2, 3, 1) 
             x = self.proj_ch_for_shuffle(x)
             x = x.permute(0, 3, 1, 2)
        else:
             x = self.proj_ch_for_shuffle(x) 


        x = self.pixel_shuffle(x) 

        return x

class DViT(nn.Module):
    """ Dual Vision Transformer Block combining Spatial and Channel ViTs """
    def __init__(self, in_chans=256, out_chans=256, embed_dim=512, depth=2, num_heads=8, mlp_ratio=4.,
                 feature_size=(32, 32), **kwargs):
        super().__init__()
        self.spatial_vit = SpatialViT(in_chans=in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                      mlp_ratio=mlp_ratio, feature_size=feature_size, **kwargs)
        self.channel_vit = ChannelViT(in_chans=in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                      mlp_ratio=mlp_ratio, feature_size=feature_size, **kwargs)

        # Calculate output channels from each ViT (after pixel shuffle)
        spatial_out_chans = self.spatial_vit.final_channels
        channel_out_chans = self.channel_vit.final_channels

        combined_chans = spatial_out_chans + channel_out_chans

   
        self.conv_res = nn.Sequential(
            nn.Conv2d(combined_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans)
        )
        if in_chans != out_chans:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_chans)
            )
        else:
            self.shortcut = nn.Identity()

        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        spatial_features = self.spatial_vit(x)
        channel_features = self.channel_vit(x)

        combined = torch.cat((spatial_features, channel_features), dim=1)

        residual = self.conv_res(combined)
        shortcut = self.shortcut(identity)

        out = self.final_relu(residual + shortcut)
        return out