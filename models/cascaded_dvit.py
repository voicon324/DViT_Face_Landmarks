import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from .dvit_modules import DViT # Relative import

class PredictionHead(nn.Module):
    """ Regresses features to heatmaps using a 1x1 Conv """
    def __init__(self, in_chans=256, num_landmarks=98):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, num_landmarks, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class CascadedDViT(nn.Module):
    def __init__(self, img_size=256, num_blocks=8, num_landmarks=98,
                 backbone_out_chans=256, 
                 dvit_chans=256,        
                 embed_dim=512,         
                 depth=2, num_heads=8,  
                 feature_size=(32, 32)  
                 ):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_landmarks = num_landmarks
        self.feature_size = feature_size

        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)

        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
     
        self.backbone_layers = nn.Sequential(
            self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool,
            self.backbone.layer1, self.backbone.layer2
        )
       
        if backbone_out_chans != dvit_chans:
             self.initial_proj = nn.Conv2d(backbone_out_chans, dvit_chans, kernel_size=1)
        else:
             self.initial_proj = nn.Identity()


        self.lsc_proj = nn.Conv2d(backbone_out_chans + dvit_chans, dvit_chans, kernel_size=1)


        self.prediction_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.prediction_blocks.append(
                DViT(in_chans=dvit_chans, out_chans=dvit_chans, embed_dim=embed_dim,
                     depth=depth, num_heads=num_heads, feature_size=feature_size)
            )

        self.heads = nn.ModuleList()
        for _ in range(num_blocks):
            self.heads.append(
                PredictionHead(in_chans=dvit_chans, num_landmarks=num_landmarks)
            )

    def forward(self, x):

        backbone_features = self.backbone_layers(x) 
      
        if backbone_features.shape[-2:] != self.feature_size:
             print(f"Warning: Backbone output size {backbone_features.shape[-2:]} != target {self.feature_size}. Adjusting...")
         
        current_features = self.initial_proj(backbone_features)


        intermediate_heatmaps = []

        for i in range(self.num_blocks):
    
            lsc_input = torch.cat((backbone_features, current_features), dim=1)
         
            combined_features = self.lsc_proj(lsc_input)

            refined_features = self.prediction_blocks[i](combined_features)

            heatmap = self.heads[i](refined_features)
            intermediate_heatmaps.append(heatmap)

            current_features = refined_features 

        return intermediate_heatmaps