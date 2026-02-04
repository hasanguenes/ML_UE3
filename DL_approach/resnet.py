# source:
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# https://medium.com/@meda.abdullah/transfer-learning-for-computer-vision-a-pytorch-tutorial-c5c4e022bcdf

import torch
import torch.nn as nn
from torchvision import models


class ResNet18(nn.Module):
    """    
    Features:
    - Training from scratch or with pretrained weights (ImageNet)
    - Optional dropout for regularization
    - Freeze/unfreeze for transfer learning strategies
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 43,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        dropout: float = 0.0,
    ):
        """
        Initialize ResNet-18 model.
        in_channels: Number of input channels (3 for RGB, 1 for grayscale)
        num_classes: Number of output classes (43 for GTSRB, 10 for CIFAR-10)
        pretrained: Use ImageNet pretrained weights for transfer learning
        freeze_backbone: If True, only train final layer (feature extraction mode)
        dropout: Dropout probability before final layer (0.0 = no dropout)
                Typical values: 0.3-0.5 for regularization
        """
        super().__init__()
        
        # ResNet-18 architecture
        if pretrained:
            # Load with ImageNet weights (1000 classes)
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        else:
            # Random initialization (train from scratch)
            self.backbone = models.resnet18(weights=None)
        
        # Adapt first layer for non-RGB inputs if needed
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Freeze backbone layers if requested (for transfer learning)
        # This means: only the final layer will be trained - much faster
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        in_features = self.backbone.fc.in_features  # 512 for ResNet-18
        
        if dropout > 0.0:
            # Add dropout before final layer
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            # No dropout
            self.backbone.fc = nn.Linear(in_features, num_classes)
        
        # Store configuration
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout = dropout
        self._is_frozen = freeze_backbone
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def freeze_backbone(self):
        # Freeze all backbone layers except the final classifier.

        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:  # Don't freeze final layer
                param.requires_grad = False
        self._is_frozen = True
        print("Backbone frozen - only final layer is trainable")
    
    def unfreeze_backbone(self):
        
        #Unfreeze all layers for fine-tuning
        
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._is_frozen = False
        print("Backbone unfrozen - all layers are trainable")
    
    def is_frozen(self) -> bool:
        
        # Check if backbone is currently frozen.
        # Returns: True if backbone is frozen, False otherwise
        
        return self._is_frozen
