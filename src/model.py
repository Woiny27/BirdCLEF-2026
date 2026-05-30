"""
Create a BirdCLEF 2026 model using timm ConvNeXt Tiny.

Requirements:

- Use pretrained weights
- Output 234 classes
- Multi-label classification
- BCEWithLogitsLoss compatible
- Return logits
"""

import torch
import torch.nn as nn
import timm


class BirdCLEFModel(nn.Module):
    """
    BirdCLEF 2026 Multi-label Classification Model
    
    Uses ConvNeXt Tiny backbone with pretrained weights for bird species
    classification on mel spectrogram inputs.
    """
    
    def __init__(self, num_classes=234, pretrained=True):
        super(BirdCLEFModel, self).__init__()
        
        # Load pretrained ConvNeXt Tiny
        self.backbone = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            in_chans=1  # Grayscale mel spectrograms
        )
        
        # Get number of features from backbone
        num_features = self.backbone.get_classifier().in_features
        
        # Replace classifier head
        self.backbone.head = nn.Linear(num_features, num_classes)
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 1, height, width)
               representing mel spectrograms
        
        Returns:
            logits: Output logits of shape (batch_size, num_classes)
                   compatible with BCEWithLogitsLoss
        """
        logits = self.backbone(x)
        return logits


if __name__ == "__main__":
    # Test model instantiation
    model = BirdCLEFModel(num_classes=234, pretrained=True)
    print(model)
    
    # Test forward pass
    x = torch.randn(4, 1, 128, 300)  # Batch of 4 mel spectrograms
    logits = model(x)
    print(f"Output shape: {logits.shape}")  # Should be (4, 234)
