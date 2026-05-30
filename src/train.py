"""
Create a PyTorch training script for BirdCLEF 2026.

Requirements:
- Load Dataset
- Create DataLoader
- Use ConvNeXt Tiny model
- BCEWithLogitsLoss
- AdamW optimizer
- Save best model.pt
- Display training and validation loss
- Support GPU if available
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src.model import BirdCLEFModel


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (spectrograms, labels) in enumerate(tqdm(train_loader, desc="Training")):
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(spectrograms)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for spectrograms, labels in tqdm(val_loader, desc="Validating"):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(spectrograms)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main():
    """Main training function."""
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-3
    num_classes = 234
    
    # Create model
    model = BirdCLEFModel(num_classes=num_classes, pretrained=True).to(device)
    print(f"Model loaded on {device}")
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # TODO: Load train and validation datasets
    # train_dataset = BirdCLEFDataset(...)
    # val_dataset = BirdCLEFDataset(...)
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # TODO: Uncomment when datasets are ready
        # train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        # val_loss = validate(model, val_loader, criterion, device)
        
        # print(f"Train Loss: {train_loss:.4f}")
        # print(f"Val Loss: {val_loss:.4f}")
        
        # # Save best model
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), "model.pt")
        #     print("Model saved!")
    
    print("Training complete!")


if __name__ == "__main__":
    main()
