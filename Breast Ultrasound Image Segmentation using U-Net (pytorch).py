import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 1. Device Configuration (Crucial for Apple Silicon!)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Configuration & Paths
DATASET_PATH = '/Users/abrarimtiyaz/Downloads/Dataset_BUSI_with_GT/*/*.png'
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 60

# 3. Data Loading & Preprocessing
def load_image(path, size):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image at: {path}")
    image = cv2.resize(image, (size, size))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image / 255.0  # Normalize to 0-1
    return image

def load_data(root_path, size):
    images, masks = [], []
    has_mask = False 
    
    for path in sorted(glob(root_path)):
        img = load_image(path, size)
        if 'mask' in path:
            if has_mask: 
                masks[-1] += img
                masks[-1] = np.array(masks[-1] > 0.5, dtype='float32')
            else:
                masks.append(img)
                has_mask = True
        else:
            images.append(img)
            has_mask = False
            
    # PyTorch expects (Batch, Channels, Height, Width)
    # So we add a channel dimension using np.expand_dims
    X = np.expand_dims(np.array(images, dtype=np.float32), axis=1)
    y = np.expand_dims(np.array(masks, dtype=np.float32), axis=1)
    return X, y

print("Loading dataset into memory...")
X, y = load_data(DATASET_PATH, IMAGE_SIZE)

# Split 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split Train into Train & Validation (90/10 split of the training set)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Convert NumPy arrays to PyTorch Datasets
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. Define the PyTorch U-Net Architecture
class DoubleConv(nn.Module):
    """(Conv2D -> ReLU -> Conv2D -> ReLU)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(1, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256)) # Bottleneck
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128) # 128 + 128 (skip) = 256
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64, 32)
        
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(32, 16)
        
        # Output layer
        self.outc = nn.Conv2d(16, 1, kernel_size=1)
        # Note: We don't apply Sigmoid here. We use BCEWithLogitsLoss for numerical stability.

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)
        
        logits = self.outc(x)
        return logits

# Initialize Model, Loss, and Optimizer
model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss() # Combines Sigmoid and Binary Crossentropy safely
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 5. Training Loop with Early Stopping
print("Beginning Neural Network Training...")
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(EPOCHS):
    # --- Training Phase ---
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()           # Clear old gradients
        outputs = model(batch_X)        # Forward pass
        loss = criterion(outputs, batch_y) # Calculate loss
        loss.backward()                 # Backpropagation
        optimizer.step()                # Update weights
        
        train_loss += loss.item() * batch_X.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
            
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    # --- Early Stopping ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model weights
        torch.save(model.state_dict(), 'best_unet_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered! Restoring best weights.")
            break

# Load the best weights before evaluation
model.load_state_dict(torch.load('best_unet_model.pth'))

# 6. Evaluation & Metrics
print("\n--- Model Evaluation ---")
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        logits = model(batch_X)
        # Apply sigmoid to convert logits to probabilities, then threshold at 0.5
        probs = torch.sigmoid(logits)
        preds = (probs > 0.3).float().cpu().numpy() # Lowered threshold to boost Recall
        
        all_preds.append(preds)
        all_true.append(batch_y.numpy())

# Flatten arrays and force them into strict binary integers for scikit-learn
y_true_flat = np.concatenate(all_true).flatten().astype(int)
y_pred_flat = np.concatenate(all_preds).flatten().astype(int)

# Calculate Metrics
iou = jaccard_score(y_true_flat, y_pred_flat)
precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)

print(f"Mean IoU:      {iou:.3f}")
print(f"Precision:     {precision:.3f}")
print(f"Recall:        {recall:.3f}")
print(f"F1 Score:      {f1:.3f}")