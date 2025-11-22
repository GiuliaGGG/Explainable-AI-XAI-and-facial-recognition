import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split

# ----------------------------------------------------
# 1. PATH TO DATASET
# ----------------------------------------------------
DATA_DIR = "BFW-Release/bfw-cropped-aligned/"  # change if needed

# ----------------------------------------------------
# 2. TRANSFORMS
# ----------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------------------------------
# 3. LOAD DATASET
# ----------------------------------------------------
print("Loading dataset...")
train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)

class_names = train_dataset.classes
num_classes = len(class_names)

print(f"Loaded dataset with {len(train_dataset)} images")
print(f"Detected {num_classes} classes:", class_names)

# ----------------------------------------------------
# 4. SAMPLE DATA & TRAIN/VAL SPLIT
# ----------------------------------------------------
# Use only a sample of the data for faster training
SAMPLE_SIZE = 2000  # Adjust this value (e.g., 1000, 2000, 5000)
print(f"Sampling {SAMPLE_SIZE} images from dataset...")

indices = list(range(len(train_dataset)))
sample_idx, _ = train_test_split(
    indices,
    train_size=SAMPLE_SIZE,
    shuffle=True,
    stratify=train_dataset.targets
)

# Now split the sample into train/val
print("Splitting sampled data into train/val...")
train_idx, val_idx = train_test_split(
    sample_idx,
    test_size=0.2,
    shuffle=True,
    stratify=[train_dataset.targets[i] for i in sample_idx]
)
print(f"Split complete: {len(train_idx)} train, {len(val_idx)} val")

# Create subsets
train_ds = torch.utils.data.Subset(train_dataset, train_idx)
val_ds = torch.utils.data.Subset(val_dataset, val_idx)

# Create data loaders (num_workers=0 for macOS compatibility)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

if __name__ == '__main__':
    # ----------------------------------------------------
    # 5. MODEL: RESNET18 (TRANSFER LEARNING)
    # ----------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # replace final FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # ----------------------------------------------------
    # 6. LOSS + OPTIMIZER
    # ----------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ----------------------------------------------------
    # 7. TRAINING LOOP
    # ----------------------------------------------------
    EPOCHS = 10
    print(f"\nStarting training for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # --- Validation ---
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total

        print(f"\nEpoch {epoch+1}/{EPOCHS} Summary: "
              f"Train Loss: {running_loss/len(train_loader):.4f} | "
              f"Val Acc: {accuracy*100:.2f}%\n")

    # ----------------------------------------------------
    # 8. SAVE MODEL
    # ----------------------------------------------------
    torch.save(model.state_dict(), "face_classifier_resnet18.pth")
    print("Model saved to face_classifier_resnet18.pth")