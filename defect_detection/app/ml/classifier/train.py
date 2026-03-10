import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms

from app.ml.classifier.model import DefectCNN
from app.config import settings


def get_transform(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize(settings.image_size),
            transforms.CenterCrop(settings.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    return transforms.Compose([
        transforms.Resize(settings.image_size),
        transforms.CenterCrop(settings.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def train(epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    os.makedirs(settings.checkpoint_dir, exist_ok=True)

    real_dataset = datasets.ImageFolder(settings.real_dir, transform=get_transform(train=True))

    syn_normal = os.path.join(settings.synthetic_dir, "normal")
    syn_defect = os.path.join(settings.synthetic_dir, "defective")

    if os.path.exists(syn_normal) and os.path.exists(syn_defect):
        syn_dataset = datasets.ImageFolder(settings.synthetic_dir, transform=get_transform(train=True))
        combined = ConcatDataset([real_dataset, syn_dataset])
        print(f"Dataset: {len(real_dataset)} real + {len(syn_dataset)} synthetic = {len(combined)} total")
    else:
        combined = real_dataset
        print(f"Dataset: {len(combined)} real images only")

    loader = DataLoader(combined, batch_size=settings.batch_size, shuffle=True, num_workers=2)

    model = DefectCNN(num_classes=settings.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=settings.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        acc = 100.0 * correct / total
        print(f"Epoch [{epoch}/{epochs}]  Loss: {running_loss/len(loader):.4f}  Acc: {acc:.2f}%")
        scheduler.step()

    ckpt_path = os.path.join(settings.checkpoint_dir, "classifier.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Classifier saved to {ckpt_path}")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    train(args.epochs)
