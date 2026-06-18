import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Tuple, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.models import ResNetSpoofDetector
    from src.dataset import create_dataloader
except ImportError:
    from models import ResNetSpoofDetector
    from dataset import create_dataloader


def train_resnet(
    train_protocol: str = "data/protocols/train_protocol.csv",
    val_protocol: str = "data/protocols/dev_protocol.csv",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    save_path: str = "models/resnet_spoof_detector.pth",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[nn.Module, Dict[str, Any]]:
    print(f"[*] Training ResNet-18 model on device: {device}")

    train_loader = create_dataloader(train_protocol, batch_size=batch_size, shuffle=True, augment=True)
    val_loader = create_dataloader(val_protocol, batch_size=batch_size, shuffle=False, augment=False) if Path(val_protocol).exists() else None

    model = ResNetSpoofDetector(in_channels=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        start_t = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_train_loss = running_loss / max(1, total)
        epoch_train_acc = 100.0 * correct / max(1, total)
        epoch_val_loss, epoch_val_acc = epoch_train_loss, epoch_train_acc

        if val_loader is not None:
            model.eval()
            val_running_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, targets, _ in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_running_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            epoch_val_loss = val_running_loss / max(1, val_total)
            epoch_val_acc = 100.0 * val_correct / max(1, val_total)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        elapsed = time.time() - start_t

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_loss <= best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)

    print(f"[+] Saved model checkpoint to {save_path}")
    return model, history


if __name__ == "__main__":
    train_resnet(epochs=5)
