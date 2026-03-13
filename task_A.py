
import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Check device for training
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device} | torch={torch.__version__} | cuda={torch.cuda.is_available()}")


target_shift = lambda t: t - 1          #PyTorch’s CrossEntropyLoss expects class indices starting from 0.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Loading data, training and testing
data_root = "./data"
train_set = datasets.EMNIST(root=data_root, 
                            split="letters", 
                            train=True,
                            download=True, 
                            transform=transform, 
                            target_transform=target_shift)
test_set  = datasets.EMNIST(root=data_root, 
                            split="letters", 
                            train=False,
                            download=True, 
                            transform=transform, 
                            target_transform=target_shift)

# Save EMNIST sample grid (letters split)
import numpy as np
os.makedirs("results", exist_ok=True)

def save_emnist_row(dataset, n=4, path="results/taskA_emnist_letters_row.png"):
    idx = np.random.choice(len(dataset), size=n, replace=False)
    imgs, labels = [], []
    for i in idx:
        img, lab = dataset[i]
        imgs.append(img.squeeze(0).numpy())
        labels.append(int(lab))
    # 1 row with n columns
    fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
    for ax, im, lab in zip(axes, imgs, labels):
        ax.imshow(im, cmap="gray")
        ax.set_title(chr(lab + ord('A')))  # Capital letters A–Z
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)

# Calling it once after loading train_set
save_emnist_row(train_set)
print("Saved EMNIST sample row to results/taskA_emnist_letters_row.png")


batch_size = 128
num_workers = 2 if device.type != "mps" else 0
pin_mem = (device.type == "cuda")  

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_mem)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_mem)

# Simple CNN architecture
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(in_features=8 * 14 * 14, out_features=26),
).to(device)

criterion = nn.CrossEntropyLoss()                     # loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop for one epoch
def train_one_epoch(model, loader, optimizer, criterion, device, max_batches_store=None):
    model.train()
    batch_losses_all = []
    batch_losses_limited = []
    correct = total = 0

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        batch_losses_all.append(loss_val)
        if max_batches_store is None or batch_idx <= max_batches_store:
            batch_losses_limited.append(loss_val)

        with torch.no_grad():
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    avg_loss = sum(batch_losses_all) / len(batch_losses_all)
    train_acc = 100.0 * correct / total
    return avg_loss, train_acc, batch_losses_all, batch_losses_limited

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total

def main():
    epochs = 5
    per_epoch_all_losses = []
    per_epoch_limited_losses = []

    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        avg_loss, tr_acc, all_losses, limited_losses = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_batches_store=120
        )
        te_acc = evaluate(model, test_loader, device)
        per_epoch_all_losses.append(all_losses)
        per_epoch_limited_losses.append(limited_losses)
        print(f"Epoch {ep:02d}/{epochs} | loss={avg_loss:.4f} | train_acc={tr_acc:.2f}% | test_acc={te_acc:.2f}%")

    total_time = time.perf_counter() - t0
    print(f"\nTotal training time: {total_time:.2f}s")
    os.makedirs("results", exist_ok=True)

    # Plot 1: First 120 batches only
    plt.figure(figsize=(7.5, 4.8))
    for ep, losses in enumerate(per_epoch_limited_losses, start=1):
        xs = range(1, len(losses) + 1)
        plt.plot(xs, losses, label=f"Epoch {ep}")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Batch (First 120)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/taskA_loss_first120.png", dpi=160)

    # Plot 2: All batches
    plt.figure(figsize=(7.5, 4.8))
    for ep, losses in enumerate(per_epoch_all_losses, start=1):
        xs = range(1, len(losses) + 1)
        plt.plot(xs, losses, label=f"Epoch {ep}")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Batch (All Batches)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/taskA_loss_all_batches.png", dpi=160)

    with open("results/taskA_metrics.txt", "w") as f:
        f.write(f"device={device}\n")
        f.write(f"torch={torch.__version__}\n")
        f.write(f"epochs={epochs}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"total_time_s={total_time:.2f}\n")

    print('Saved plots to results/')

if __name__ == "__main__":
    main()
