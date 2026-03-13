
import os, time, csv, json
from typing import Tuple
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
if device.type == "mps":
    torch.set_default_dtype(torch.float32)   # MPS works well with float32

# Loading data, training and testing
def get_datasets():
    target_shift = lambda t: t - 1
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    root = "./data"
    train_set = datasets.EMNIST(root=root, split="letters", train=True,
                                download=True, transform=transform, target_transform=target_shift)
    test_set  = datasets.EMNIST(root=root, split="letters", train=False,
                                download=True, transform=transform, target_transform=target_shift)
    return train_set, test_set

def get_loaders(train_set, test_set, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    num_workers = 0 if device.type == "mps" else 2
    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_mem)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_mem)
    return train_loader, test_loader

# Complex CNN architecture
def make_cnn(
    num_classes: int,
    c1: int = 8,
    c2: int | None = None,
    c3: int | None = None,
    k1: int = 3,
    k2: int = 3,
    k3: int = 3,             
    fc_hidden: int | None = None,
    use_bn: bool = False,
    dropout_p: float | None = None
) -> nn.Module:

    feats: list[nn.Module] = []

    # Block 1
    feats += [nn.Conv2d(1, c1, kernel_size=k1, padding=k1 // 2)]
    if use_bn:
        feats += [nn.BatchNorm2d(c1)]
    feats += [nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)]  

    # Block 2
    if c2 is not None:
        feats += [nn.Conv2d(c1, c2, kernel_size=k2, padding=k2 // 2)]
        if use_bn:
            feats += [nn.BatchNorm2d(c2)]
        feats += [nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)]
    else:
        c2 = c1  # in case we skip, use same channels for next block

    # Block 3 
    if c3 is not None:
        feats += [nn.Conv2d(c2, c3, kernel_size=k3, padding=k3 // 2)]
        if use_bn:
            feats += [nn.BatchNorm2d(c3)]
        feats += [nn.ReLU(inplace=True)]
    else:
        c3 = c2

    features = nn.Sequential(*feats)

    # Flattened size
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 28, 28)
        f = features(dummy)
        flat_dim = int(f.numel())

    # Classifier
    head: list[nn.Module] = [nn.Flatten()]
    if dropout_p:
        head += [nn.Dropout(dropout_p)]
    if fc_hidden:
        head += [nn.Linear(flat_dim, fc_hidden), nn.ReLU(inplace=True)]
        if dropout_p:
            head += [nn.Dropout(dropout_p)]
        head += [nn.Linear(fc_hidden, num_classes)]
    else:
        head += [nn.Linear(flat_dim, num_classes)]

    model = nn.Sequential(features, *head).to(device)
    return model



# Training loop for one epoch

criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, loader, optimizer):
    model.train()
    batch_losses, total, correct = [], 0, 0
    for x, y in loader:
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())
        with torch.no_grad():
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    avg_loss = sum(batch_losses) / len(batch_losses)
    train_acc = 100.0 * correct / total
    return avg_loss, train_acc

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, dtype=torch.long,    non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total

# Save model with metadata (as .pth file)

def save_model(model, save_dir, meta):
    os.makedirs(save_dir, exist_ok=True)
    
    conv_part = "-".join(str(meta.get(c)) for c in ["c1", "c2", "c3"] if meta.get(c) is not None)
    k_part    = "-".join(str(meta.get(k)) for k in ["k1", "k2", "k3"] if meta.get(k) is not None)

    name = (f"letters_c{conv_part}_k{k_part}"
            f"_bs{meta['batch_size']}_lr{meta['lr']}_ep{meta['epochs']}"
            f"_acc{meta['test_acc']:.2f}")

    path = os.path.join(save_dir, name + ".pth")

    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)

    with open(os.path.join(save_dir, name + ".json"), "w") as f:
        json.dump(meta, f, indent=2)

    return path



def run_experiment(train_set, test_set,
                   c1, c2, c3,
                   k1, k2, k3,
                   fc_hidden, lr, batch_size, epochs,
                   use_bn=False, dropout_p=None):
    train_loader, test_loader = get_loaders(train_set, test_set, batch_size)

   
    model = make_cnn(num_classes=26,
                     c1=c1, c2=c2, c3=c3,
                     k1=k1, k2=k2, k3=k3,
                     fc_hidden=fc_hidden,
                     use_bn=use_bn, dropout_p=dropout_p)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        loss, tr_acc = train_one_epoch(model, train_loader, optimizer)
        te_acc = evaluate(model, test_loader)
        epoch_losses.append(loss)
        print(f"  Ep {ep:02d}/{epochs} | loss={loss:.4f} | train_acc={tr_acc:.2f}% | test_acc={te_acc:.2f}%")
    dur = time.perf_counter() - t0

    test_acc = evaluate(model, test_loader)


    meta = dict(c1=c1, c2=c2, c3=c3,
                k1=k1, k2=k2, k3=k3,
                fc_hidden=fc_hidden,
                lr=lr, batch_size=batch_size, epochs=epochs,
                test_acc=test_acc, time_s=dur,
                device=str(device), torch=torch.__version__)

    model_path = save_model(model, "models", meta)

    # Loss plot 
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Avg training loss")
    title = (f"c1={c1}, c2={c2}, c3={c3}, "
             f"k1={k1}, k2={k2}, k3={k3}, "
             f"bs={batch_size}, lr={lr}, ep={epochs}")
    plt.title(title); plt.grid(alpha=0.3); plt.tight_layout()
    safe = title.replace(" ", "").replace(",", "_").replace("=", "")
    plt.savefig(f"results/TaskB_{safe}.png", dpi=140)
    plt.close()

    return meta | {"model_path": model_path}


if __name__ == "__main__":
    train_set, test_set = get_datasets()

    # Architecture + hyperparameter grid
    EXPERIMENTS = [
    # Similar to task A with c3 and k3
    dict(c1=8,  c2=None, c3=None, k1=3, k2=3, k3=3, fc_hidden=None, lr=1e-3, batch_size=128, epochs=10,  use_bn=False, dropout_p=None),

    #  Wider single-block
    dict(c1=32, c2=None, c3=None, k1=3, k2=3, k3=3, fc_hidden=None, lr=1e-3, batch_size=128, epochs=10,  use_bn=True,  dropout_p=0.2),

    # Deeper two-blocks
    dict(c1=16, c2=32, c3=16, k1=3, k2=3, k3=3, fc_hidden=None, lr=1e-3, batch_size=128, epochs=10,  use_bn=True,  dropout_p=0.2),

    # Larger first kernel
    dict(c1=16, c2=32, c3=32, k1=5, k2=3, k3=3, fc_hidden=None, lr=1e-3, batch_size=128, epochs=10,  use_bn=True,  dropout_p=0.3),

    # Add hidden FC (more capacity)
    dict(c1=16, c2=32, c3=16, k1=3, k2=3, k3=3, fc_hidden=128,  lr=1e-3, batch_size=128, epochs=10,  use_bn=True,  dropout_p=0.3),

    # Lower LR + more epochs to observe learning dynamics
    dict(c1=16, c2=32, c3=16, k1=3, k2=3, k3=3, fc_hidden=128, lr=5e-4, batch_size=128, epochs=20,  use_bn=True,  dropout_p=0.2),
    ]


    # CSV 
    os.makedirs("results", exist_ok=True)
    csv_path = "results/taskB_results.csv"
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fcsv:
        writer = None

    all_results = []
    best = None

    for exp in EXPERIMENTS:
        print("-"*80)
        print("Running:", exp)
        res = run_experiment(train_set, test_set, **exp)
        all_results.append(res)
        if best is None or res["test_acc"] > best["test_acc"]:
            best = res

        # append to CSV
        row = res.copy()
        with open(csv_path, "a", newline="") as fcsv:
            fieldnames = ["c1","c2", "c3", "k1","k2", "k3", "fc_hidden","lr","batch_size","epochs",
                          "test_acc","time_s","device","torch","model_path"]
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow({k: row.get(k) for k in fieldnames})

    print("\n=== BEST LETTERS CLASSIFIER ===")
    print(json.dumps(best, indent=2))
    print("Use this .pth in Task C.")
