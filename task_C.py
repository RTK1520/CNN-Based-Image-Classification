
import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

def get_mnist_datasets():
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train = datasets.MNIST(root='./data', train=True,  download=True, transform=tfm)
    test  = datasets.MNIST(root='./data', train=False, download=True, transform=tfm)
    return train, test

#  Returning a flat head to match Task B's saved checkpoint
def make_cnn(num_classes, c1=16, c2=32, c3=16, k1=3, k2=3, k3=3,
             fc_hidden=128, use_bn=True, dropout_p=0.3):
    feats = [
        nn.Conv2d(1, c1, kernel_size=k1, padding=k1 // 2),
        nn.BatchNorm2d(c1) if use_bn else nn.Identity(),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(c1, c2, kernel_size=k2, padding=k2 // 2),
        nn.BatchNorm2d(c2) if use_bn else nn.Identity(),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(c2, c3, kernel_size=k3, padding=k3 // 2),
        nn.BatchNorm2d(c3) if use_bn else nn.Identity(),
        nn.ReLU(inplace=True)
    ]
    features = nn.Sequential(*feats)

    with torch.no_grad():
        dummy = torch.zeros(1,1,28,28)
        f = features(dummy)
        flat_dim = int(f.numel())

    head = [nn.Flatten()]
    if dropout_p:
        head.append(nn.Dropout(dropout_p))
    if fc_hidden:
        head.extend([
            nn.Linear(flat_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p) if dropout_p else nn.Identity(),
            nn.Linear(fc_hidden, num_classes)
        ])
    else:
        head.append(nn.Linear(flat_dim, num_classes))

    return nn.Sequential(features, *head)

def main():
    train_data, test_data = get_mnist_datasets()
    batch_size = 128
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

    model_path = "models/letters_c16-32-16_k3-3-3_bs128_lr0.001_ep10_acc92.82.pth"

    
    model = make_cnn(num_classes=26, c1=16, c2=32, c3=16,
                     k1=3, k2=3, k3=3, fc_hidden=128, use_bn=True, dropout_p=0.3)

    
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])  
    model.to(device)

    # Finding classifier boundaries in the flat Sequential
    linear_indices = [i for i, m in enumerate(model) if isinstance(m, nn.Linear)]
    assert len(linear_indices) >= 1, "No Linear layers found in head."
    first_linear_idx = linear_indices[0]
    last_linear_idx  = linear_indices[-1]

    # Freeze feature extractor
    for i, m in enumerate(model):
        trainable = (i >= first_linear_idx)
        for p in m.parameters(recurse=True):
            p.requires_grad = trainable


        if i < first_linear_idx and isinstance(m, nn.BatchNorm2d):
            m.eval()

    # Replacing the final Linear to output 10 classes (digits)
    in_features = model[last_linear_idx].in_features
    model[last_linear_idx] = nn.Linear(in_features, 10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)

    def train_epoch(model, loader):
        model.train()
        # Ensure frozen part's BNs stay eval:
        for i, m in enumerate(model):
            if i < first_linear_idx and isinstance(m, nn.BatchNorm2d):
                m.eval()

        running_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        return running_loss/len(loader), 100.0*correct/total

    def evaluate(model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        return 100.0*correct/total

    print("Starting transfer learning...")
    epochs = 10
    t0 = time.time()
    for ep in range(1, epochs+1):
        loss, tr_acc = train_epoch(model, train_loader)
        te_acc = evaluate(model, test_loader)
        print(f"Epoch {ep:02d}/{epochs} | loss={loss:.4f} | train_acc={tr_acc:.2f}% | test_acc={te_acc:.2f}%")
    train_time = time.time() - t0
    final_acc = evaluate(model, test_loader)

    os.makedirs("models_transferred", exist_ok=True)
    save_path = "models_transferred/digits_model.pth"
    torch.save({
        "state_dict": model.state_dict(),
        "test_accuracy": final_acc,
        "training_time": train_time,
        "original_model": model_path,
        "architecture": {
            "c1":16,"c2":32,"c3":16,"k1":3,"k2":3,"k3":3,
            "fc_hidden":128,"use_bn":True,"dropout_p":0.3
        }
    }, save_path)

    print(f"\nTransfer Learning Results:")
    print(f"Training Time: {train_time:.2f} s")
    print(f"Test Accuracy: {final_acc:.2f}%")
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    main()
