
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Check device for training
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# Matching Task B and C
def make_cnn(num_classes, c1=16, c2=32, c3=16,
             k1=3, k2=3, k3=3, fc_hidden=128,
             use_bn=True, dropout_p=0.3):
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
        dummy = torch.zeros(1, 1, 28, 28)
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

    # Flat head to match how checkpoints were saved in B/C
    return nn.Sequential(features, *head)

# checkpoint loader
def load_checkpoint(path, device):

    try:
        sd = torch.load(path, map_location=device, weights_only=True)
        if isinstance(sd, dict) and all(torch.is_tensor(v) for v in sd.values()):
            return {"state_dict": sd}
    except Exception:
        pass

    return torch.load(path, map_location=device, weights_only=False)

# Build and Load model
def load_model(model_path, num_classes):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = make_cnn(num_classes)
    ckpt = load_checkpoint(model_path, device)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    return model

# Preprocess 
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0).to(device)

# Prediction
@torch.no_grad()
def predict(image_path, letter_model, digit_model):
    x = preprocess_image(image_path)
    letter_logits = letter_model(x)
    digit_logits  = digit_model(x)

    letter_probs = torch.softmax(letter_logits, dim=1)
    digit_probs  = torch.softmax(digit_logits,  dim=1)

    li = int(letter_probs.argmax(1).item())
    di = int(digit_probs.argmax(1).item())

    letter_char = chr(li + 65)  
    return {
        'letter': {'class': letter_char, 'confidence': float(letter_probs[0, li])},
        'digit':  {'class': di,        'confidence': float(digit_probs[0, di])}
    }

# Evaluate image folder
def evaluate_custom_images(image_folder, letter_model_path, digit_model_path):
    letter_model = load_model(letter_model_path, num_classes=26)
    digit_model  = load_model(digit_model_path,  num_classes=10)

    image_files = [f for f in os.listdir(image_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in {image_folder}")
        return

    print(f"\n{'Image':<20} {'Letter Prediction':<22} {'Digit Prediction':<22}")
    print("-" * 70)

    results = []
    for fname in sorted(image_files):
        path = os.path.join(image_folder, fname)
        pred = predict(path, letter_model, digit_model)
        letter_str = f"{pred['letter']['class']} ({pred['letter']['confidence']:.1%})"
        digit_str  = f"{pred['digit']['class']} ({pred['digit']['confidence']:.1%})"
        print(f"{fname:<20} {letter_str:<22} {digit_str:<22}")
        results.append({'image_path': path, 'letter_pred': pred['letter'], 'digit_pred': pred['digit']})

    visualize_predictions(results)

# Visualize predictions
def visualize_predictions(results):
    plt.figure(figsize=(3.2 * len(results), 4.2))
    for i, r in enumerate(results):
        img = Image.open(r['image_path']).convert('L')
        plt.subplot(1, len(results), i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(
            f"{os.path.basename(r['image_path'])}\n"
            f"Letter: {r['letter_pred']['class']} ({r['letter_pred']['confidence']:.1%})\n"
            f"Digit:  {r['digit_pred']['class']} ({r['digit_pred']['confidence']:.1%})"
        )
        plt.axis('off')
    os.makedirs("results", exist_ok=True)
    out = 'results/custom_image_results.png'
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.show()
    print(f"\nSaved visualization to: {out}")


if __name__ == "__main__":
   
    LETTER_MODEL_PATH = "models/letters_c16-32-16_k3-3-3_bs128_lr0.001_ep10_acc92.82.pth"
    DIGIT_MODEL_PATH  = "models_transferred/digits_model.pth"                               

    CUSTOM_IMAGES_FOLDER = "./test_images"  
    evaluate_custom_images(CUSTOM_IMAGES_FOLDER, LETTER_MODEL_PATH, DIGIT_MODEL_PATH)
