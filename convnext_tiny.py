import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset
import timm
import gc
import csv

# ==========================================
# 0. Clean up previous memory
# ==========================================
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# ==========================================
# 1. Hyperparameters & DP Configuration
# ==========================================

# 🔒 Toggle Differential Privacy on/off
USE_DIFFERENTIAL_PRIVACY = True

TARGET_EPSILON = 8.0
TARGET_DELTA   = 1e-5
MAX_GRAD_NORM  = 1.0
EPOCHS         = 30

LEARNING_RATE           = 1e-3
VIRTUAL_BATCH_SIZE      = 8000 if USE_DIFFERENTIAL_PRIVACY else 256
MAX_PHYSICAL_BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

# ==========================================
# 2. Data Loading & Transforms
# ==========================================
class HFDatasetWrapper(Dataset):
    """Wraps Hugging Face datasets into PyTorch Datasets with image transforms."""
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image'].convert('RGB')
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label

print("Loading Tiny ImageNet...")
hf_train = load_dataset("zh-plus/tiny-imagenet", split="train")
hf_test  = load_dataset("zh-plus/tiny-imagenet", split="valid")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=16),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    normalize,
])

trainset = HFDatasetWrapper(hf_train, transform=train_transform)
testset  = HFDatasetWrapper(hf_test,  transform=test_transform)

trainloader = DataLoader(
    trainset,
    batch_size=VIRTUAL_BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
testloader = DataLoader(
    testset,
    batch_size=MAX_PHYSICAL_BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

# ==========================================
# 3. Model & Optimizer Initialization
# ==========================================
print("Initializing ConvNeXt-Small...")
model = timm.create_model('convnext_small', pretrained=True, num_classes=200)
model = model.to(DEVICE)

# Weight decay must be 0 for Differential Privacy; fine for non-DP too
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0 if USE_DIFFERENTIAL_PRIVACY else 1e-4,
)
criterion = nn.CrossEntropyLoss()

# ==========================================
# 4. Optionally Attach the Privacy Engine
# ==========================================
privacy_engine = None

if USE_DIFFERENTIAL_PRIVACY:
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager

    print("Configuring Opacus Privacy Engine...")
    privacy_engine = PrivacyEngine()

    model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        epochs=EPOCHS,
        target_epsilon=TARGET_EPSILON,
        target_delta=TARGET_DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )
    print(f"Using noise multiplier: {optimizer.noise_multiplier:.4f}")
else:
    print("⚠️  Differential Privacy DISABLED — training without privacy guarantees.")

# Cosine annealing LR scheduler — attach AFTER make_private_with_epsilon
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ==========================================
# 5. Training Loop
# ==========================================
best_val_acc = 0.0
sample_rate_q = VIRTUAL_BATCH_SIZE / len(trainset)
steps_T = int(EPOCHS / sample_rate_q)
sigma = float(getattr(optimizer, "noise_multiplier", 0.0)) if USE_DIFFERENTIAL_PRIVACY else 0.0

for epoch in range(EPOCHS):
    model.train()
    # Using a dictionary to hold mutable state across the inner function scope
    train_stats = {'loss': 0.0, 'correct': 0, 'total': 0}

    def run_train_epoch(data_loader):
        """Inner training loop, shared by both DP and non-DP paths."""
        for images, targets in data_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_stats['loss'] += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_stats['total']   += targets.size(0)
            train_stats['correct'] += predicted.eq(targets).sum().item()

    if USE_DIFFERENTIAL_PRIVACY:
        with BatchMemoryManager(
            data_loader=trainloader,
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
            optimizer=optimizer,
        ) as memory_safe_data_loader:
            run_train_epoch(memory_safe_data_loader)
    else:
        run_train_epoch(trainloader)

    train_acc = 100. * train_stats['correct'] / train_stats['total']
    scheduler.step()

    # ==========================================
    # Validation Loop
    # ==========================================
    model.eval()
    test_loss    = 0.0
    test_correct = 0
    test_total   = 0

    with torch.no_grad():
        for images, targets in testloader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            outputs = model(images)
            loss    = criterion(outputs, targets)

            test_loss    += loss.item() * images.size(0)
            _, predicted  = outputs.max(1)
            test_total   += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

    test_acc = 100. * test_correct / test_total

    if test_acc > best_val_acc:
        best_val_acc = test_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  ✅ New best model saved ({best_val_acc:.2f}%)")

    current_lr = scheduler.get_last_lr()[0]

    print(f"\nEnd of Epoch {epoch+1}:")
    print(f"  Train Acc : {train_acc:.2f}%")
    print(f"  Val Acc   : {test_acc:.2f}%  (Best: {best_val_acc:.2f}%)")
    if USE_DIFFERENTIAL_PRIVACY:
        epsilon = privacy_engine.get_epsilon(TARGET_DELTA)
        print(f"  Privacy   : ε = {epsilon:.4f}, δ = {TARGET_DELTA}")
    else:
        print(f"  Privacy   : Disabled")
    print(f"  LR        : {current_lr:.6f}")
    print("-" * 60)

print(f"\nTraining complete. Best Val Acc: {best_val_acc:.2f}%")

results_csv = "convnext_tiny_results.csv"
csv_fields = ["Epochs", "batch size", "sample rate (q)", "steps (T)", "best_val_acc", "sigma"]
csv_row = {
    "Epochs": EPOCHS,
    "batch size": VIRTUAL_BATCH_SIZE,
    "sample rate (q)": sample_rate_q,
    "steps (T)": steps_T,
    "best_val_acc": best_val_acc,
    "sigma": sigma,
}

with open(results_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    if f.tell() == 0:
        writer.writeheader()
    writer.writerow(csv_row)

print(f"Saved training summary to {results_csv}")
