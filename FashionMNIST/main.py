import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
config = {
    "batch_size": 64,
    "epochs": 15,
    "learning_rate": 1e-3,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# Dataset loading
transform = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

def load_dataset():
    train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    return train_data, test_data

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

# Training function
def train_one_epoch(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for X, y in tqdm(dataloader, desc="Training", leave=False):
        X, y = X.to(config["device"]), y.to(config["device"])

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(config["device"]), y.to(config["device"])
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    accuracy = correct / len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss

# Training loop
def train_model():
    train_data, test_data = load_dataset()
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=False)

    model = CNN().to(config["device"])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_losses, test_losses, test_accuracies = [], [], []

    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer)
        test_acc, test_loss = evaluate(test_loader, model, loss_fn)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {test_acc*100:.2f}%")

    torch.save(model.state_dict(), "cnn_fashionmnist.pth")
    print("Model saved as cnn_fashionmnist.pth")

    print(train_losses)
    print(test_losses)
    print(test_accuracies)

    plot_metrics(train_losses, test_losses, test_accuracies)

# Visualization
def plot_metrics(train_losses, test_losses, test_accuracies):
    epochs_range = range(1, config["epochs"] + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, test_losses, label="Test Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [a*100 for a in test_accuracies], label="Test Accuracy")
    plt.title("Test Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_model()
