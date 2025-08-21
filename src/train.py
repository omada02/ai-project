import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from src.model import FashionMNISTModel


def train_model():
    transform = transforms.ToTensor()
    train_data = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = FashionMNISTModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(5):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "fashion_mnist_model.pt")


if __name__ == "__main__":
    train_model()

