import torch
from torchvision import datasets, transforms
from src.model import FashionMNISTModel
from pathlib import Path
import pytest  # opzionale, serve solo per evidenziare test compatibili

def main():
    model_path = "fashion_mnist_model.pt"

    # Controlla che il file esista
    if not Path(model_path).exists():
        print(f"File modello '{model_path}' non trovato!")
        return

    # Carica modello
    model = FashionMNISTModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Dataset di test
    test_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )

    # Calcola accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    print(f"Accuracy sul test set: {accuracy:.2f}%")
    return accuracy  # utile per test


# Funzione di test per GitHub Actions
def test_model_runs():
    """Verifica che il modello possa fare inferenza su un batch senza errori"""
    model = FashionMNISTModel()
    model.eval()

    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)
    images, labels = next(iter(test_loader))
    outputs = model(images)
    
    # Controllo minimo
    assert outputs.shape[0] == labels.shape[0]

    
if __name__ == "__main__":
    main()
