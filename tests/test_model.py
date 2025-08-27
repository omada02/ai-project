import torch
from torchvision import datasets, transforms
from src.model import FashionMNISTModel


def test_fashion_mnist_accuracy():
    """Testa l'accuracy del modello FashionMNIST su dataset di test."""
    # Carica modello
    model = FashionMNISTModel()
    model.load_state_dict(
        torch.load("fashion_mnist_model.pt", map_location=torch.device('cpu'))
    )
    model.eval()

    # Dataset di test
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
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
    print(f'Accuracy sul test set: {accuracy:.2f}%')

    # Controllo minimo per passare il test CI
    assert accuracy > 50, "Accuracy troppo bassa sul test set"
