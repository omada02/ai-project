from torchvision.datasets import FashionMNIST
from torchvision import transforms
import os

# Trasformazione base (MNIST è già 28x28, scala di grigi)
transform = transforms.ToTensor()

# Scarica dataset di test
test_dataset = FashionMNIST(root="./data", train=False, download=True, transform=transform)

# Cartella dove salvare le immagini
os.makedirs("fashionmnist_examples", exist_ok=True)

# Salva le prime 10 immagini di test
for i in range(10):
    img_tensor, label = test_dataset[i]
    # Trasforma tensore in immagine PIL
    img = transforms.ToPILImage()(img_tensor)
    # Salva immagine con nome che include l'etichetta
    img.save(f"fashionmnist_examples/{i}_label{label}.png")
    print(f"Salvata image {i} con label {label}")
