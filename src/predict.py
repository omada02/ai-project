import sys
import torch
from PIL import Image
from torchvision import transforms
from src.model import FashionMNISTModel

# Dizionario delle classi FashionMNIST
classes = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

def predict(image_path):
    # Carica modello
    model = FashionMNISTModel()
    model.load_state_dict(torch.load("fashion_mnist_model.pt", map_location=torch.device('cpu')))
    model.eval()

    # Preprocessing immagine
    transform = transforms.ToTensor()
    image = Image.open(image_path).convert("L").resize((28, 28))
    image_tensor = transform(image).unsqueeze(0)

    # Predizione
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).item()

    return classes[prediction]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    result = predict(sys.argv[1])
    print(f"Predicted class: {result}")
