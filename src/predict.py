import sys

import torch
from PIL import Image
from torchvision import transforms

from src.model import FashionMNISTModel


def predict(image_path):
    model = FashionMNISTModel()
    model.load_state_dict(torch.load("fashion_mnist_model.pt"))
    model.eval()

    transform = transforms.ToTensor()
    image = Image.open(image_path).convert("L").resize((28, 28))
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).item()

    return prediction


if __name__ == "__main__":
    print(predict(sys.argv[1]))

