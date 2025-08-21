import torch

from src.model import FashionMNISTModel


def test_model_output_shape():
    model = FashionMNISTModel()
    sample_input = torch.randn(1, 1, 28, 28)
    output = model(sample_input)
    assert output.shape == (1, 10)
