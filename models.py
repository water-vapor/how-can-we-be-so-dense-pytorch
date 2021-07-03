import torch

from layers import SparseLinear, KWinner, SparseConv2D


class DenseMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_stack = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(784, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )

    def forward(self, inputs):
        return self.layers_stack(inputs)


class SparseMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_stack = torch.nn.Sequential(
            torch.nn.Flatten(),
            SparseLinear(784, 128),
            KWinner(k=40),
            SparseLinear(128, 64),
            KWinner(k=20),
            torch.nn.Linear(64, 10)
        )

    def forward(self, inputs):
        return self.layers_stack(inputs)


class DenseCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_stack = torch.nn.Sequential(
            torch.nn.Conv2d(1, 30, (3, 3)),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(5070, 150),
            torch.nn.ReLU(),
            torch.nn.Linear(150, 10)
        )

    def forward(self, inputs):
        return self.layers_stack(inputs)


class HybridCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_stack = torch.nn.Sequential(
            torch.nn.Conv2d(1, 30, (3, 3)),
            torch.nn.MaxPool2d(2, stride=2),
            KWinner(k=400),
            torch.nn.Flatten(),
            SparseLinear(5070, 150),
            torch.nn.ReLU(),
            torch.nn.Linear(150, 10)
        )

    def forward(self, inputs):
        return self.layers_stack(inputs)


class SparseCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_stack = torch.nn.Sequential(
            SparseConv2D(1, 30, (3, 3)),
            torch.nn.MaxPool2d(2, stride=2),
            KWinner(k=400),
            torch.nn.Flatten(),
            SparseLinear(5070, 150),
            KWinner(k=50),
            torch.nn.Linear(150, 10)
        )

    def forward(self, inputs):
        return self.layers_stack(inputs)
