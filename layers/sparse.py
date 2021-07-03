import torch

from ops import random_binary_mask


class SparseLinear(torch.nn.Linear):
    def __init__(self, *args, sparsity=0.5, strict_sparsity=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparsity = sparsity
        self.strict_sparsity = strict_sparsity
        self.binary_mask = random_binary_mask(
            self.weight.shape, self.sparsity, self.strict_sparsity
        )

    def forward(self, inputs):
        self.weight.data = torch.where(self.binary_mask, self.weight, torch.tensor(0.))
        return super().forward(inputs)


class SparseConv2D(torch.nn.Conv2d):
    def __init__(self, *args, sparsity=0.5, strict_sparsity=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparsity = sparsity
        self.strict_sparsity = strict_sparsity
        self.binary_mask = random_binary_mask(
            self.weight.shape, self.sparsity, self.strict_sparsity
        )

    def forward(self, inputs):
        self.weight.data = torch.where(self.binary_mask, self.weight, torch.tensor(0.))
        return super().forward(inputs)
