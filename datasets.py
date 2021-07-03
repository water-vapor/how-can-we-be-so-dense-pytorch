import torch
from torchvision import datasets, transforms

from ops import add_noise


def transform_add_noise(eta):
    if eta == 0.:
        return transforms.ToTensor()
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: add_noise(x, eta))
        ])


def get_train_dataloader(batch_size=64):
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform_add_noise(0.))
    return torch.utils.data.DataLoader(mnist_train, batch_size=batch_size)


def get_test_dataloader(eta=0., batch_size=64):
    mnist_test_noise = datasets.MNIST('./data', train=False, transform=transform_add_noise(eta))
    return torch.utils.data.DataLoader(mnist_test_noise, batch_size=batch_size)
