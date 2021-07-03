import argparse

from datasets import get_test_dataloader, get_train_dataloader
from models import *
from trainer import train, test
from utils import seed_random


def main(dataset_str, model_name_str, batch_size, epochs, device):
    if dataset_str == 'mnist':
        train_dataloader = get_train_dataloader(batch_size=batch_size)
        test_dataloader = get_test_dataloader(batch_size=batch_size)
    else:
        raise NotImplementedError
    seed_random(0)

    if model_name_str == 'sparse_cnn':
        model = SparseCNN()
    elif model_name_str == 'dense_cnn':
        model = DenseCNN()
    elif model_name_str == 'hybrid_cnn':
        model = HybridCNN()
    elif model_name_str == 'sparse_mlp':
        model = SparseMLP()
    elif model_name_str == 'dense_mlp':
        model = DenseMLP()
    else:
        raise NotImplementedError

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for i in range(epochs):
        train(model, device, train_dataloader, optimizer, i + 1)
        test(model, device, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train a single model without noise')
    parser.add_argument('-d', dest='dataset', type=str, default='mnist', help='Name of the dataset')
    parser.add_argument('-m', dest='model_name', type=str, default='sparse_cnn', help='Name of the model')
    parser.add_argument('-b', dest='batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-e', dest='epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--no-gpu', dest='no_gpu', action='store_true', default=False, help='Disable GPU')
    args = parser.parse_args()
    if args.no_gpu:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args.dataset, args.model_name, args.batch_size, args.epochs, device)
