import argparse

from datasets import get_train_dataloader, get_test_dataloader
from models import *
from trainer import train, test
from utils import seed_random


def main(dataset_str, model_name_str, batch_size, epochs, device):
    if dataset_str == 'mnist':
        train_dataloader = get_train_dataloader(batch_size=batch_size)
    else:
        raise NotImplementedError
    seed_random(0)

    if model_name_str == 'sparse_cnn':
        sparse_model = SparseCNN()
    elif model_name_str == 'hybrid_cnn':
        sparse_model = HybridCNN()
    else:
        raise NotImplementedError
    dense_model = DenseCNN()
    test_dataloaders = {noise_level: get_test_dataloader(batch_size=batch_size, eta=noise_level / 100) for noise_level
                        in range(0, 60, 5)}
    print('Training dense_cnn')
    dense_optimizer = torch.optim.SGD(dense_model.parameters(), lr=0.01, momentum=0.9)
    for i in range(epochs):
        train(dense_model, device, train_dataloader, dense_optimizer, i + 1)
        test(dense_model, device, test_dataloaders[0])
    print(f'Training {model_name_str}')
    sparse_optimizer = torch.optim.SGD(sparse_model.parameters(), lr=0.01, momentum=0.9)
    for i in range(epochs):
        train(sparse_model, device, train_dataloader, sparse_optimizer, i + 1)
        test(sparse_model, device, test_dataloaders[0])
    dense_acc = [test(dense_model, device, test_dataloaders[i])[1] for i in range(0, 60, 5)]
    sparse_acc = [test(sparse_model, device, test_dataloaders[i])[1] for i in range(0, 60, 5)]
    for noise_level, acc in zip(range(0, 60, 5), dense_acc):
        print(f'dense_cnn on noise level {noise_level}%, test accuracy: {acc}')
    for noise_level, acc in zip(range(0, 60, 5), sparse_acc):
        print(f'{model_name_str} on noise level {noise_level}%, test accuracy: {acc}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compare dense cnn and hybrid/sparse cnn\'s performance with noise')
    parser.add_argument('-d', dest='dataset', type=str, default='mnist', help='Name of the dataset')
    parser.add_argument('-m', dest='sparse_model_name', type=str,
                        default='hybrid_cnn', help='Name of the model to compare with (hybrid_cnn or sparse_cnn)')
    parser.add_argument('-b', dest='batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-e', dest='epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--no-gpu', dest='no_gpu', action='store_true', default=False, help='Disable GPU')
    args = parser.parse_args()
    if args.no_gpu:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args.dataset, args.sparse_model_name, args.batch_size, args.epochs, device)
