import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import os

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tf


def get_datasets(args):
    if args.normalize:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    data_path = os.path.join(args.data_path, 'CIFAR10')

    train = datasets.CIFAR10(
        data_path,
        train=True,
        download=True,
        transform=transform,
    )
    test = datasets.CIFAR10(
        data_path,
        train=False,
        download=True,
        transform=transforms.ToTensor())

    return train, test


def get_catsets(args):
    if args.normalize:
        trainset = ImageFolder(os.path.join(args.data_path+'catsdogs', 'train'),
                               transform=tf.Compose([tf.Resize(256),
                                                     tf.CenterCrop(224),
                                                     tf.ToTensor()]))
        testset = ImageFolder(os.path.join(args.data_path+'catsdogs', 'val'),
                             transform=tf.Compose([tf.Resize(256),
                                                   tf.CenterCrop(224),
                                                   tf.ToTensor()]))
    else:
        trainset = datasets.ImageFolder(os.path.join(args.data_path+'catsdogs', 'train'),
                                 transform=tf.Compose([tf.Resize(256),
                                                       tf.CenterCrop(224),
                                                       tf.ToTensor(),
                                                       tf.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])]))
        testset = datasets.ImageFolder(os.path.join(args.data_path+'catsdogs', 'val'),
                                transform=tf.Compose([tf.Resize(256),
                                                       tf.CenterCrop(224),
                                                       tf.ToTensor(),
                                                       tf.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])]))

    return trainset, testset

# FIXME redundant
def get_catloaders(args):
    dataset_train, dataset_test = get_catsets(args)

    train_size, val_size = args.split_sizes
    lengths = [int(len(dataset_train)*train_size),
               int(len(dataset_train)*val_size)]

    train_data, val_data = random_split(
        dataset_train, lengths, torch.Generator().manual_seed(args.seed))

    loader_args = dict(batch_size=args.batch_size,
                       num_workers=args.workers, drop_last=True)

    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    test_loader = DataLoader(dataset_test, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader


def get_dataloaders(args):
    dataset_train, dataset_test = get_datasets(args)

    train_size, val_size = args.split_sizes
    lengths = [int(len(dataset_train)*train_size),
               int(len(dataset_train)*val_size)]

    train_data, val_data = random_split(
        dataset_train, lengths, torch.Generator().manual_seed(args.seed))

    loader_args = dict(batch_size=args.batch_size,
                       num_workers=args.workers, drop_last=True)

    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    test_loader = DataLoader(dataset_test, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader
