import pandas as pd
import os
import torch
import sys
from torchvision.transforms import transforms
from utilities.Data import CelebaDataset
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

torch.manual_seed(42)


def path_check(path):
    if os.path.exists(path):
        return False
    else:
        return True


def load_dataset(dataset, root_folder):

    data_path = os.path.join(root_folder, dataset)

    if dataset =='mnist':
        train_size = 50000
        val_size = 10000
        test_size = 10000

        download = path_check(data_path)

        mndata = MNIST(root=data_path, train=True, transform=transforms.ToTensor(), download=download)
        train_set, val_set = torch.utils.data.random_split(mndata, [train_size, val_size])
        test_set = MNIST(root=data_path, train=False, transform=transforms.ToTensor(), download=download)

    elif dataset == 'celeba':
        if not ((os.path.exists(os.path.join(data_path, 'celeba-train.csv'))) &
                (os.path.exists(os.path.join(data_path, 'celeba-valid.csv'))) &
                (os.path.exists(os.path.join(data_path, 'celeba-test.csv')))):

            df = pd.read_csv(os.path.join(data_path, 'list_eval_partition.csv'))
            df.columns = ['Filename', 'Partition']
            df = df.set_index('Filename')

            df.loc[df['Partition'] == 0].to_csv(os.path.join(data_path, 'celeba-train.csv'))
            df.loc[df['Partition'] == 1].to_csv(os.path.join(data_path, 'celeba-valid.csv'))
            df.loc[df['Partition'] == 2].to_csv(os.path.join(data_path, 'celeba-test.csv'))

        transform = transforms.Compose([transforms.CenterCrop((140, 140)),
                                        transforms.Resize((64, 64)), transforms.ToTensor(), ])

        train_set = CelebaDataset(csv_path=os.path.join(data_path, 'celeba-train.csv'),
                                  img_dir=os.path.join(data_path, 'img_align_celeba'), transform=transform)

        val_set = CelebaDataset(csv_path=os.path.join(data_path, 'celeba-valid.csv'),
                                img_dir=os.path.join(data_path, 'img_align_celeba'), transform=transform)

        test_set = CelebaDataset(csv_path=os.path.join(data_path, 'celeba-test.csv'),
                                 img_dir=os.path.join(data_path, 'img_align_celeba'), transform=transform)

    elif dataset == 'cifar10':
        download = path_check(data_path)
        cidata = CIFAR10(root=data_path, train=True, transform=transforms.ToTensor(), download=download)
        test_set = CIFAR10(root=data_path, train=False, transform=transforms.ToTensor(), download=download)

        val_size = 5000
        train_size = len(cidata) - val_size
        train_set, val_set = torch.utils.data.random_split(cidata, [train_size, val_size])
    else:
        sys.exit('Dataset not available')
    return train_set, val_set, test_set


def create_dataloaders(dataset, root_folder, batch_size, num_workers):
    train_set, val_set, test_set = load_dataset(dataset, root_folder)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, pin_memory=True,
                                   num_workers=num_workers, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, pin_memory=True,
                                 num_workers=num_workers, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, pin_memory=True,
                                  num_workers=num_workers, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader


def create_test_set_shuffle(dataset, root_folder, batch_size, num_workers):
    _, _, test_set = load_dataset(dataset, root_folder)
    test_loader_shuffle = DataLoader(dataset=test_set, batch_size=batch_size, pin_memory=True,
                                  num_workers=num_workers, shuffle=True, drop_last=True)
    return test_loader_shuffle







