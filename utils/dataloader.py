import numpy as np
import torch

from utils.transform import ToFloatTensor, bacteria_train_transform, bacteria_valid_transform
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split as tts
from torchvision import transforms
import torch.nn.functional as F

train_transform = transforms.Compose([ToFloatTensor()])
valid_transform = transforms.Compose([ToFloatTensor()])


class myDataset(Dataset):
    """create dataset"""

    def __init__(self, X, y, transform=None, pool_dim=None):
        self.data = X
        self.transform = transform
        self.label = torch.FloatTensor(y) if y.ndim == 2 else torch.LongTensor(y)
        self.pool_dim = pool_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        X = self.data[item, :]
        X = self.transform(X)
        y = self.label[item]
        if self.pool_dim:
            X = F.adaptive_avg_pool1d(X, self.pool_dim)
        return X, y


def make_trainloader(ds, batch_size=16, num_workers=12, train_size=0.8, seed=42, tune=False, pool_dim=None):

    if tune:
        data_x = np.load(f'datasets/{ds}/tune_x.npy')
        data_y = np.load(f'datasets/{ds}/tune_y.npy')
    else:
        data_x = np.load(f'datasets/{ds}/train_x.npy')
        data_y = np.load(f'datasets/{ds}/train_y.npy')

    ids = np.arange(len(data_y))

    transform_train = bacteria_train_transform if ds == 'Bacteria' else train_transform
    transform_valid = bacteria_valid_transform if ds == 'Bacteria' else valid_transform

    stratify = None if ds == 'FunctionalGroups' else data_y

    if train_size:
        train_id, val_id = tts(ids, shuffle=True, train_size=train_size, random_state=seed, stratify=stratify)
        trainset = myDataset(data_x[train_id], data_y[train_id], transform=transform_train, pool_dim=pool_dim)
        valset = myDataset(data_x[val_id], data_y[val_id], transform=transform_valid, pool_dim=pool_dim)

    else:
        test_x = np.load(f'datasets/{ds}/test_x.npy')
        test_y = np.load(f'datasets/{ds}/test_y.npy')

        trainset = myDataset(data_x, data_y, transform=transform_train, pool_dim=pool_dim)
        valset = myDataset(test_x, test_y, transform=transform_valid, pool_dim=pool_dim)

    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    valloader = DataLoader(valset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    return trainloader, valloader


def make_testloader(ds, batch_size=128, num_workers=12, pool_dim=256):
    data_x = np.load(f'datasets/{ds}/test_x.npy')
    data_y = np.load(f'datasets/{ds}/test_y.npy')

    transform_valid = bacteria_valid_transform if ds == 'Bacteria' else valid_transform

    testset = myDataset(data_x, data_y, transform=transform_valid, pool_dim=pool_dim)
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

