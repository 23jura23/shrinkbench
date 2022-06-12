import datetime
import os
import copy

import numpy as np
import torchvision
import torch
from torchvision.datasets.cifar import CIFAR10

from experiment import PruningExperiment, TrainingExperiment

datapath = 'datasets'
os.environ['DATAPATH'] = datapath
CIFAR10(os.path.join(datapath, "CIFAR10"), train=True, download=True)


def mag_criterion(mags, train_grads, val_grads):
    return np.abs(mags)


def train_criterion(mags, train_grads, val_grads):
    return np.abs(train_grads)


def mag_mul_train_criterion(mags, train_grads, val_grads):
    return np.abs(mags * train_grads)


def mag_add_train_criterion(mags, train_grads, val_grads):
    return np.abs(mags + train_grads)


def mag_sub_train_criterion(mags, train_grads, val_grads):
    return np.abs(mags - train_grads)


def fine_tune_experiment(model):
    model = copy.deepcopy(model)

    pretrained_params = [param for name, param in model.named_parameters() if 'fc' not in str(name)]
    optim = torch.optim.Adam([{'params': pretrained_params, 'lr': 1e-4}, {'params': model.fc.parameters(), 'lr': 1e-3}])

    train_experiment = TrainingExperiment(
        dataset='CIFAR10',
        model=model,
        dl_kwargs={'batch_size': 128, 'pin_memory': False, 'num_workers': 2},
        train_kwargs={'optim': optim, 'epochs': 10},
        save_freq=10
    )

    train_experiment.run()


def random_pruning_experiment(model, compression):
    model = copy.deepcopy(model)

    pretrained_params = [param for name, param in model.named_parameters() if 'fc' not in str(name)]
    optim = torch.optim.Adam([{'params': pretrained_params, 'lr': 1e-4}, {'params': model.fc.parameters(), 'lr': 1e-3}])

    experiment = PruningExperiment(
        dataset='CIFAR10',
        model=model,
        strategy="RandomPruning",
        strategy_name=f"random_{compression}",
        compression=compression,
        train_kwargs={'optim': optim, 'epochs': 10},
        pruning_dl_kwargs={'batch_size': 128, 'pin_memory': False, 'num_workers': 2},
        dl_kwargs={'batch_size': 128, 'pin_memory': False, 'num_workers': 2},
        save_freq=10
    )
    experiment.run()


def global_criterion_pruning_experiment(model, compression, criterion, criterion_name):
    model = copy.deepcopy(model)

    pretrained_params = [param for name, param in model.named_parameters() if 'fc' not in str(name)]
    optim = torch.optim.Adam([{'params': pretrained_params, 'lr': 1e-4}, {'params': model.fc.parameters(), 'lr': 1e-3}])

    experiment = PruningExperiment(
        dataset='CIFAR10',
        model=model,
        strategy="GlobalMagGradValF",
        strategy_name=criterion_name,
        compression=compression,
        train_kwargs={'optim': optim, 'epochs': 10},
        pruning_dl_kwargs={'batch_size': 128, 'pin_memory': False, 'num_workers': 2},
        save_pruning={
            'mags': f'mags_{criterion_name}_{compression}_{datetime.datetime.now()}',
            'train_grads': f'train_grads_{criterion_name}_{compression}_{datetime.datetime.now()}',
            'val_grads': f'val_grads_{criterion_name}_{compression}_{datetime.datetime.now()}',
            'importances': f'importances_{criterion_name}_{compression}_{datetime.datetime.now()}'
        },
        dl_kwargs={'batch_size': 128, 'pin_memory': False, 'num_workers': 2},
        strategy_kwargs={'F': criterion},
        save_freq=10
    )
    experiment.run()


if __name__ == '__main__':
    resnet34 = torchvision.models.resnet34(pretrained=True)
    resnet34.fc = torch.nn.Linear(512, 10)
    resnet34.fc.is_classifier = True

    fine_tune_experiment(resnet34)

    for compression in [4, 8, 16, 32]:
        random_pruning_experiment(resnet34, compression)

    for compression in [4, 8, 16, 32]:
        for criterion in [mag_criterion, train_criterion, mag_mul_train_criterion, mag_add_train_criterion, mag_sub_train_criterion]:
            global_criterion_pruning_experiment(resnet34, compression, criterion, criterion.__name__)
