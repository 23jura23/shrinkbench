import copy

import numpy as np

from experiment import TrainingExperiment, PruningExperiment
from models import resnet20
import os

datapath = 'datasets'
os.environ['DATAPATH'] = datapath
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.cifar import CIFAR100

MNIST(os.path.join(datapath, "MNIST"), train=True, download=True)
CIFAR10(os.path.join(datapath, "CIFAR10"), train=True, download=True)
CIFAR100(os.path.join(datapath, "CIFAR100"), train=True, download=True)


def training_pruning(before_pruning_epoch: int, after_pruning_epoch: int, compression: int):
    model = resnet20(pretrained=False)

    def our_strategy(value, train_grad, val_grad):
        return np.abs(value)

    train_experiment = TrainingExperiment(
        dataset='CIFAR10',
        model=model,
        dl_kwargs={'batch_size': 128, 'pin_memory': False, 'num_workers': 2},
        train_kwargs={'epochs': before_pruning_epoch},
        save_freq=10
    )
    train_experiment.run()
    model.cpu()
    for strategy in ['RandomPruning', 'GlobalMagGradValF']:
        exp_our = PruningExperiment(dataset='CIFAR10',
                                    model=copy.deepcopy(model),
                                    strategy=strategy,
                                    strategy_name="Ours",
                                    compression=compression,
                                    train_kwargs={'epochs': after_pruning_epoch},
                                    pretrained=False,
                                    strategy_kwargs={'F': our_strategy})
        exp_our.run()


if __name__ == "__main__":
    training_pruning(10, 10, 2)
