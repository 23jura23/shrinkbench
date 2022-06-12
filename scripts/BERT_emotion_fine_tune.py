import datetime
import os
import torch
import numpy as np
import copy

from transformers import AutoModelForSequenceClassification

from experiment import PruningExperiment, TrainingExperiment

datapath = 'datasets'
os.environ['DATAPATH'] = datapath


def delete_dropout(model):
    for name, child in model.named_children():
        if type(child) == torch.nn.Dropout:
            setattr(model, name, torch.nn.Identity())
        else:
            delete_dropout(child)


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

    optim = torch.optim.Adam(model.parameters(), lr=2e-5)

    train_experiment = TrainingExperiment(
        dataset='emotion',
        model=model,
        is_multiple_input_model=True,
        dl_kwargs={'batch_size': 64, 'pin_memory': False, 'num_workers': 2},
        train_kwargs={'optim': optim, 'epochs': 10},
        save_freq=10
    )

    train_experiment.run()


def random_pruning_experiment(model, compression):
    model = copy.deepcopy(model)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    experiment = PruningExperiment(
        dataset='emotion',
        model=model,
        strategy="RandomPruning",
        strategy_name=f"random_{compression}",
        compression=compression,
        is_multiple_input_model=True,
        train_kwargs={'optim': optim, 'epochs': 10},
        pruning_dl_kwargs={'batch_size': 128, 'pin_memory': False, 'num_workers': 2},
        dl_kwargs={'batch_size': 64, 'pin_memory': False, 'num_workers': 2},
        save_freq=10
    )
    experiment.run()


def global_criterion_pruning_experiment(model, compression, criterion, criterion_name):
    model = copy.deepcopy(model)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    experiment = PruningExperiment(
        dataset='emotion',
        model=model,
        strategy="GlobalMagGradValF",
        strategy_name=criterion_name,
        compression=compression,
        is_multiple_input_model=True,
        train_kwargs={'optim': optim, 'epochs': 10},
        pruning_dl_kwargs={'batch_size': 128, 'pin_memory': False, 'num_workers': 2},
        save_pruning={
            'mags': f'mags_{criterion_name}_{compression}_{datetime.datetime.now()}',
            'train_grads': f'train_grads_{criterion_name}_{compression}_{datetime.datetime.now()}',
            'val_grads': f'val_grads_{criterion_name}_{compression}_{datetime.datetime.now()}',
            'importances': f'importances_{criterion_name}_{compression}_{datetime.datetime.now()}'
        },
        dl_kwargs={'batch_size': 64, 'pin_memory': False, 'num_workers': 2},
        strategy_kwargs={'F': criterion},
        save_freq=10
    )
    experiment.run()


if __name__ == '__main__':
    bert = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

    fine_tune_experiment(bert)

    delete_dropout(bert)
    bert.classifier.is_classifier = True

    for compression in [2, 4]:
        random_pruning_experiment(bert, compression)

    for compression in [2, 4]:
        for criterion in [mag_criterion, train_criterion, mag_mul_train_criterion, mag_add_train_criterion, mag_sub_train_criterion]:
            global_criterion_pruning_experiment(bert, compression, criterion, criterion.__name__)

