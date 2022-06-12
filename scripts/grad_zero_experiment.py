from shrinkbench.experiment import PruningExperiment

import os

datapath = 'datasets'
os.environ['DATAPATH'] = datapath

from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.cifar import CIFAR100

MNIST(os.path.join(datapath, "MNIST"), train=True, download=True)
CIFAR10(os.path.join(datapath, "CIFAR10"), train=True, download=True)
CIFAR100(os.path.join(datapath, "CIFAR100"), train=True, download=True)

num_epochs = 3

for model, dataset in [("vgg_bn_drop", "CIFAR10"), ("vgg_bn_drop_100", "CIFAR100"), ("resnet20", "CIFAR10")]:
    for c in [2, 4, 8, 16, 32]:
        # random
        exp_rand = PruningExperiment(dataset=dataset,
                                     model=model,
                                     strategy="RandomPruning",
                                     strategy_name=f"RandomPruning",
                                     compression=c,
                                     train_kwargs={'epochs': num_epochs})
        exp_rand.run()

        # # abs(mag)
        exp_mag = PruningExperiment(dataset=dataset,
                                    model=model,
                                    strategy="GlobalMagWeight",
                                    strategy_name=f"GlobalMagWeight",
                                    compression=c,
                                    train_kwargs={'epochs': num_epochs})
        exp_mag.run()

        # # abs(mag * grad)
        exp_mag_grad = PruningExperiment(dataset=dataset,
                                         model=model,
                                         strategy="GlobalMagGrad",
                                         strategy_name=f"GlobalMagGrad",
                                         compression=c,
                                         train_kwargs={'epochs': num_epochs})
        exp_mag_grad.run()

        # # abs(mag + grad + val_grad)
        exp_sum = PruningExperiment(dataset=dataset,
                                    model=model,
                                    strategy="GlobalMagGradValSum",
                                    strategy_name=f"Sum",
                                    compression=c,
                                    strategy_kwargs={"beta": 1.0, "gamma": 1.0, "delta": 1.0},
                                    train_kwargs={'epochs': num_epochs})
        exp_sum.run()

        # # abs((mag * grad)/(val_grad))
        exp_inv_prod = PruningExperiment(dataset=dataset,
                                         model=model,
                                         strategy="GlobalMagGradValProd",
                                         strategy_name=f"Product, inverse val_grad",
                                         compression=c,
                                         strategy_kwargs={"beta": 1.0, "gamma": 1.0, "delta": -1.0, "eps": 1e-6},
                                         train_kwargs={'epochs': num_epochs})
        exp_inv_prod.run()

        # original Baley idea: https://www.aaai.org/AAAI22Papers/UC-00015-BelayK.pdf
        # ban |value| < beta && |grad| < gamma, with no restrictions on val_grad; beta and gamma are computed based on compression
        exp_inv_prod = PruningExperiment(dataset=dataset,
                                         model=model,
                                         strategy="GlobalMagGradTopVal",
                                         strategy_name=f"Baley",
                                         compression=c,
                                         strategy_kwargs={"use_val_grad": False, "have_common_threshold": True},
                                         train_kwargs={'epochs': num_epochs})
        exp_inv_prod.run()

        # original Baley idea: https://www.aaai.org/AAAI22Papers/UC-00015-BelayK.pdf
        # ban |value| < beta && |grad| < gamma, |val_grad| < delta; beta, gamma and delta are computed based on compression
        # delta is big so small deltas are banned and big are not
        for val_grad_fraction in [0.9]:  # , 0.95, 0.99]:
            exp_inv_prod = PruningExperiment(dataset=dataset,
                                             model=model,
                                             strategy="GlobalMagGradTopVal",
                                             strategy_name=f"Baley + keep top val_grad {val_grad_fraction}",
                                             compression=c,
                                             strategy_kwargs={"use_val_grad": True, "have_common_threshold": True,
                                                              "val_grad_fraction": val_grad_fraction,
                                                              "val_grad_and_or": "and"},
                                             train_kwargs={'epochs': num_epochs})
            exp_inv_prod.run()
