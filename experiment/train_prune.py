import json
import time
from collections import OrderedDict

import numpy as np
import torch
import copy

from . import TrainingExperiment

from .. import strategies
from ..metrics import model_size, flops
from ..util import printc

# TODO
def regrow(module, param_name, param, regrow_mask):
    with torch.no_grad():
        print(np.sum(regrow_mask))
        new_param = copy.deepcopy(param)
        new_param[regrow_mask] = 1.0
        getattr(module, param_name).data.copy_(torch.from_numpy(new_param).float())


class TrainingPruningExperiment(TrainingExperiment):

    def __init__(self,
                 dataset,
                 model,
                 initial_strategy,
                 strategy,
                 initial_compression,
                 compression,
                 strategy_name=None,
                 initial_strategy_name=None,
                 seed=42,
                 path=None,
                 dl_kwargs=dict(),
                 train_kwargs=dict(),
                 initial_strategy_kwargs=dict(),
                 strategy_kwargs=dict(),
                 debug=False,
                 pretrained=True,
                 resume=None,
                 resume_optim=False,
                 save_freq=10):
        super(TrainingPruningExperiment, self).__init__(dataset, model, seed, path, dl_kwargs, train_kwargs, debug, pretrained, resume, resume_optim, save_freq)

        if strategy_name is None:
            strategy_name = strategy
        if initial_strategy_name is None:
            initial_strategy_name = initial_strategy_name

        self.strategy = strategy
        self.compression = compression
        self.strategy_kwargs = strategy_kwargs

        self.add_params(strategy=strategy, compression=compression, strategy_name=strategy_name,
                        initial_strategy=initial_strategy, initial_compression=initial_compression, initial_strategy_name=initial_strategy_name)

        self.apply_pruning(initial_strategy, initial_compression, initial_strategy_kwargs)

        self.path = path
        self.save_freq = save_freq

    def apply_pruning(self, strategy, compression, strategy_kwargs):
        constructor = getattr(strategies, strategy)
        train_x, train_y = next(iter(self.train_dl))
        val_x, val_y = next(iter(self.val_dl))
        strategy_kwargs['val_x'] = val_x
        strategy_kwargs['val_y'] = val_y
        self.pruning = constructor(self.model, train_x, train_y, compression=compression, **strategy_kwargs)
        _, masked_parameters = self.pruning.apply()
        printc("Masked model", color='GREEN')
        return masked_parameters

    def run(self):
        self.freeze()
        printc(f"Running {repr(self)}", color='YELLOW')
        self.to_device()
        self.build_logging(self.train_metrics, self.path)

        self.save_metrics()

        if self.pruning.compression > 1:
            self.run_epochs()

    def run_epochs(self):
        """
        It was copied from TrainingExperiment, except the ### {code} #### blocks below
        """
        since = time.time()
        try:
            for epoch in range(self.epochs):
                printc(f"Start epoch {epoch}", color='YELLOW')
                #####
                if epoch > 0:
                    masked_parameters = self.apply_pruning(self.strategy, self.compression, self.strategy_kwargs)
                    prunable_params = OrderedDict({k: OrderedDict(v) for k, v in self.pruning.params().items()})
                    all_params = [
                        params.reshape((-1,)) for mod_params in prunable_params.values() for params in mod_params.values()
                    ]
                    all_params = np.concatenate(all_params)
                    regrowed_idx = np.random.choice(np.where(all_params == 0.0)[0], masked_parameters, replace=False)

                    regrowed_mask = np.zeros_like(all_params, dtype=np.int)
                    regrowed_mask[regrowed_idx] = 1

                    start = 0
                    for module, mod_params in prunable_params.items():
                        regrow_masks = {}
                        for param_name, param in mod_params.items():
                            param_length = param.size
                            regrow_mask = regrowed_mask[start:(start + param_length)].reshape(param.shape)
                            regrow(module, param_name, param, regrow_mask)
                            regrow_masks[param_name] = regrow_mask
                            start += param_length

                        def _calc_new_mask(old_mask, new_mask):
                            old_mask = old_mask.detach().cpu().numpy()
                            assert not np.any(old_mask * new_mask)
                            return np.float32((old_mask + new_mask))

                        module.set_masks(**{f'{k}_mask': _calc_new_mask(getattr(module, f'{k}_mask'), v) for k, v in
                                            regrow_masks.items()})
                #####
                self.train(epoch)
                self.eval(epoch)
                # Checkpoint epochs
                # TODO Model checkpointing based on best val loss/acc
                if epoch % self.save_freq == 0:
                    self.checkpoint()
                # TODO Early stopping
                # TODO ReduceLR on plateau?
                self.log(timestamp=time.time()-since)
                self.log_epoch(epoch)


        except KeyboardInterrupt:
            printc(f"\nInterrupted at epoch {epoch}. Tearing Down", color='RED')

    def save_metrics(self):
        self.metrics = self.pruning_metrics()
        with open(self.path / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        printc(json.dumps(self.metrics, indent=4), color='GRASS')
        summary = self.pruning.summary()
        summary_path = self.path / 'masks_summary.csv'
        summary.to_csv(summary_path)
        print(summary)

    def pruning_metrics(self):

        metrics = {}
        # Model Size
        size, size_nz = model_size(self.model)
        metrics['size'] = size
        metrics['size_nz'] = size_nz
        metrics['compression_ratio'] = size / size_nz

        x, y = next(iter(self.val_dl))
        x, y = x.to(self.device), y.to(self.device)

        # FLOPS
        ops, ops_nz = flops(self.model, x)
        metrics['flops'] = ops
        metrics['flops_nz'] = ops_nz
        metrics['theoretical_speedup'] = ops / ops_nz

        # Accuracy
        loss, acc1, acc5 = self.run_epoch(False, -1)
        self.log_epoch(-1)

        metrics['loss'] = loss
        metrics['val_acc1'] = acc1
        metrics['val_acc5'] = acc5

        return metrics
