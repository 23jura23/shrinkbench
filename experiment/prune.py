import json
import numpy as np

from .train import TrainingExperiment

from .. import strategies
from ..metrics import model_size, flops
from ..util import printc


class PruningExperiment(TrainingExperiment):

    def __init__(self,
                 dataset,
                 model,
                 strategy,
                 compression,
                 strategy_name=None,
                 seed=42,
                 path=None,
                 dl_kwargs=dict(),
                 train_kwargs=dict(),
                 strategy_kwargs=dict(),
                 debug=False,
                 pretrained=True,
                 resume=None,
                 resume_optim=False,
                 save_freq=10):
        super(PruningExperiment, self).__init__(dataset, model, seed, path, dl_kwargs, train_kwargs, debug, pretrained, resume, resume_optim, save_freq)

        if strategy_name is None:
            strategy_name = strategy
        self.add_params(strategy=strategy, compression=compression, strategy_name=strategy_name)

        self.apply_pruning(strategy, compression, strategy_kwargs)

        self.path = path
        self.save_freq = save_freq

    def _calc_nonzero_prunable_params(self, mags):
        return sum(sum(np.sum(p != 0) for p in mod_params.values()) for mod_params in mags.values())

    def _calculate_total_compressed_params(self, strategy, strategy_kwargs, train_x, train_y):
        constructor = getattr(strategies, strategy)
        pruning = constructor(self.model, train_x, train_y, compression=1.0, **strategy_kwargs)

        num_prunable_params = sum(sum(p.size for p in mod_params.values()) for mod_params in pruning.mags.values())

        fraction = strategy_kwargs['train_grad_fraction']
        if fraction < 0:
            fraction = 1 - abs(fraction)

        return int(num_prunable_params * fraction)

    def apply_pruning(self, strategy, compression, strategy_kwargs):
        # TODO For now it works only with this strategy
        assert strategy == 'GlobalMagGradValSeparate'
        constructor = getattr(strategies, strategy)
        iters = strategy_kwargs['iters']

        one_iter_params = self._calculate_total_compressed_params(strategy, strategy_kwargs, *next(iter(self.train_dl)))
        one_iter_params /= iters

        one_iter_strategy_kwargs = {"union_method": strategy_kwargs["union_method"]}
        if 'train_grad_threshold' in strategy_kwargs:
            one_iter_strategy_kwargs['train_grad_threshold'] = strategy_kwargs['train_grad_threshold']
        else:
            one_iter_strategy_kwargs['train_grad_fraction'] = 1.0

        for i, (train_x, train_y) in zip(range(iters), self.train_dl):
            self.pruning = constructor(self.model, train_x, train_y, compression=1.0, **one_iter_strategy_kwargs)
            self.pruning.train_grad_fraction = one_iter_params / self._calc_nonzero_prunable_params(self.pruning.mags)
            self.pruning.apply()
        printc("Masked model", color='GREEN')

    def run(self):
        self.freeze()
        printc(f"Running {repr(self)}", color='YELLOW')
        self.to_device()
        self.build_logging(self.train_metrics, self.path)

        self.save_metrics()

        if self.pruning.compression > 1:
            self.run_epochs()

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
