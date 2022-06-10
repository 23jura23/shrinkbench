import json
import copy

from torch.utils.data import DataLoader

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
                 pruning_dl_kwargs=dict(),
                 save_pruning=None,
                 dl_kwargs=dict(),
                 train_kwargs=dict(),
                 is_multiple_input_model=False,
                 strategy_kwargs=dict(),
                 debug=False,
                 pretrained=True,
                 resume=None,
                 resume_optim=False,
                 save_freq=10):
        super(PruningExperiment, self).__init__(dataset, model, seed, path, dl_kwargs, train_kwargs, is_multiple_input_model, debug, pretrained, resume, resume_optim, save_freq)

        if strategy_name is None:
            strategy_name = strategy
        self.add_params(strategy=strategy, compression=compression, strategy_name=strategy_name)

        self.pruning_dl_kwargs = pruning_dl_kwargs

        self.apply_pruning(strategy, compression, strategy_kwargs, save_pruning)

        self.path = path
        self.save_freq = save_freq

    def _get_pruning_train_dl(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.pruning_dl_kwargs)

    def _get_pruning_val_dl(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.pruning_dl_kwargs)

    def apply_pruning(self, strategy, compression, strategy_kwargs, save_pruning):
        constructor = getattr(strategies, strategy)
        train_batch = next(iter(self._get_pruning_train_dl()))
        val_batch = next(iter(self._get_pruning_val_dl()))
        if self.is_multiple_input_model:
            train_x = copy.copy(train_batch)
            train_x.pop('label')
            train_y = train_batch['label']

            val_x = copy.copy(val_batch)
            val_x.pop('label')
            val_y = val_batch['label']
        else:
            train_x, train_y = train_batch
            val_x, val_y = val_batch
        strategy_kwargs['val_x'] = val_x
        strategy_kwargs['val_y'] = val_y
        self.pruning = constructor(self.model, train_x, train_y, compression=compression, is_multiple_input_model=self.is_multiple_input_model, **strategy_kwargs)
        self.pruning.apply(save=save_pruning)
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
