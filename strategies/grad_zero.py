from abc import ABC
from collections import defaultdict

import numpy as np

from .utils import (fraction_threshold,
                    flatten_importances,
                    importance_masks, threshold_mask)
from ..pruning import (VisionPruning,
                       GradientMixin)


def threshold_n_mask(tensors, thresholds):
    assert len(tensors) == len(thresholds)
    assert len(tensors) > 0
    tensor0 = tensors[0]
    idx = np.ones_like(tensor0, dtype=bool)
    for tensor, threshold in zip(tensors, thresholds):
        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == tensor0.shape

        idx = np.logical_and(
            idx,
            np.logical_and(tensor < threshold, tensor > -threshold),
        )
    mask = np.ones_like(idx)
    mask[idx] = 0
    return mask


class GlobalMagGradValBased(GradientMixin, VisionPruning, ABC):
    def __init__(self, model, inputs=None, outputs=None, compression=1,
                 **pruning_params):
        super().__init__(model, inputs, outputs, compression, **pruning_params)
        self.train_x = inputs
        self.train_y = outputs

        self.mags = self.params()

        self.train_grads = self.param_gradients()
        self.val_x = pruning_params['val_x']
        self.val_y = pruning_params['val_y']

        self.inputs = self.val_x
        self.outputs = self.val_y
        self.val_grads = self.param_gradients(update_anyway=True)

        self.inputs = self.train_x
        self.outputs = self.train_y


class GlobalMagGradValF(GlobalMagGradValBased):
    def __init__(self, model, inputs=None, outputs=None, compression=1, **pruning_params):
        super().__init__(model, inputs, outputs, compression, **pruning_params)

        # F: mag -> train_grad -> grad_val -> float
        # the lowest by absolute value will be masked
        self.F = pruning_params['F']

    def model_masks(self, prunable=None):
        importances = {mod:
            {p: self.F(
                self.mags[mod][p],
                self.train_grads[mod][p],
                self.val_grads[mod][p]
            )
                for p in mod_params}
            for mod, mod_params in self.mags.items()}
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks


class GlobalMagGradValSum(GlobalMagGradValF):
    def __init__(self, model, inputs=None, outputs=None, compression=1, **pruning_params):
        # following Belay's original notation for beta and gamma
        beta = pruning_params.get('beta') or 1.0
        gamma = pruning_params.get('gamma') or 1.0
        delta = pruning_params.get('delta') or 1.0
        pruning_params['F'] = lambda mag, train_grad, val_grad: \
            beta * np.abs(mag) + \
            gamma * np.abs(train_grad) + \
            delta * np.abs(val_grad)

        super().__init__(model, inputs, outputs, compression, **pruning_params)


class GlobalMagGradValMin(GlobalMagGradValF):
    def __init__(self, model, inputs=None, outputs=None, compression=1, **pruning_params):
        # following Belay's original notation for beta and gamma
        beta = pruning_params.get('beta') or 1.0
        gamma = pruning_params.get('gamma') or 1.0
        delta = pruning_params.get('delta') or 1.0
        pruning_params['F'] = lambda mag, train_grad, val_grad: np.minimum.reduce([
            beta * np.abs(mag),
            gamma * np.abs(train_grad),
            delta * np.abs(val_grad)
        ])

        super().__init__(model, inputs, outputs, compression, **pruning_params)


class GlobalMagGradValProd(GlobalMagGradValF):
    def __init__(self, model, inputs=None, outputs=None, compression=1, **pruning_params):
        # following Belay's original notation for beta and gamma
        beta = pruning_params.get('beta') or 1.0
        gamma = pruning_params.get('gamma') or 1.0
        delta = pruning_params.get('delta') or 1.0
        eps = pruning_params.get('eps') or 1e-6
        pruning_params['F'] = lambda mag, train_grad, val_grad: np.abs(
            ((np.abs(mag) + eps) ** beta) *
            ((np.abs(train_grad) + eps) ** gamma) *
            ((np.abs(val_grad) + eps) ** delta)
        )

        super().__init__(model, inputs, outputs, compression, **pruning_params)


class GlobalMagGradValSeparate(GlobalMagGradValBased):
    def __init__(self, model, inputs=None, outputs=None, compression=1, **pruning_params):
        super().__init__(model, inputs, outputs, compression, **pruning_params)

        self.param_names = ["mag", "train_grad", "val_grad"]

        def _init_fraction_and_threshold(param_name):
            if f"{param_name}_threshold" in pruning_params and f"{param_name}_fraction" in pruning_params:
                raise ValueError(f"Only on of '{param_name}_threshold' and '{param_name}_fraction' must be present")

            # Attention: framework authors use fraction as "fraction to keep", which is not convenient in my opinion
            # so our fraction is "fraction to remove", e.g. 0.05 means removing 0.05 neurons with the lowest importance

            if f"{param_name}_threshold" in pruning_params:
                setattr(self, f"{param_name}_threshold", pruning_params.get(f"{param_name}_threshold"))
                setattr(self, f"{param_name}_fraction", None)
            elif f"{param_name}_fraction" in pruning_params:
                setattr(self, f"{param_name}_threshold", None)
                setattr(self, f"{param_name}_fraction", pruning_params.get(f"{param_name}_fraction"))
            else:
                setattr(self, f"{param_name}_threshold", None)
                setattr(self, f"{param_name}_fraction", 1 - self.fraction)

        for param_name in self.param_names:
            _init_fraction_and_threshold(param_name)

        # either "any" or "all": any mask disabled neuron vs all masks disabled neuron
        self.union_method = pruning_params["union_method"]
        if self.union_method not in ["all", "any"]:
            raise ValueError(f"Unknown union_method: {self.union_method}")

    def model_masks(self, prunable=None):
        masks = defaultdict(dict)

        for mod, mod_params in self.params():
            for p in mod_params:
                masks_p = []
                for param_name in self.param_names:
                    param = getattr(self, f"{param_name}s")[mod][p]  # e.g. self.mags[mod][p]

                    true_threshold = getattr(self, f"{param_name}_threshold")
                    if true_threshold is not None:
                        negate = true_threshold < 0
                        threshold = abs(true_threshold)
                    else:
                        true_fraction = getattr(self, f"{param_name}_fraction")
                        negate = true_fraction < 0
                        fraction = abs(true_fraction)
                        threshold = np.quantile(param, fraction)

                    mask = threshold_mask(param, threshold)
                    if negate:
                        mask = 1 - mask
                    masks_p.append(mask)

                mask = masks_p[0]
                if self.union_method == "any":
                    for i in range(1, len(masks_p)):
                        mask *= masks_p[i]
                elif self.union_method == "all":
                    mask = 1 - mask
                    for i in range(1, len(masks_p)):
                        mask *= 1 - masks_p[i]
                    mask = 1 - mask

                masks[mod][p] = np.array(mask)

        return masks


# set fraction for magnitude, train_grad and (optionally) val_grad basing on the total compression level
class GlobalMagGradTopVal(GlobalMagGradValBased):
    def __init__(self, model, inputs=None, outputs=None, compression=1, **pruning_params):
        super().__init__(model, inputs, outputs, compression, **pruning_params)

        self.param_names = ["mag", "train_grad", "val_grad"]
        self.use_val_grad = pruning_params.get("use_val_grad") or False
        self.mag_ub = pruning_params.get("mag_ub") or 0.3
        self.grad_ub = pruning_params.get("grad_ub") or 0.3
        self.val_grad_lb = pruning_params.get("val_grad_lb") or 0.7

    def model_masks(self, prunable=None):
        fraction_l, fraction_r = 0., 1.
        masks = defaultdict(dict)
        for it in range(100):
            fraction = (fraction_l + fraction_r) / 2

            prunable_size_h = 0
            non_pruned_size_h = 0

            for mod, mod_params in self.params():
                for p in mod_params:
                    masks_p = []
                    for param_name in self.param_names:
                        param = getattr(self, f"{param_name}s")[mod][p]  # e.g. self.mags[mod][p]

                        # negate = true_fraction < 0
                        # fraction = abs(true_fraction)
                        if param_name == "mag":
                            true_fraction = fraction * self.mag_ub
                        elif param_name == "grad":
                            true_fraction = fraction * self.grad_ub
                        elif param_name == "val_grad":
                            if self.use_val_grad:
                                true_fraction = self.val_grad_lb + fraction * (1 - self.val_grad_lb)
                            else:
                                true_fraction = 1.
                        else:
                            raise ValueError()

                        threshold = np.quantile(param, true_fraction)

                        mask = threshold_mask(param, threshold)
                        # if negate:
                        #     mask = 1 - mask
                        masks_p.append(mask)

                    mask = masks_p[0]
                    mask = 1 - mask
                    for i in range(1, len(masks_p)):
                        mask *= 1 - masks_p[i]
                    mask = 1 - mask

                    masks[mod][p] = np.array(mask)

                    prunable_size_h += np.prod(masks[mod][p].shape)
                    non_pruned_size_h += np.sum(masks[mod][p])

            from ..metrics import model_size
            total_size, _ = model_size(self.model)
            prunable_size = sum([model_size(m)[0] for m in self.prunable])
            nonprunable_size = total_size - prunable_size
            print(total_size, nonprunable_size, prunable_size)
            print(prunable_size_h, non_pruned_size_h, prunable_size_h - non_pruned_size_h)

            real_fraction = non_pruned_size_h / prunable_size_h  # real fraction to keep

            if real_fraction < self.fraction:
                fraction_r = fraction
            else:
                fraction_l = fraction

        return masks

# class LayerMagGradZero(GradientMixin, LayerPruning, VisionPruning):
#     def __init__(self, model, inputs=None, outputs=None, compression=1, exact_fraction=None, **pruning_params):
#         super().__init__(model, inputs, outputs, compression, **pruning_params)
#         if exact_fraction is not None:
#             self.fraction = exact_fraction
#
#     def layer_masks(self, module):
#         params = self.module_params(module)
#         grads = self.module_param_gradients(module)
#         importances = {param: np.abs(grads[param]) for param, value in params.items()}
#         masks = {param: fraction_mask(importances[param], self.fraction)
#                  for param, value in params.items() if value is not None}
#         return masks
