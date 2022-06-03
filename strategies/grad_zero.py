from abc import ABC
from collections import defaultdict

import numpy as np

from . import threshold2_mask
from .utils import (fraction_threshold,
                    fraction_mask,
                    flatten_importances,
                    importance_masks, threshold_mask)
from ..pruning import (LayerPruning,
                       VisionPruning,
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
        self.val_grads = self.param_gradients()

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
        pruning_params['F'] = lambda mag, train_grad, val_grad: min(
            beta * np.abs(mag),
            gamma * np.abs(train_grad),
            delta * np.abs(val_grad)
        )

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

            if f"{param_name}_threshold" in pruning_params:
                setattr(self, f"{param_name}_threshold", pruning_params.get(f"{param_name}_threshold"))
                setattr(self, f"{param_name}_fraction", None)
            elif f"{param_name}_fraction" in pruning_params:
                setattr(self, f"{param_name}_threshold", None)
                setattr(self, f"{param_name}_fraction", pruning_params.get(f"{param_name}_fraction"))
            else:
                setattr(self, f"{param_name}_threshold", None)
                setattr(self, f"{param_name}_fraction", self.fraction)

        for param_name in self.param_names:
            _init_fraction_and_threshold(param_name)

        # either "any" or "all": any mask disabled neuron vs all masks disabled neuron
        self.union_method = pruning_params["union_method"]
        if self.union_method not in ["all", "any"]:
            raise ValueError(f"Unknown union_method: {self.union_method}")

    def model_masks(self, prunable=None):
        masks = defaultdict(dict)

        for mod, mod_params in self.mags.items():
            for p in mod_params:
                masks_p = []
                for param_name in self.param_names:
                    param = getattr(self, f"{param_name}s")[mod][p]  # e.g. self.mags[mod][p]

                    if hasattr(self, f"{param_name}_threshold"):
                        true_threshold = getattr(self, f"{param_name}_threshold")
                        negate = true_threshold < 0
                        threshold = abs(true_threshold)
                    else:
                        true_fraction = getattr(self, f"{param_name}_fraction")
                        negate = true_fraction < 0
                        fraction = abs(true_fraction)
                        threshold = fraction_threshold(param, fraction)

                    mask = threshold_mask(param, threshold)
                    if negate:
                        mask = 1 - mask
                    masks_p.append(mask)

                mask = masks_p[0]
                if self.union_method == "any":
                    for i in range(1, len(masks_p)):
                        mask &= masks_p[i]
                elif self.union_method == "all":
                    mask = np.invert(mask)
                    for i in range(1, len(masks_p)):
                        mask &= np.invert(masks_p[i])
                    mask = np.invert(mask)

                masks[mod][p] = mask

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
