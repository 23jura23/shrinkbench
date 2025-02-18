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


# mask based on arbitrary function F
class GlobalMagGradValF(GlobalMagGradValBased):
    def __init__(self, model, inputs=None, outputs=None, compression=1, **pruning_params):
        super().__init__(model, inputs, outputs, compression, **pruning_params)

        # F: mag -> train_grad -> grad_val -> tensor[float]
        # synapses with the lowest by absolute value will be masked with 0 (and so removed)
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

# use F(mag, train_grad, val_grad) = abs(beta * mag + gamma * train_grad + delta * val_grad)
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


# use F(mag, train_grad, val_grad) = min(abs(beta * mag), abs(gamma * train_grad), abs(delta * val_grad))
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

# use F(mag, train_grad, val_grad) = abs(mag^beta * train_grad^gamma * val_grad^delta)
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

# specify mag_threshold/mag_fraction, train_grad_threshold/train_grad_fraction, val_grad_threshold/val_grad_fraction
# e.g. mag_threshold: if abs(mag) < mag_threshold, then ban
# e.g. mag_fraction: if abs(mag) < mag_fraction quantile, then ban
# also specify union_method: all => ban if all criteria ban, any => ban if any criteria ban
# e.g. Belay with beta = 0.05, gamma = 0.05: mag_threshold = 0.05, train_grad_threshold = 0.05, val_grad_fraction = 1.0
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
                        threshold = np.quantile(np.abs(param), fraction)

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


# used for modification of Belay
# set val_grad_threshold or val_grad_fraction (see GlobalMagGradValSeparate)
# then mag_threshold and train_grad_threshold will be found using binary search, so that the total model compression
# is equal to passed parameter compression
# also there is attribute have_common_threshold: find common threshold with binary search for all layers,
# or search own threshold for each layer
# original Belay: have_common_threshold = True, compression ~= 2, val_grad_fraction = 1.0
# modified Belay:have_common_threshold = True, compression ~= 2, val_grad_fraction = [0.99, 0.95, 0.9]
class GlobalMagGradTopVal(GlobalMagGradValBased):
    def __init__(self, model, inputs=None, outputs=None, compression=1, **pruning_params):
        super().__init__(model, inputs, outputs, compression, **pruning_params)

        self.param_names = ["mag", "train_grad", "val_grad"]
        self.have_common_threshold = pruning_params.get("have_common_threshold") or True
        self.mag_ub = pruning_params.get("mag_ub") or 1.
        self.grad_ub = pruning_params.get("grad_ub") or 1.

        self.use_val_grad = pruning_params.get("use_val_grad") or False

        if not self.use_val_grad and \
                ("val_grad_fraction" in pruning_params or
                 "val_grad_threshold" in pruning_params or
                 "val_grad_and_or" in pruning_params):
            raise ValueError("use_val_grad = False, but val_grad args specified")

        self.val_grad_fraction = pruning_params.get("val_grad_fraction")
        self.val_grad_threshold = pruning_params.get("val_grad_threshold")
        self.val_grad_and_or = pruning_params.get("val_grad_and_or") or "and"

        self.debug = pruning_params.get("debug") or False

        if not self.use_val_grad:
            self.val_grad_fraction = 1.
            self.val_grad_and_or = "and"

    def model_masks(self, prunable=None):
        fraction_l, fraction_r = 0., 10.
        masks = defaultdict(dict)
        for it in range(20):
            fraction = (fraction_l + fraction_r) / 2

            prunable_size_h = 0
            non_pruned_size_h = 0
            val_grad_diff_size_h = 0
            val_grad_only_size_h = 0

            for mod, mod_params in self.params().items():
                for p in mod_params:
                    masks_p = []
                    for param_name in self.param_names:
                        param = getattr(self, f"{param_name}s")[mod][p]  # e.g. self.mags[mod][p]

                        if param_name == "mag":
                            true_fraction = fraction * self.mag_ub
                            if self.have_common_threshold:
                                threshold = true_fraction
                            else:
                                threshold = np.quantile(np.abs(param), true_fraction)
                        elif param_name == "train_grad":
                            true_fraction = fraction * self.grad_ub
                            if self.have_common_threshold:
                                threshold = true_fraction
                            else:
                                threshold = np.quantile(np.abs(param), true_fraction)
                        else:
                            if self.val_grad_threshold is not None:
                                threshold = self.val_grad_threshold
                            elif self.val_grad_fraction is not None:
                                threshold = np.quantile(np.abs(param), self.val_grad_fraction)
                            else:
                                raise ValueError("Either val_grad_threshold or val_grad_fraction must be set")

                        mask = threshold_mask(param, threshold)
                        masks_p.append(mask)

                    mask = 1 - (1 - masks_p[0]) * (1 - masks_p[1])

                    pre_val = np.array(mask)
                    if self.val_grad_and_or == "and":
                        mask = 1 - (1 - mask) * (1 - masks_p[2])
                    else:
                        mask = mask * masks_p[2]
                    post_val = np.array(mask)

                    masks[mod][p] = np.array(mask)

                    prunable_size_h += np.prod(masks[mod][p].shape)
                    non_pruned_size_h += np.sum(masks[mod][p])
                    val_grad_diff_size_h += np.sum(post_val) - np.sum(pre_val)
                    val_grad_only_size_h += np.sum(masks_p[2])

            from ..metrics import model_size
            total_size, _ = model_size(self.model)
            prunable_size = sum([model_size(m)[0] for m in self.prunable])
            nonprunable_size = total_size - prunable_size
            if self.debug:
                print(it, fraction)
                print(total_size, nonprunable_size, prunable_size)
                print(prunable_size_h, non_pruned_size_h, prunable_size_h - non_pruned_size_h)
                print("val_grad_diff_size_h", val_grad_diff_size_h)
                print("val_grad_size_only_h", val_grad_only_size_h, prunable_size_h - val_grad_only_size_h)

            real_fraction = non_pruned_size_h / prunable_size_h  # real fraction to keep
            if self.debug:
                print(real_fraction, self.fraction)
                print()

            if real_fraction < self.fraction:
                fraction_r = fraction
            else:
                fraction_l = fraction
        if self.debug:
            print(f"Choose global fraction: {fraction_l}")
            print(f"Mag threshold: ban if < {fraction_l * self.mag_ub}")
            print(f"Grad threshold: ban if < {fraction_l * self.grad_ub}")
            if not self.use_val_grad:
                print("Val threshold: ignored")
            else:
                print(f"Val threshold: ban if < {self.val_grad_threshold or f'{self.val_grad_fraction}%'}")
        return masks
