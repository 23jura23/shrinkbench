import numpy as np

from ..pruning import (LayerPruning,
                       VisionPruning,
                       GradientMixin,
                       ActivationMixin)
from .utils import (fraction_threshold,
                    fraction_mask,
                    map_importances,
                    flatten_importances,
                    importance_masks,
                    activation_importance)

class GlobalMagGradZero(GradientMixin, VisionPruning):

    def model_masks(self):
        params = self.params()
        grads = self.param_gradients()
        importances = {mod:
                       {p: np.abs(grads[mod][p])
                        for p in mod_params}
                       for mod, mod_params in params.items()}
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks

class LayerMagGradZero(GradientMixin, LayerPruning, VisionPruning):

    def layer_masks(self, module):
        params = self.module_params(module)
        grads = self.module_param_gradients(module)
        importances = {param: np.abs(grads[param]) for param, value in params.items()}
        masks = {param: fraction_mask(importances[param], self.fraction)
                 for param, value in params.items() if value is not None}
        return masks