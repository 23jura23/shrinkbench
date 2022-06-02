import numpy as np

from ..pruning import (LayerPruning,
                       VisionPruning,
                       GradientMixin)


def threshold2_mask(tensor1, tensor2, threshold1, threshold2):
    assert isinstance(tensor1, np.ndarray)
    assert isinstance(tensor2, np.ndarray)
    assert tensor1.shape == tensor2.shape
    idx = np.logical_and(
        np.logical_and(tensor1 < threshold1, tensor1 > -threshold1),
        np.logical_and(tensor2 < threshold2, tensor2 > -threshold2),
    )
    mask = np.ones_like(tensor1)
    mask[idx] = 0
    return mask


class GlobalMagGradBelay(GradientMixin, VisionPruning):
    def __init__(self, model, inputs=None, outputs=None, compression=1, beta=0.05, gamma=0.05, **pruning_params):
        super().__init__(model, inputs, outputs, compression, **pruning_params)
        self.beta = beta
        self.gamma = gamma

    def model_masks(self):
        params = self.params()
        grads = self.param_gradients()
        masks = {mod:
                     {p: threshold2_mask(params[mod][p], grads[mod][p], self.beta, self.gamma)
                      for p in mod_params}
                 for mod, mod_params in params.items()}
        return masks


class LayerMagGradBelay(GradientMixin, LayerPruning, VisionPruning):
    def __init__(self, model, inputs=None, outputs=None, compression=1, beta=0.05, gamma=0.05, **pruning_params):
        super().__init__(model, inputs, outputs, compression, **pruning_params)
        self.beta = beta
        self.gamma = gamma

    def layer_masks(self, module):
        params = self.module_params(module)
        grads = self.module_param_gradients(module)
        masks = {param: threshold2_mask(value, grads[param], self.beta, self.gamma)
                 for param, value in params.items() if value is not None}
        return masks
