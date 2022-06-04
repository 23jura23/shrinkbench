import numpy as np
import pandas as pd

from .modules import masked_modules, _ensure_tensor, _same_device, MaskedModule, _same_shape


def mask_module(module, masks):
    """Recursively mask a torch.nn Module

    Changes layers so that backprop doesn't get to masked parameters
    Note this operates inplace and modifies the passed network

    Arguments:
        module {torch.nn.Module} -- Module to mask
        masks Dict(torch.nn.Module : Dict(str:numpy.ndarray))
            -- Dictionary with masks for each weight tensor

    Returns:
        torch.nn.Module -- Same as id as input module, but after masking
    """

    # Need to store the new children so iteration won't break
    new_children = {}

    masked_parameters = 0

    for name, submodule in module.named_children():

        if submodule in masks:
            mask_kwargs = {k+'_mask': v for k, v in masks[submodule].items()}
            if isinstance(submodule, MaskedModule):
                new_masks = {k: _calc_new_mask(getattr(submodule, k), v) for k, v in mask_kwargs.items()}
                masked_parameters += sum(_calc_diff(getattr(submodule, k), v) for k, v in mask_kwargs.items())
                submodule.set_masks(**new_masks)
            else:
                masked_parameters += sum(_calc_diff(None, v) for v in mask_kwargs.values())
                masked = masked_modules[type(submodule)](submodule, **mask_kwargs)
                new_children[name] = masked

        # Recurse for children
        mask_module(submodule, masks)

    # We replace the children outside of loop
    # otherwise the iterator will change
    for name, masked in new_children.items():
        setattr(module, name, masked)

    return module, int(masked_parameters)


def _calc_new_mask(old_mask, new_mask):
    assert old_mask is not None and new_mask is not None
    if old_mask is None:
        return new_mask
    else:
        return old_mask.detach().cpu().numpy() * new_mask


def _calc_diff(old_mask, new_mask):
    assert new_mask is not None
    assert old_mask is None or _same_shape(new_mask, old_mask)
    if old_mask is None:
        return np.sum(1 - new_mask)
    else:
        old_mask = old_mask.detach().cpu().numpy()
        return np.sum(1 - new_mask * old_mask) - np.sum(1 - old_mask)


def apply_masks(module, masks):
    """Recursively mask a torch.nn Module

    Zeros out masked parameters, does not change the layer
    Note this operates inplace and modifies the passed network

    Arguments:
        module {torch.nn.Module} -- Module to mask
        masks Dict(torch.nn.Module : Dict(str:numpy.ndarray))
            -- Dictionary with masks for each weight tensor

    Returns:
        torch.nn.Module -- Same as id as input module, but after masking
    """
    for name, submodule in module.named_children():

        if submodule in masks:

            for attr, mask in masks[submodule].items():
                param = getattr(submodule, attr)
                mask = _same_device(_ensure_tensor(mask), param)
                param.data.mul_(mask)

        # Recurse if children
        apply_masks(submodule, masks)

    return module


# Aux functions

def masks_details(model, masks):
    """Debug information for collection of masks

    Returns a dataframe with summary information of all masks

    Arguments:
        model {torch.nn.Module} -- torch module that the masks correspond to
        masks Dict(torch.nn.Module : Dict(str:numpy.ndarray))
            -- Dictionary with masks for each weight tensor

    Returns:
        pandas.DataFrame -- DataFrame with compression, size and shape for each mask
    """
    rows = []
    for name, module in model.named_modules():
        if module in masks:
            for k, v in masks[module].items():
                rows.append([name, k, 1/v.mean(), np.prod(v.shape), v.shape])
    columns = ['module', 'param', 'comp', 'size', 'shape']
    return pd.DataFrame(rows, columns=columns)
