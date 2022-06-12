import torch


def pretrained_weights(model):
    if model == "cifar100_resnet20":
        url = "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet20-23dac2f1.pt"
    else:
        url = f'https://raw.githubusercontent.com/JJGO/shrinkbench-models/master/cifar10/{model}.th'

    print(url)
    return torch.hub.load_state_dict_from_url(url, map_location='cpu')
