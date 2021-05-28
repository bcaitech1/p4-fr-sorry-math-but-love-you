from torch import nn


def get_criterion(type: str):
    if type == 'CE':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
    return criterion