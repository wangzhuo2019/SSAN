import torch


def get_optimizer(name, model, lr, momentum=0.9, weight_decay=0.0005):
    parameters = model.parameters()
    if name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer