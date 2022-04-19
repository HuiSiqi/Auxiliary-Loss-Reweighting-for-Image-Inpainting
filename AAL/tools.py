import torch
from torch import nn
def one_step_update(model:nn.Module,lr,grads):
    """

    :param model:
    :param loss:
    :param lr:
    :return: updated model parameters
    """
    dictionary = model.state_dict()
    for _,grad in zip(model.named_parameters(),grads):
        param, value = _
        dictionary[param] = value-lr*grad
    return dictionary

def flatt_parameter(grad):
    return torch.cat([_.view(-1, 1) for _ in grad], dim=0)
def flatt_feature(grad):
    grad = grad[0]
    # grad: nxcxhxw
    return grad.view(-1)

def param_grad_dot(a_grads,b_grads):
    a_grads = flatt_parameter(a_grads)
    b_grads = flatt_parameter(b_grads)
    return torch.matmul(a_grads.view(-1),b_grads.view(-1))

def feature_grad_dot(a_grads,b_grads):
    a_grads = flatt_feature(a_grads)
    b_grads = flatt_feature(b_grads)
    return torch.matmul(a_grads,b_grads)





