import torch
from torch import nn
def meta_update_simple(model:nn.Module,lr,grads):
    """

    :param model:
    :param loss:
    :param lr:
    :return: updated model parameters
    """
    dictionary = model.state_dict()
    for param,value,grad in zip(dictionary.keys(),dictionary.values(),grads):
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

class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.m = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.m(x)

class MetaM(meta.MetaModule):
    def __init__(self):
        super(MetaM, self).__init__()
        self.m = meta.MetaSequential(
            meta.MetaConv2d(3, 3, 3, padding=1),
            nn.ReLU(),
            meta.MetaConv2d(3, 3, 3, padding=1),
            nn.ReLU(),
            meta.MetaConv2d(3, 3, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.m(x)

if __name__ == '__main__':
    X = MetaM()
    X_ = MetaM()
    state_dict = X.state_dict()
    lr = 0.1
    t = torch.randn(1).cuda()
    a = t.clone()
    I = torch.randn(2, 3, 256, 256).cuda()
    def print_origin_grad():
        X_ = MetaM()
        X_.load_state_dict(X.state_dict())
        a = t.clone()
        a.requires_grad_(True)
        O = X_(I)
        l1 = O.mean()
        l2 = a*(1-O).mean()
        loss = l1+l2
        grads = torch.autograd.grad(loss, X_.params(),create_graph=True)
        X_.update_params(lr,source_params=grads)
        O = X_(I)
        l1 = O.mean()
        l1.backward()
        print(a.grad)

    def check_param(X,Y):
        for p,q in zip(X.params(),Y.parameters()):
            print(p-q)
    Y,Z = M(),M()

    Y.cuda()
    Z.cuda()
    Y.load_state_dict(state_dict)
    a = t.clone()
    def print_grad1():
        O = Y(I)
        l1 = O.mean()
        l2 = (1 - O).mean()
        loss = l1 + a*l2
        grads = torch.autograd.grad(loss, Y.parameters(),retain_graph=True)
        Z.load_state_dict(meta_update_simple(Y,lr,grads))
        grads_aux = torch.autograd.grad(l2,Y.parameters(),retain_graph=True)
        O = Z(I)
        l1 = O.mean()
        grads_p = torch.autograd.grad(l1,Z.parameters(),retain_graph=False)
        print(lr*param_grad_dot(grads_aux,grads_p))
        Y.zero_grad()
        Z.zero_grad()
    # def print_grad2():
    #     O1 = Y(I)
    #     grad_o1_t1 = torch.autograd.grad(O1,Y.parameters(),grad_outputs=torch.ones_like(O1),retain_graph=True)
    #     l1 = O1.mean()
    #     l2 = (1 - O1).mean()
    #     loss = l1 + a * l2
    #     grads = torch.autograd.grad(loss, Y.parameters(), retain_graph=True)
    #     Z.load_state_dict(meta_update_simple(Y, lr, grads))
    #     O2 = Z(I)
    #     grad_o2_t2 = torch.autograd.grad(O2,Z.parameters(),grad_outputs=torch.ones_like(O2),retain_graph=True)
    #     grad_l1_o1 = torch.autograd.grad(l1,O1)
    #     l2 = (1-O2).mean()
    #     grad_l2_o2 = torch.autograd.grad(l2,O2)
    #     print(param_grad_dot(grad_o2_t2,grad_o1_t1)*lr*feature_grad_dot(grad_l1_o1,grad_l2_o2))
    import time
    s = time.time()
    for i in range(20):
        print_origin_grad()
    e = time.time()
    print('time:{}'.format((e-s)/100))
    s = e
    for i in range(20):
        print_grad1()
    e = time.time()
    print('time:{}'.format((e-s)/100))
    # print_grad2()









