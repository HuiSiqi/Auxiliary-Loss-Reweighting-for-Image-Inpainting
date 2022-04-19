import torch
from torch import nn

class TPL_TSL(nn.Module):
    def __init__(self,pw_scale=2,sw_scale=2):
        super(TPL_TSL, self).__init__()
        self.pw_scale = pw_scale
        self.sw_scale = sw_scale
        pw0 = -torch.log(torch.tensor(pw_scale - 1))
        sw0 = -torch.log(torch.tensor(sw_scale - 1))
        self.perc_weight = nn.Parameter(torch.tensor([pw0]*3).view(-1),requires_grad=True)
        self.styl_weight = nn.Parameter(torch.tensor([sw0]*3).view(-1),requires_grad=True)
    def forward(self,x=0):
        return self.pw_scale*(self.perc_weight).sigmoid(),self.sw_scale*(self.styl_weight).sigmoid()
