import torch
import torchvision.utils as vutils
from matplotlib import colors
import os
from torch import nn

COLORMAP = [
    colors.to_rgba('tab:blue'),
    colors.to_rgba('tab:orange'),
    colors.to_rgba('tab:green'),
    colors.to_rgba('tab:red'),
    colors.to_rgba('tab:purple'),
    colors.to_rgba('tab:brown'),
    colors.to_rgba('tab:pink'),
    colors.to_rgba('tab:gray'),
    colors.to_rgba('tab:olive'),
    colors.to_rgba('tab:cyan'),
]

def index2color(index: torch.tensor, cmap: list):
    n,c,h,w = index.shape
    index = index.repeat(1,3,1,1).unsqueeze(dim=1)
    cmap = torch.tensor([list(c)[:3] for c in cmap]).view(1, -1, 3, 1, 1).to(index.device)  # 1x10
    cmap = cmap.repeat(n,1,1,h,w)
    return cmap.gather(index=index,dim=1).squeeze()

def save_img( name, *args):
    # this only collect a piece of data 1/ngpus
    # REQUIRED
    with torch.no_grad():
        # batch = batch.to(self.hparams['gpu_ids'][0])
        viz_max_out = 4
        if args[0].size(0) > viz_max_out:
            viz_images = torch.stack(
                [x[:viz_max_out] for x in args],
                dim=1)
        else:
            viz_images = torch.stack(args, dim=1)
        viz_images = viz_images.view(-1, *list(args[0].size())[1:])
        vutils.save_image(viz_images,
                          name,
                          nrow=len(args),
                          normalize=True, scale_each=True)

def save_task(aux_loss_fun: nn.Module, path):
    o = aux_loss_fun.state_dict()
    torch.save(o,os.path.join(path))
if __name__ == '__main__':
    index =  torch.randint(0,10,(2,1,2,2))
    out = index2color(index,COLORMAP)
    print(out)