import warnings
warnings.filterwarnings("ignore")
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils import data

from models.discriminator.discriminator import Discriminator
from models.generator.generator import Generator
from models.generator.vgg16 import VGG16FeatureExtractor
from models.auxiliary_loss_weight import TPL_TSL
from options.train_options import TrainOptions
from datasets.dataset import create_image_dataset
from utils.distributed import synchronize
from utils.ddp import data_sampler
from trainer import train
from lpips import lpips

opts = TrainOptions().parse

os.makedirs('{:s}'.format(opts.save_dir), exist_ok=True)

is_cuda = torch.cuda.is_available()

if is_cuda:
    print('Cuda is available')
    cudnn.enable = True
    cudnn.benchmark = True

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    print('GPU number: ', n_gpu)
    opts.distributed = n_gpu > 1
    if opts.distributed:
        torch.cuda.set_device(opts.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

# model
generator = Generator(image_in_channels=3, edge_in_channels=2, out_channels=3)
generator_ = Generator(image_in_channels=3, edge_in_channels=2, out_channels=3)
discriminator = Discriminator(image_in_channels=3, edge_in_channels=2)
a = TPL_TSL(pw_scale=20,sw_scale=3)
extractor = VGG16FeatureExtractor()
pips = lpips.LPIPS(net='vgg')
if opts.distributed:
    generator = nn.SyncBatchNorm.convert_sync_batchnorm(generator)
    generator_ = nn.SyncBatchNorm.convert_sync_batchnorm(generator_)
    discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

# cuda
if is_cuda:
    generator,generator_, discriminator, extractor,a ,pips= generator.cuda(),generator_.cuda(), discriminator.cuda(), extractor.cuda(),a.cuda(),pips.cuda()


lr = opts.gen_lr
generator_optim = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=lr)
discriminator_optim = optim.Adam(discriminator.parameters(), lr=lr * opts.D2G_lr)
aux_optim = optim.AdamW(a.parameters(), lr=0.001,betas=[0.5,0.9],weight_decay=0.2)

# load checkpoints
if opts.pre_trained != '':
    ckpt_dict = torch.load(opts.pre_trained, map_location=lambda storage, loc: storage)
    opts.start_iter = ckpt_dict['n_iter']
    generator.load_state_dict(ckpt_dict['generator'])
    discriminator.load_state_dict(ckpt_dict['discriminator'])
    a.load_state_dict(ckpt_dict['aux'])
    print('Starting from iter', opts.start_iter)
else:
    print('Starting from iter', opts.start_iter)

if opts.distributed:
    generator = nn.parallel.DistributedDataParallel(
        generator,
        device_ids=[opts.local_rank],
        output_device=opts.local_rank,
    )
    generator_ = nn.parallel.DistributedDataParallel(
        generator_,
        device_ids=[opts.local_rank],
        output_device=opts.local_rank,
    )
    discriminator = nn.parallel.DistributedDataParallel(
        discriminator,
        device_ids=[opts.local_rank],
        output_device=opts.local_rank,
    )
    a = nn.parallel.DistributedDataParallel(
        a,
        device_ids=[opts.local_rank],
        output_device=opts.local_rank,
    )


# dataset
image_dataset = create_image_dataset(opts)
image_dataset2 = create_image_dataset(opts)
print(image_dataset.__len__())

image_data_loader = data.DataLoader(
    image_dataset,
    batch_size=opts.batch_size,
    sampler=data_sampler(
        image_dataset, shuffle=True, distributed=opts.distributed
    ),
    drop_last=True
)
torch.manual_seed(opts.seed+100)
image_data_loader2 = data.DataLoader(
    image_dataset,
    batch_size=opts.batch_size,
    sampler=data_sampler(
        image_dataset, shuffle=True, distributed=opts.distributed
    ),
    drop_last=True
)
generator_.module.train(finetune=True)
torch.cuda.empty_cache()
# training
train(opts, image_data_loader,image_data_loader2, generator,generator_,a, discriminator, extractor,pips, generator_optim, aux_optim,discriminator_optim, is_cuda)