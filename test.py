import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
from tqdm import tqdm
from imageio import imsave

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils import data
from torchvision.utils import save_image

from models.generator.generator import Generator
from datasets.dataset import create_image_dataset
from options.test_options import TestOptions
from utils.misc import sample_data, postprocess
from utils.distributed import synchronize

is_cuda = torch.cuda.is_available()

opts = TestOptions().parse

os.makedirs('{:s}'.format(opts.result_root), exist_ok=True)
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

# model & load model
generator = Generator(image_in_channels=3, edge_in_channels=2, out_channels=3)
if opts.distributed:
    generator = nn.SyncBatchNorm.convert_sync_batchnorm(generator)

if opts.pre_trained != '':
    generator.load_state_dict(torch.load(opts.pre_trained)['generator'])
    torch.cuda.empty_cache()
else:
    print('Please provide pre-trained model!')

if is_cuda:
    generator = generator.cuda()

if opts.distributed:
    generator = nn.parallel.DistributedDataParallel(
        generator,
        device_ids=[opts.local_rank],
        output_device=opts.local_rank,
    )

# dataset
image_dataset = create_image_dataset(opts)
image_data_loader = data.DataLoader(
    image_dataset,
    batch_size=opts.batch_size,
    shuffle=False,
    num_workers=opts.num_workers,
    drop_last=False
)
image_data_loader = sample_data(image_data_loader)
opts.number_eval=int(len(image_dataset.image_files)/opts.batch_size)
print('start test...')
with torch.no_grad():
    generator.train(finetune=True)
    for _ in tqdm(range(opts.number_eval)):

        ground_truth, mask, edge, gray_image,index = next(image_data_loader)
        if is_cuda:
            ground_truth, mask, edge, gray_image = ground_truth.cuda(), mask.cuda(), edge.cuda(), gray_image.cuda()

        input_image, input_edge, input_gray_image = ground_truth * mask, edge * mask, gray_image * mask

        output, __, __ = generator(input_image, torch.cat((input_edge, input_gray_image), dim=1), mask)
        output_comp = ground_truth * mask + output * (1 - mask)
        output_comp = postprocess(output_comp)
        for i in range(output_comp.size(0)):
            filename = image_dataset.load_name(index[i])
            if not filename.endswith('png'): filename = filename[:-3]+'png'
            save_image(output_comp[i:i+1], opts.result_root + '/{}'.format(filename))
