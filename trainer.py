import os
from torchvision import utils as vutils
import torch
import numpy as np
from tqdm import tqdm
from AAL import tools
from torch import optim
from utils.distributed import get_rank, reduce_loss_dict
from utils.misc import requires_grad, sample_data
from criteria.loss import weighted_generator_loss_func, discriminator_loss_func

def train(opts, image_data_loader,image_data_loader2, generator,generator_,a, discriminator, extractor,lpips, generator_optim,a_optim, discriminator_optim, is_cuda):
    global lr
    image_data_loader = sample_data(image_data_loader)
    image_data_loader2 = sample_data(image_data_loader2)
    pbar = range(opts.train_iter+opts.finetune_iter)
    os.makedirs(os.path.join(opts.log_dir,'img'),exist_ok=True)
    file = open(opts.log_dir + '/weights.txt', 'w').close()
    file = open(opts.log_dir + '/l1.txt', 'w').close()
    lr = opts.gen_lr
    IsTrain=True
    def decay_lr():
        global lr
        for param_group in generator_optim.param_groups:
            param_group['lr'] = opts.lr_finetune
            if opts.local_rank == 0:
                print('===> Current G learning rate: ', param_group['lr'])
        for param_group in discriminator_optim.param_groups:
            param_group['lr'] = opts.lr_finetune*opts.D2G_lr
            if opts.local_rank == 0:
                print('===> Current D learning rate: ', param_group['lr'])
        lr = opts.lr_finetune
        # for param_group in self.optm_A.param_groups:
        #     param_group['lr'] = self.lr * 0.1
        #     if self.rank == 0:
        #         print('===> Current A learning rate: ', param_group['lr'])
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=opts.start_iter, dynamic_ncols=True, smoothing=0.1)

    if opts.distributed:
        generator_module, a_module, discriminator_module = generator.module, a.module, discriminator.module
    else:
        generator_module, a_module, discriminator_module = generator, a, discriminator

    for index in pbar:
        i = index + opts.start_iter
        if i > opts.train_iter+opts.finetune_iter:
            print('Done...')
            break
        if i > opts.train_iter and IsTrain:
            print('Begin Finetune')
            decay_lr()
            generator.module.train(finetune=True)
            IsTrain=False

        ground_truth, mask, edge, gray_image = next(image_data_loader)

        if is_cuda:
            ground_truth, mask, edge, gray_image = ground_truth.cuda(), mask.cuda(), edge.cuda(), gray_image.cuda()

        input_image, input_edge, input_gray_image = ground_truth * mask, edge * mask, gray_image * mask

        # ---------
        # Auxiliary Parameter
        # ---------
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        generator_.zero_grad()
        with torch.no_grad():
            plw,slw = a(0)
        output, projected_image, projected_edge = generator(input_image,
                                                            torch.cat((input_edge, input_gray_image), dim=1), mask)
        comp = ground_truth * mask + output * (1 - mask)
        output_pred, output_edge = discriminator(output, gray_image, edge, is_real=False)
        vgg_comp, vgg_output, vgg_ground_truth = extractor(comp), extractor(output), extractor(ground_truth)

        generator_loss_dict = weighted_generator_loss_func(
            mask, output, ground_truth, edge, output_pred,
            vgg_comp, vgg_output, vgg_ground_truth,
            projected_image, projected_edge,
            output_edge,plw,slw
        )

        generator_loss = generator_loss_dict['loss_hole'] * opts.HOLE_LOSS + \
                         generator_loss_dict['loss_valid'] * opts.VALID_LOSS + \
                         generator_loss_dict['loss_perceptual'] * opts.PERCEPTUAL_LOSS + \
                         generator_loss_dict['loss_style'] * opts.STYLE_LOSS + \
                         generator_loss_dict['loss_adversarial'] * opts.ADVERSARIAL_LOSS + \
                         generator_loss_dict['loss_intermediate'] * opts.INTERMEDIATE_LOSS
        lp,ls = generator_loss_dict['loss_perceptual_terms'],generator_loss_dict['loss_style_terms']
        grads = torch.autograd.grad(generator_loss.mean(), generator.parameters(), retain_graph=True)
        grads_lp = [torch.autograd.grad(lpi.mean(), generator.module.out_layer.parameters(), retain_graph=True) for lpi in lp]
        grads_ls = [torch.autograd.grad(lsi.mean(), generator.module.out_layer.parameters(), retain_graph=True) for lsi in ls]
        # print(lr)
        generator_.module.load_state_dict(tools.one_step_update(generator.module, lr, grads))

        items = next(image_data_loader2)

        if is_cuda:
            items = [_.cuda() for _ in items]
        ground_truth2, mask2, edge2, gray_image2 = items
        input_image2, input_edge2, input_gray_image2 = ground_truth2 * mask2, edge2 * mask2, gray_image2 * mask2

        output, projected_image, projected_edge = generator_(input_image2,
                                                            torch.cat((input_edge2, input_gray_image2), dim=1), mask2)
        comp = ground_truth * mask + output * (1 - mask)
        lpips_loss = lpips(ground_truth,comp).mean()
        grads_ = torch.autograd.grad(lpips_loss.mean(), generator_.module.out_layer.parameters(), retain_graph=False)
        wperc, wstyl = a(0)
        loss_wperc = 0.0
        for j in range(len(grads_lp)): loss_wperc += wperc[j] * (-tools.param_grad_dot(grads_, grads_lp[j]))
        loss_wstyl = 0.0
        for j in range(len(grads_lp)): loss_wstyl += wstyl[j] * (-tools.param_grad_dot(grads_, grads_ls[j]))
        loss_A = loss_wperc + loss_wstyl
        a_optim.zero_grad()
        loss_A.backward()
        a_optim.step()

        # ---------
        # Generator
        # ---------
        with torch.no_grad():
            plw,slw = a(0)
        output, projected_image, projected_edge = generator(input_image,
                                                            torch.cat((input_edge, input_gray_image), dim=1), mask)
        comp = ground_truth * mask + output * (1 - mask)

        output_pred, output_edge = discriminator(output, gray_image, edge, is_real=False)

        vgg_comp, vgg_output, vgg_ground_truth = extractor(comp), extractor(output), extractor(ground_truth)

        generator_loss_dict = weighted_generator_loss_func(
            mask, output, ground_truth, edge, output_pred,
            vgg_comp, vgg_output, vgg_ground_truth,
            projected_image, projected_edge,
            output_edge,plw,slw
        )
        generator_loss = generator_loss_dict['loss_hole'] * opts.HOLE_LOSS + \
                         generator_loss_dict['loss_valid'] * opts.VALID_LOSS + \
                         generator_loss_dict['loss_perceptual'] * opts.PERCEPTUAL_LOSS + \
                         generator_loss_dict['loss_style'] * opts.STYLE_LOSS + \
                         generator_loss_dict['loss_adversarial'] * opts.ADVERSARIAL_LOSS + \
                         generator_loss_dict['loss_intermediate'] * opts.INTERMEDIATE_LOSS
        generator_loss_dict['loss_joint'] = generator_loss
        generator_loss_dict['l1'] = generator_loss_dict['loss_hole']+generator_loss_dict['loss_valid']
        generator_loss_dict['lpips'] = lpips_loss
        generator_optim.zero_grad()
        generator_loss.backward()
        generator_optim.step()

        # -------------
        # Discriminator
        # -------------
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        real_pred, real_pred_edge = discriminator(ground_truth, gray_image, edge, is_real=True)
        fake_pred, fake_pred_edge = discriminator(output.detach(), gray_image, edge, is_real=False)

        discriminator_loss_dict = discriminator_loss_func(real_pred, fake_pred, real_pred_edge, fake_pred_edge, edge)
        discriminator_loss = discriminator_loss_dict['loss_adversarial']
        discriminator_loss_dict['loss_joint'] = discriminator_loss

        discriminator_optim.zero_grad()
        discriminator_loss.backward()
        discriminator_optim.step()

        # ---
        # log
        # ---
        generator_loss_dict.pop('loss_perceptual_terms')
        generator_loss_dict.pop('loss_style_terms')
        generator_loss_dict.pop('loss_hole')
        generator_loss_dict.pop('loss_valid')
        generator_loss_dict.pop('loss_style')
        generator_loss_dict.pop('loss_perceptual')
        if opts.distributed:
            generator_loss_dict_reduced, discriminator_loss_dict_reduced = reduce_loss_dict(
                generator_loss_dict), reduce_loss_dict(discriminator_loss_dict)
        else:
            generator_loss_dict_reduced, discriminator_loss_dict_reduced = generator_loss_dict, discriminator_loss_dict
        pbar_l1= generator_loss_dict_reduced['l1'].mean().item()
        pbar_lpips= generator_loss_dict_reduced['lpips'].mean().item()

        pbar_d_loss_joint = discriminator_loss_dict_reduced['loss_joint'].mean().item()
        if get_rank() == 0:
            with open(os.path.join(opts.log_dir, 'weights.txt'), 'a') as file:
                with torch.no_grad(): wperc, wstyl = a(0)
                wperc = np.around(wperc.cpu().numpy(), 4)
                wstyl = np.around(wstyl.cpu().numpy(), 4)
                file.write('{} {} \n'.format(wperc, wstyl))
            with open(os.path.join(opts.log_dir,'loss.txt'), 'a') as file:
                file.write('{} {} \n'.format(pbar_l1, pbar_lpips))
        if get_rank() == 0 :
            if i % 50==0:
                pbar.set_description((
                    f'l1: {pbar_l1:.4f} '
                    f'lpips: {pbar_lpips:.4f}'
                    f'd_loss_joint: {pbar_d_loss_joint:.4f}'
                ))
                print('wpl{} wsl{}'.format(wperc,wstyl))
            if i % 500 ==0 or i ==1:
                with torch.no_grad():
                    save_img('{}/{}.png'.format(os.path.join(opts.log_dir, 'img'),i),
                                  ground_truth, input_image, output
                                  )
            if i % opts.save_interval == 0:
                print('saving')
                torch.save(
                    {
                        'n_iter': i,
                        'generator': generator_module.state_dict(),
                        'aux':a_module.state_dict(),
                        'discriminator': discriminator_module.state_dict(),
                        'optm_g':generator_optim.state_dict(),
                        'optm_d':discriminator_optim.state_dict(),
                        'optm_a':a_optim.state_dict()
                    },
                    f"{opts.save_dir}/{str(i).zfill(6)}.pt",
                )

def save_img(name, *args):
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