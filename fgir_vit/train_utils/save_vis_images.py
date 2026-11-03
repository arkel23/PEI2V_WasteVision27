import sys
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
from einops import rearrange
import wandb

from ..data_utils.build_transform import MEANS, STDS


def inverse_normalize(tensor, norm_custom=False):
    if norm_custom == 'clip':
        mean, std = MEANS['clip'], STDS['clip']
    elif norm_custom:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def vis_images(args, curr_img, img, title=None):
    img = rearrange(img, '1 c h w -> h w c')

    img = inverse_normalize(img.data, args.custom_mean_std)
    img = np.uint8(np.clip(img.cpu().numpy(), 0, 1))

    fig = plt.figure()
    axs = fig.add_subplot(111)

    axs.imshow(img)
    if title is not None:
        axs.set_title(title, fontsize=8, wrap=True)

    axs.axis('off')
    fig.tight_layout()

    if args.vis_errors_save:
        fp = osp.join(args.results_dir, f'test_{curr_img}.png')
        fig.savefig(fp, dpi=300)
    else:
        plt.show(block=False)

        inp = input("input 'exit' to stop visualizing: ")
        if inp == 'exit':
            sys.exit()

    plt.close()
    return 0


def save_samples(images, output, train, curr_iter, saved, args):
    img_sz = args.image_size

    vis_cols = getattr(args, 'vis_cols', None)

    if args.save_images and train and (curr_iter % args.save_images == 0):
        save_images(images, osp.join(args.results_dir, f'{curr_iter}.png'), img_sz, args.custom_mean_std, vis_cols)

    elif args.save_images and not train and not saved:
        save_images(images, osp.join(args.results_dir, f'test.png'), img_sz, args.custom_mean_std, vis_cols)

        if args.wandb_log:
            if args.save_images_wandb:
                wandb.log({'crops': wandb.Image(images)}, step=curr_iter)

        return True
    
    if saved:
        return True
    return False


def save_images(images, fp, img_sz, custom_mean_std, vis_cols=None):
    with torch.no_grad():
        if images.dim() == 5:
            # clips: each clip's frames side by side, vis_cols clips per row
            ncols = (vis_cols or 1) * images.shape[2]
            images = rearrange(images, 'b c t h w -> (b t) c h w')
        else:
            ncols = vis_cols if vis_cols else int(images.shape[0] ** 0.5)
        samples = inverse_normalize(images.data, custom_mean_std)

        # nrow is the number of images per row (so the number of columns)
        save_image(samples, fp, nrow=ncols, padding=2)

    return 0
