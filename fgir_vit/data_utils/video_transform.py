import random

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2


class ClipRandAugment(nn.Module):
    # timm RandAugment before the crop, as Frame2Freq; one random draw per clip
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transforms = {}

    def forward(self, frames):
        h, w = frames.shape[-2:]
        if (h, w) not in self.transforms:
            from PIL import Image
            from timm.data.auto_augment import rand_augment_transform
            hparams = dict(translate_const=int(min(h, w) * 0.45), interpolation=Image.BICUBIC)
            self.transforms[h, w] = rand_augment_transform(self.config, hparams)
        state = random.getstate()
        np_state = np.random.get_state()
        out = []
        for frame in frames:
            random.setstate(state)
            np.random.set_state(np_state)
            image = self.transforms[h, w](v2.functional.to_pil_image(frame))
            out.append(v2.functional.pil_to_tensor(image))
        return torch.stack(out)


class VideoTransform(nn.Module):
    # standard_transform on uint8 frames; one random draw per clip
    def __init__(self, args, is_train, mean, std):
        super().__init__()
        image_size = args.image_size
        resize_size = args.resize_size

        t = []
        if is_train:
            if getattr(args, 'rand_aug_timm', None):
                t.append(ClipRandAugment(args.rand_aug_timm))
            if args.affine:
                t.append(v2.Resize((resize_size, resize_size)))
                t.append(v2.RandomAffine(degrees=15, scale=(0.85, 1.15)))
                t.append(v2.RandomCrop((image_size, image_size)))
            elif args.train_resize_directly:
                t.append(v2.Resize((image_size, image_size)))
            elif args.random_resized_crop:
                t.append(v2.RandomResizedCrop((image_size, image_size)))
            elif args.square_resize_random_crop:
                t.append(v2.Resize((resize_size, resize_size)))
                t.append(v2.RandomCrop((image_size, image_size)))
            elif args.short_side_resize_random_crop:
                t.append(v2.Resize(resize_size))
                t.append(v2.RandomCrop((image_size, image_size)))
            elif args.square_center_crop:
                t.append(v2.Resize((resize_size, resize_size)))
                t.append(v2.CenterCrop((image_size, image_size)))

            if args.rand_aug:
                t.append(v2.RandAugment())
            if args.horizontal_flip:
                t.append(v2.RandomHorizontalFlip())
            if args.vertical_flip:
                t.append(v2.RandomVerticalFlip())
            if args.jitter_prob > 0:
                t.append(v2.RandomApply([v2.ColorJitter(args.jitter_bcs, args.jitter_bcs, args.jitter_bcs,
                                                        args.jitter_hue)], p=args.jitter_prob))
            if args.greyscale > 0:
                t.append(v2.RandomGrayscale(p=args.greyscale))
            if args.blur > 0:
                kernel_size = image_size // 20 * 2 + 1
                t.append(v2.RandomApply([v2.GaussianBlur(kernel_size, (0.1, 2.0))], p=args.blur))
        elif args.test_resize_directly:
            t.append(v2.Resize((image_size, image_size)))
        else:
            t.append(v2.Resize((args.test_resize_size, args.test_resize_size)))
            t.append(v2.CenterCrop((image_size, image_size)))

        t.append(v2.ToDtype(torch.float32, scale=True))
        t.append(v2.Normalize(mean=mean, std=std))
        if is_train and args.re > 0:
            t.append(v2.RandomErasing(p=args.re, scale=(0.02, args.re_sh), ratio=(args.re_r1, 3.3)))

        self.transform = v2.Compose(t)
        print(self.transform)

    def forward(self, clip):
        # frames as the batch axis
        return self.transform(clip.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
