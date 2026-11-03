from types import SimpleNamespace

import timm
import torch
import torch.nn as nn

from .vit import ViT
from .misc import ViTConfig, TEMPORAL_ADAPTERS
from .timm_vit import *


VITS = (
    'vit_t4', 'vit_t8', 'vit_t16', 'vit_t32', 'vit_s8', 'vit_s16', 'vit_s32',
    'vit_b8', 'vit_b16', 'vit_b32', 'vit_l16', 'vit_l32', 'vit_h14')


def build_model(args):
    # initiates model and loss
    if getattr(args, 'video', None) and args.video_model == 'joint' and args.adapter in TEMPORAL_ADAPTERS:
        # temporal adapters reshape (b t) n d: only the per-frame blocks before the merge take one
        args.adapter_depth = args.merge_depth

    if args.model_name in VITS:
        model = VisionTransformer(args)
    elif args.model_name in timm.list_models(pretrained=True):
        model = TIMMBackbone(args)
    else:
        raise NotImplementedError

    if getattr(args, 'video', None):
        from .video_model import VideoModel
        model = VideoModel(args, model)

    if getattr(args, 'fused_attn', False):
        for m in model.modules():
            if hasattr(m, 'fused_attn'):
                m.fused_attn = True

    args.seq_len_post_reducer = model.cfg.seq_len_post_reducer

    if args.ckpt_path:
        state_dict = torch.load(
            args.ckpt_path, map_location=torch.device('cpu'), weights_only=False)['model']

        expected_missing_keys = []
        if args.transfer_learning:
            # modifications to load partial state dict
            if ('model.head.weight' in state_dict):
                expected_missing_keys += ['model.head.weight', 'model.head.bias']
            elif ('model.head.head.weight' in state_dict):
                expected_missing_keys += ['model.head.head.weight', 'model.head.head.bias']
            elif getattr(args, 'video', None):
                head = 'head.classifier' if 'head.classifier.weight' in state_dict else 'head'
                expected_missing_keys += [f'{head}.weight', f'{head}.bias']

            for key in expected_missing_keys:
                state_dict.pop(key)

        ret = model.load_state_dict(state_dict, strict=False)

        print('''Missing keys when loading pretrained weights: {}
              Expected missing keys: {}'''.format(ret.missing_keys, expected_missing_keys))
        print('Unexpected keys when loading pretrained weights: {}'.format(
            ret.unexpected_keys))

        print('Loaded from custom checkpoint.')

    if args.freeze_backbone:
        freeze_backbone(args, model)

    if args.distributed:
        model.cuda()
    else:
        model.to(args.device)

    print(f'Initialized classifier: {args.model_name}')
    return model


def freeze_backbone(args, model):
    keywords = ['head', 'aggregator', 'adapter', 'lora']

    if args.unfreeze_cls:
        keywords.append('class_token')
        keywords.append('cls_token')
    if args.unfreeze_positional_embedding:
        keywords.append('positional_embedding')
    if args.unfreeze_encoder_norm:
        keywords.append('encoder_norm')
    if args.unfreeze_mhsa_proj:
        keywords.append('proj')
    if args.unfreeze_encoder_last_block:
        keywords.append(str(args.num_hidden_layers - 1))

    for name, param in model.named_parameters():
        if any(kw in name for kw in keywords):
            param.requires_grad = True
            print('Unfrozen: ', name)
        else:
            param.requires_grad = False

    print('Total parameters (M): ', sum([p.numel() for p in model.parameters()]) / (1e6))
    print('Trainable parameters (M): ', sum([p.numel() for p in model.parameters() if p.requires_grad]) / (1e6))


class VisionTransformer(nn.Module):
    def __init__(self, args):
        super(VisionTransformer, self).__init__()
        # init default config
        cfg = ViTConfig(model_name=args.model_name)
        # modify config if given an arg otherwise keep defaults
        args_temp = vars(args)
        for k, v in args_temp.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v if v is not None else getattr(cfg, k))
        cfg.calc_dims()
        # update the args with the final model config
        for attribute in vars(cfg):
            if hasattr(args, attribute):
                setattr(args, attribute, getattr(cfg, attribute))
        # init model
        self.model = ViT(cfg, pretrained=args.pretrained)
        self.cfg = cfg

    def forward(self, images, targets=None):
        out = self.model(images)
        return out


class TIMMBackbone(nn.Module):
    def __init__(self, args):
        super(TIMMBackbone, self).__init__()
        # init default config

        try:
            self.model = timm.create_model(
                args.model_name, pretrained=args.pretrained, num_classes=args.num_classes,
                img_size=args.image_size, drop_path_rate=args.sd, args=args)
        except TypeError:
            self.model = timm.create_model(
                args.model_name, pretrained=args.pretrained, num_classes=args.num_classes,
                drop_path_rate=args.sd)

        # self.model.reset_classifier(args.num_classes)

        self.cfg = SimpleNamespace(**{'seq_len_post_reducer': args.image_size // 16})

    def forward(self, images, targets=None):
        x = self.model(images)
        return x
