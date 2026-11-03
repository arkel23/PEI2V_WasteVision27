def check_args(args):
    # option combinations that fail or are silently ignored, one assert each; args are final (configs applied)
    from .build_args import VITS

    assert not (args.distributed and args.test_only), 'test_only cannot be used with multi gpu'

    assert args.video or not args.qk_lora, '--qk_lora needs --video'
    if args.video:
        assert not (args.cm or args.mu or args.vis_errors or args.save_images_wandb), \
            'video clips do not support cutmix/mixup, vis_errors or save_images_wandb'

    # 2 is the --warmup_epochs default, which every run that keeps --warmup_steps carries
    assert args.t_in_epochs or not args.warmup_steps or args.warmup_epochs in (0, 2), \
        '--warmup_epochs is ignored while --warmup_steps is nonzero: pass --warmup_steps 0'
    assert args.video or not args.rand_aug_timm, '--rand_aug_timm applies to video clips only; images take --rand_aug'
    assert not args.head_dropout or (args.video and args.video_model == 'joint'), \
        '--head_dropout applies to the joint video head only'
    assert args.model_name not in VITS or args.adapter_init == 'xavier', \
        f'--adapter_init is ignored by {args.model_name}: only PEViT adapters read it'
    assert args.model_name not in VITS or args.adapter_scale == 1.0 or args.adapter == 'aim', \
        f'--adapter_scale is ignored by {args.model_name} unless --adapter aim'

