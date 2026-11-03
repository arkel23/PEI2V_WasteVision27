import os
import argparse
import torch

from .config import adjust_config
from .check_args import check_args


VITS = ['vit_t4', 'vit_t8', 'vit_t16', 'vit_t32', 'vit_s8', 'vit_s16', 'vit_s32',
        'vit_b8', 'vit_b16', 'vit_b32', 'vit_l16', 'vit_l32', 'vit_h14']
HEADS = (None, 'cls')


def add_adjust_common_dependent(args):
    if args.vis_errors_save:
        args.vis_errors = True

    if args.vis_errors:
        args.test_only_bs1 = True

    if args.test_only_bs1:
        print('When using test only changes batch size to 1 to simulate streaming')
        args.test_only = True
        args.batch_size = 1

    args.effective_batch_size = args.batch_size * args.gradient_accumulation_steps

    if args.base_lr:
        args.lr = args.base_lr * (args.effective_batch_size / 8)

    # distributed
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        args.device = torch.device(f'cuda:{args.local_rank}')
        torch.cuda.set_device(args.device)

        args.effective_batch_size = args.effective_batch_size * args.world_size

        if args.base_lr:
            args.lr = args.base_lr * (args.effective_batch_size / 8)

    if not args.resize_size:
        if args.image_size == 32:
            args.resize_size = 36
        elif args.image_size == 64:
            args.resize_size = 72
        elif args.dataset_name == 'ncfm':
            args.resize_size = args.image_size + 32
        else:
            args.resize_size = int(args.image_size * 1.34)

    if not args.test_resize_size:
        args.test_resize_size = args.resize_size

    if args.dataset_name and ('vtab' in args.dataset_name):
        args.train_resize_directly = True
        args.warmup_steps = 0
        args.warmup_epochs = 10
        args.horizontal_flip = False

    if not (args.train_resize_directly or args.random_resized_crop or args.square_resize_random_crop or
            args.short_side_resize_random_crop or args.square_center_crop):
        print('Needs at least one crop, using square_resize_random_crop by default')
        args.square_resize_random_crop = True

    check_args(args)

    args.wandb_log = not (args.debugging or args.offline or args.vis_errors)

    return args


def add_common_args():
    parser = argparse.ArgumentParser('Arguments for code: FGIR-ViT')
    # general
    parser.add_argument('--project_name', type=str, default='PEI2V_WasteVision27',
                        help='project name for wandb')
    parser.add_argument('--entity', type=str, default='nycu_pcs', help='wandb entity')
    parser.add_argument('--debugging', action='store_true',
                        help='when true disables wandb and exits after a single pass')
    parser.add_argument('--serial', default=0, type=int, help='serial number for run')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--seed_numpy', action='store_true', help='also seed numpy (Mixup and CutMix draws)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--log_freq', type=int,
                        default=100, help='print frequency (iters)')
    parser.add_argument('--save_freq', type=int,
                        default=200, help='save frequency (epochs)')
    parser.add_argument('--results_dir', type=str, default='results_train',
                        help='dir to save models from base_path')
    parser.add_argument('--per_class_acc_results', type=str, default='per_class_acc.csv')
    parser.add_argument('--ind_preds_results', type=str, default='ind_preds.csv')
    return parser


def add_data_args(parser):
    parser.add_argument('--image_size', type=int, default=224, help='image_size')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--cpu_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--custom_mean_std', nargs='?', const=True, default=False,
                        help='custom mean/std: alone the dataset\'s (0.5 if none), clip for CLIP\'s')
    parser.add_argument('--dataset_in_memory', action='store_true',
                        help='load whole dataset into RAM to reduce reads from storage')
    parser.add_argument('--pin_memory', action='store_false', help='pin memory for gpu (def: true)')
    # dataset
    parser.add_argument('--shuffle_test', action='store_true',
                        help='if true then shuffles test data')
    parser.add_argument('--dataset_name', default=None, type=str, help='dataset name')
    parser.add_argument('--dataset_root_path', type=str, default=None,
                        help='the root directory for where the data/feature/label files are')
    # folders with images (can be same: those where it's all stored in 'data')
    parser.add_argument('--folder_train', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/train/')
    parser.add_argument('--folder_val', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/val/')
    parser.add_argument('--folder_test', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/test/')
    # df files with img_dir, class_id
    parser.add_argument('--df_train', type=str, default='train.csv',
                        help='the df csv with img_dirs, targets, def: train.csv')
    parser.add_argument('--df_trainval', type=str, default='train_val.csv',
                        help='the df csv with img_dirs, targets, def: train_val.csv')
    parser.add_argument('--df_val', type=str, default='val.csv',
                        help='the df csv with img_dirs, targets, def: val.csv')
    parser.add_argument('--df_test', type=str, default='test.csv',
                        help='the df csv with img_dirs, targets, root/test.csv')
    parser.add_argument('--train_subset', type=float, default=None,
                        help='part of each class to train on: a fraction if below 1, else a number of samples')
    parser.add_argument('--df_classid_classname', type=str, default='classid_classname.csv',
                        help='the df csv with classnames and class ids, root/classid_classname.csv')
    # video: each csv row is a clip, `dir` a list of frame paths or an mp4 name
    parser.add_argument('--video', type=str, default=None, choices=(None, 'frames', 'mp4'),
                        help='clips stored as frame images or as mp4 files')
    parser.add_argument('--n_frames', type=int, default=16, help='frames sampled per clip')
    parser.add_argument('--temporal_subsample', type=str, default='uniform',
                        choices=('uniform', 'center', 'random', 'stride', 'segment'),
                        help='uniform, a centered or random chunk, a fixed stride, or one frame per segment')
    parser.add_argument('--temporal_stride', type=int, default=2,
                        help='stride for --temporal_subsample stride; short clips repeat their last frame')
    return parser


def add_optim_scheduler_args(parser):
    # optimizer
    parser.add_argument('--opt', default='sgd', type=str,
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--base_lr', type=float, default=None,
                        help='base_lr if using scaling lr (lr = base_lr * effective batch size / 8)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--opt_eps', default=1e-8, type=float,
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    # fp 16 stability and gradient accumulation
    parser.add_argument('--fp16', action='store_false', help='use fp16 (on by default)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate before backward/update pass.')
    # lr scheduler
    parser.add_argument('--sched', default='cosine', type=str,
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--t_in_epochs', action='store_true',
                        help='update per epoch (instead of per iter)')
    parser.add_argument('--lr_noise', type=float, nargs='+', default=None,
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr_noise_pct', type=float, default=0.67,
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr_noise_std', type=float, default=1.0,
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6,
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay_epochs', type=float, default=30,
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='steps to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown_epochs', type=int, default=5,
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience_epochs', type=int, default=10,
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1,
                        help='LR decay rate (default: 0.1)')
    return parser


def add_vit_args(parser):
    # default config is based on vit_b16
    parser.add_argument('--patch_stride', type=int,
                        help='patch stride for patchifier, e.g. 16 for vit_b16')
    parser.add_argument('--classifier', type=str, default=None, choices=HEADS)

    # encoder related
    parser.add_argument('--attention', type=str,
                        help='use global attention or a mix of locan and global')
    parser.add_argument('--pos_embedding_type', type=str, default=None,
                        help='positional embedding for encoder, def: learned')
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument('--intermediate_size', type=int, default=None)
    parser.add_argument('--num_attention_heads', type=int, default=None)
    parser.add_argument('--num_hidden_layers', type=int, default=None)
    parser.add_argument('--encoder_norm', action='store_false',
                        help='norm after encoder (def: true)')
    # transformers in general
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=None)
    parser.add_argument('--hidden_dropout_prob', type=float, default=None)
    parser.add_argument('--layer_norm_eps', type=float, default=None)
    parser.add_argument('--hidden_act', type=str, default=None)
    parser.add_argument('--fused_attn', action='store_true',
                        help='attention with F.scaled_dot_product_attention')
    # video models
    parser.add_argument('--video_model', type=str, default='temporal', choices=('temporal', 'joint'),
                        help='per-frame backbone + temporal transformer, or frames merged into one sequence')
    parser.add_argument('--temporal_dim', type=int, default=None,
                        help='width of the temporal transformer; default: the backbone width')
    parser.add_argument('--temporal_embedding', type=str, default='learned',
                        choices=('learned', 'learned_zero', 'sincos', 'none', 'tconv'),
                        help='temporal position embedding of the temporal head and the joint merge')
    parser.add_argument('--temporal_depth', type=int, default=1, help='temporal: blocks in the temporal head')
    parser.add_argument('--merge_depth', type=int, default=0,
                        help='joint: blocks before this index run per frame, the rest on all frames')
    parser.add_argument('--video_cls', type=str, default='mean',
                        choices=('mean', 'concat', 'video', 'video_mean'),
                        help='joint: head input from the frame class tokens, a video token, or both')
    parser.add_argument('--head_dropout', type=float, default=0.0,
                        help='joint: dropout before the classifier')
    parser.add_argument('--qk_lora', type=int, default=0,
                        help='rank of zero-init lora on query and key (0: off); joint: from --merge_depth on')
    return parser


def add_fgir_args(parser):
    # frozen backbone
    parser.add_argument('--freeze_backbone', action='store_true')

    parser.add_argument('--adapter', type=str, default=None,
                        help='adapter module inside transformer blocks; temporal: frame2freq_ms, frame2freq_st '
                             '(_og: as released), st_adapter, aim')
    parser.add_argument('--adapter_dim', type=int, default=8,
                        help='adapter module inside transformer blocks')
    parser.add_argument('--adapter_kernel_size', type=int, default=3,
                        help='st_adapter: temporal kernel of its (k, 1, 1) depthwise conv')
    parser.add_argument('--adapter_path', type=str, default='convpass')
    parser.add_argument('--adapter_init', type=str, default='xavier', choices=('xavier', 'zero'))
    parser.add_argument('--adapter_scale', type=float, default=1.0,
                        help='adaptformer scale, learnable from abs(value) if negative; aim: mlp adapter scale')
    parser.add_argument('--adapter_pre_attn', action='store_true',
                        help='frame2freq, st_adapter: second adapter before mhsa; aim: before temporal mhsa')

    parser.add_argument('--unfreeze_cls', action='store_false')
    parser.add_argument('--unfreeze_positional_embedding', action='store_true')
    parser.add_argument('--unfreeze_mhsa_proj', action='store_true')
    parser.add_argument('--unfreeze_encoder_norm', action='store_true')
    parser.add_argument('--unfreeze_encoder_last_block', action='store_true')

    return parser


def add_augmentation_args(parser):
    # cropping
    parser.add_argument('--resize_size', type=int, default=None, help='resize_size before cropping')
    parser.add_argument('--test_resize_size', type=int, default=None, help='test resize_size before cropping')
    parser.add_argument('--test_resize_directly', action='store_true',
                        help='resizes directly to target image size instead of to larger then center crop')
    parser.add_argument('--train_resize_directly', action='store_true')
    parser.add_argument('--random_resized_crop', action='store_true',
                        help='crop random aspect ratio then resize to square')
    parser.add_argument('--square_resize_random_crop', action='store_true',
                        help='resize first to square then random crop')
    parser.add_argument('--short_side_resize_random_crop', action='store_true',
                        help='resize first so short side is resize_size then random crop a square')
    parser.add_argument('--square_center_crop', action='store_true',
                        help='resize first to square then center crop when training')
    # https://github.com/ArdhenduBehera/cap/blob/main/image_datagenerator.py
    parser.add_argument('--affine', action='store_true',
                        help='affine transform as in CAP')
    # flips
    parser.add_argument('--horizontal_flip', action='store_true',
                        help='use horizontal flip when training (on by default)')
    parser.add_argument('--vertical_flip', action='store_true',
                        help='use vertical flip (off by default)')
    # augmentation policies
    parser.add_argument('--auto_aug', action='store_true', help='Auto augmentation used')
    parser.add_argument('--rand_aug', action='store_true', help='RandAugment augmentation used')
    parser.add_argument('--rand_aug_timm', type=str, default=None,
                        help='timm randaugment per clip before the crop, e.g. rand-m7-n4-mstd0.5-inc1')
    parser.add_argument('--trivial_aug', action='store_true', help='use trivialaugmentwide')
    # color and distortion
    parser.add_argument('--jitter_prob', type=float, default=0.0,
                        help='color jitter probability of applying (0.8 for simclr)')
    parser.add_argument('--jitter_bcs', type=float, default=0.4,
                        help='color jitter brightness contrast saturation (0.4 for simclr)')
    parser.add_argument('--jitter_hue', type=float, default=0.1,
                        help='color jitter hue value (0.1 for simclr)')
    parser.add_argument('--blur', type=float, default=0.0,
                        help='gaussian blur probability (0.5 for simclr)')
    parser.add_argument('--greyscale', type=float, default=0.0,
                        help='gaussian blur probability (0.2 for simclr)')
    parser.add_argument('--solarize_prob', type=float, default=0.0,
                        help='solarize transform probability (0.2 for byol if image_size>32)')
    parser.add_argument('--solarize', type=int, default=128,
                        help='solarize pixels with higher value than (def: 128)')
    # cutmix, mixup
    parser.add_argument('--cm', action='store_true', help='Use Cutmix')
    parser.add_argument('--beta', default=1.0, type=float,
                        help='hyperparameter beta (default: 1)')
    parser.add_argument('--mu', action='store_true', help='Use Mixup')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--mix_prob', default=0.5, type=float,
                        help='mixup probability')
    # random erasing
    parser.add_argument('--re_mult', type=int, default=0)
    parser.add_argument('--re', default=0.0, type=float,
                        help='Random Erasing probability (def: 0.25)')
    parser.add_argument('--re_sh', default=0.4, type=float, help='max erasing area')
    parser.add_argument('--re_r1', default=0.3, type=float, help='aspect of erasing area')
    # regularization
    parser.add_argument('--ra', type=int, default=0, help='repeated augmentation (def: 3)')
    parser.add_argument('--sd', default=0.0, type=float,
                        help='rate of stochastic depth (def: 0.1)')
    parser.add_argument('--ls', action='store_true', help='label smoothing')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--focal_gamma', type=float, default=0.0,
                        help='gamma in focal loss (def: 2.0)')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='alpha in focal loss')
    return parser


def parse_train_args(ret_parser=False):
    parser = add_common_args()
    parser.add_argument('--train_trainval', action='store_false',
                        help='when true uses trainval for train and evaluates on test \
                        otherwise use train for train and evaluates on val')

    parser.add_argument('--set_seed_all_before_train_loop', action='store_false',
                        help='if true sets random seed (not np) again just before train loop')

    parser.add_argument('--save_images', type=int, default=10000,
                        help='save images every x iterations')
    parser.add_argument('--save_images_wandb', action='store_true')
    parser.add_argument('--save_optimizer', action='store_true',
                        help='include the optimizer state in saved checkpoints')
    parser.add_argument('--plot_gradients', action='store_true',
                        help='log gradient statistics every log_freq iterations')
    parser.add_argument('--bottom_k_acc', type=int, nargs='+', default=None,
                        help='log the mean accuracy of the k%% lowest-accuracy classes at the last epoch')
    parser.add_argument('--vis_cols', type=int, default=None,
                        help='how many columns when visualizing images')

    # test and evaluation modes
    parser.add_argument('--test_only', action='store_true',
                        help='when true skips training and evaluates model on test dataloader')
    parser.add_argument('--test_only_bs1', action='store_true',
                        help='same as test_only but forces bs 1 to emulate streaming/on-demand classification')
    parser.add_argument('--offline', action='store_true',
                        help='do not upload results to wandb')
    parser.add_argument('--test_multiple', type=int, default=5,
                        help='test multiple loops (to reduce model loading time influence)')
    parser.add_argument('--vis_errors', action='store_true',
                        help='when true shows prediction errors (turns on test_only by def)')
    parser.add_argument('--vis_errors_save', action='store_true',
                        help='when true saves prediction errors (turns on test_only by def)')
    parser.add_argument('--eval_freq', type=int, default=200, help='eval every x epochs')
    parser.add_argument('--top_k', type=int, default=2, help='for top-k acc metric')

    # models in general
    parser.add_argument('--model_name', type=str, default='vit_b16')
    parser.add_argument('--pretrained', action='store_true', help='pretrained model on imagenet')
    parser.add_argument('--ckpt_path', type=str, default=None, help='path to custom pretrained ckpt')
    parser.add_argument('--transfer_learning', action='store_true',
                        help='not load fc layer when using custom ckpt')

    # distributed
    parser.add_argument('--dist_eval', action='store_true',
                        help='validate using dist sampler (else do it on one gpu)')
    parser = add_data_args(parser)
    parser = add_optim_scheduler_args(parser)
    parser = add_augmentation_args(parser)
    parser = add_vit_args(parser)
    parser = add_fgir_args(parser)
    parser.add_argument("--cfg", type=str, nargs='+',
                        help='yaml configs merged left to right; command line wins')

    if ret_parser:
        return parser

    args = parser.parse_args()
    adjust_config(parser, args)
    args = add_adjust_common_dependent(args)

    return args


if __name__ == '__main__':
    args = parse_train_args()
    print(args)
