import os
import time
import random

import wandb
from timm.optim import create_optimizer
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from fgir_vit.data_utils.build_dataloaders import build_dataloaders
from fgir_vit.model_utils.build_model import build_model
from fgir_vit.other_utils.build_args import parse_train_args
from fgir_vit.train_utils.misc_utils import summary_stats, stats_test, set_random_seed, computation_stats, max_memory_gb
from fgir_vit.train_utils.scheduler import build_scheduler
from fgir_vit.train_utils.trainer import Trainer
from fgir_vit.train_utils.calc_loss import OverallLoss


def adjust_args_general(args):
    classifier = f'_{args.classifier}' if args.classifier else ''
    if args.adapter and getattr(args, 'adapter_init', 'xavier') == 'zero' and getattr(args, 'adapter_scale', 1.0) != 1.0:
        adapter = '_adaptformer'
    else:
        adapter = f'_{args.adapter}' if args.adapter else ''
    freeze = '_fz' if args.freeze_backbone else ''

    args.run_name = '{}_{}{}{}{}_{}'.format(
        args.dataset_name, args.model_name, classifier, adapter, freeze, args.serial
    )

    args.results_dir = os.path.join(args.results_dir, args.run_name)
    return args


def build_environment(args):
    if args.serial is None:
        args.serial = random.randint(0, 1000)
    # Set device and random seed
    set_random_seed(args.seed, numpy=args.seed_numpy)

    # dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(args)

    # model and criterion
    model = build_model(args)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    model.zero_grad()

    criterion = OverallLoss(args)

    # loss and optimizer
    optimizer = create_optimizer(args, model)
    lr_scheduler = build_scheduler(args, optimizer, train_loader)

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    if not args.ckpt_path or args.transfer_learning:
        adjust_args_general(args)
    os.makedirs(args.results_dir, exist_ok=True)

    return model, criterion, optimizer, lr_scheduler, train_loader, val_loader, test_loader


def main():
    time_start = time.time()

    args = parse_train_args()

    model, criterion, optimizer, lr_scheduler, train_loader, val_loader, test_loader = build_environment(args)

    trainer = Trainer(args, model, criterion, optimizer, lr_scheduler,
                      train_loader, val_loader, test_loader)

    if not args.test_only:
        flops, flops_train, max_memory, no_params, no_params_trainable, no_params_deployable = computation_stats(
            model, args.image_size, args.device, args.n_frames if args.video else 0)

    if args.set_seed_all_before_train_loop:
        set_random_seed(args.seed, numpy=args.seed_numpy)

    if args.test_only:
        if args.wandb_log:
            wandb.init(config=args, project=args.project_name, entity=args.entity)
            wandb.run.name = args.run_name
        time_start = time.time()
        print(args, model.cfg)

        test_acc, class_deviation = trainer.test()

        if not args.debugging:
            time_total = time.time() - time_start
            # counted after testing: the train-mode FLOPs pass updates BatchNorm running statistics
            flops, _, max_memory, no_params, no_params_trainable, _ = computation_stats(
                model, args.image_size, args.device, args.n_frames if args.video else 0)

            if args.test_multiple:
                num_images = (args.test_multiple + 1) * args.num_images_test
            else:
                num_images = args.num_images_test

            stats_test(test_acc, class_deviation, flops, max_memory, no_params, no_params_trainable,
                       time_total, num_images, not args.wandb_log)
            if args.wandb_log:
                wandb.finish()
    else:
        if args.local_rank == 0:
            if args.wandb_log:
                wandb.init(config=args, project=args.project_name, entity=args.entity)
                wandb.run.name = args.run_name
                wandb.define_metric('epoch')
                wandb.define_metric('train_*', step_metric='epoch')
                wandb.define_metric('val_*', step_metric='epoch')
            if not args.distributed:
                print(model, model.cfg)
            print(args)

        best_acc, best_epoch, class_deviation = trainer.train()

        # summary stats
        if args.local_rank == 0 and not args.debugging:
            time_total = time.time() - time_start
            max_memory = max_memory_gb()
            summary_stats(args.epochs, time_total, best_acc, best_epoch, flops, max_memory,
                          no_params, no_params_trainable, class_deviation, not args.wandb_log)
            if args.wandb_log:
                wandb.finish()


if __name__ == '__main__':
    main()
