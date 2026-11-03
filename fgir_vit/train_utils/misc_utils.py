import random
import resource
from statistics import mean, stdev

import numpy as np
import torch
import wandb

from torchprofile import profile_macs


def computation_stats(model, image_size=224, device=torch.device('cpu'), n_frames=0):
    # if args.distributed:
    # flops = 0
    flops = count_flops(model, image_size, device, False, 'torchprofile', n_frames)
    flops_train = count_flops(model, image_size, device, True, 'torchprofile', n_frames)

    max_memory = max_memory_gb()

    no_params = count_params_single(model)
    no_params_trainable = count_params_trainable(model)
    no_params_deployable = count_params_deployable(model)

    return flops, flops_train, max_memory, no_params, no_params_trainable, no_params_deployable


def max_memory_gb():
    # peak VRAM reserved, or peak process RSS (KiB on Linux) without a GPU
    if torch.cuda.is_available():
        return torch.cuda.max_memory_reserved() / (1024 ** 3)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def count_flops(model, image_size=224, device='cuda', train=False, profiler='torchprofile', n_frames=0):
    # https://github.com/Lyken17/pytorch-OpCounter
    # https://github.com/sovrasov/flops-counter.pytorch
    # https://github.com/zhijian-liu/torchprofile
    # need to modify to return flops at both train and inference
    if train:
        model.train()
        bs = 2
    else:
        model.eval()
        bs = 1

    with torch.no_grad():
        # n_frames > 0: one clip (b c t h w) per sample, FLOPs per clip
        frames = (n_frames,) if n_frames else ()
        x = torch.rand(bs, 3, *frames, image_size, image_size).to(device)
        if profiler == 'torchprofile':
            try:
                flops = profile_macs(model, x)
            except Exception as e:
                print(f'FLOPs counting failed, reporting 0: {e}')
                flops = 0
        else:
            raise NotImplemented

        flops = round((flops / bs) / 1e9, 4)
    print(f'Train {train} FLOPs (G): {flops}')
    return flops


def count_params_module_list(module_list):
    return sum([count_params_single(model) for model in module_list])


def count_params_single(model):
    return sum([p.numel() for p in model.parameters()])


def count_params_trainable(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def count_params_deployable(model):
    params = []
    for name, module in model.named_modules():
        params.extend([p.numel() for p in module.parameters() if p.requires_grad])
    
    return sum(params)


def get_gradients(named_parameters):
    avg_grads, max_grads = [], []
    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None and 'bias' not in n:
            avg_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())

    return {
        'max_grad_all': max(max_grads),
        'mean_max_grad_layer': mean(max_grads),
        'std_max_grad_layer': stdev(max_grads) if len(max_grads) > 1 else 0,
        'mean_avg_grad_layer': mean(avg_grads),
        'std_avg_grad_layer': stdev(avg_grads) if len(avg_grads) > 1 else 0,
    }


def set_random_seed(seed=0, numpy=True):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if numpy:
        np.random.seed(seed)
    return 0


def summary_stats(epochs, time_total, best_acc, best_epoch, flops, max_memory,
                  no_params, no_params_trainable, class_deviation, offline=False):
    time_avg = round((time_total / epochs) / 60, 4)
    best_time = round(time_avg * best_epoch, 4)
    time_total = round(time_total / 60, 4)  # mins
    no_params = round(no_params / (1e6), 4)  # millions of parameters
    no_params_trainable = round(no_params_trainable / (1e6), 4)  # millions
    max_memory = round(max_memory, 4)

    print('''Total run time (minutes): {}
          Average time per epoch (minutes): {}
          Best accuracy (%): {} at epoch {}. Time to reach this (minutes): {}
          Class deviation: {}
          Max VRAM consumption (GB): {}
          Total number of parameters in all modules (M): {}
          Trainable number of parameters in all modules (M): {}
          '''.format(time_total, time_avg, best_acc, best_epoch, best_time,
                     class_deviation, max_memory, no_params, no_params_trainable))

    if not offline:
        wandb.run.summary['time_total'] = time_total
        wandb.run.summary['time_avg'] = time_avg
        wandb.run.summary['best_acc'] = best_acc
        wandb.run.summary['best_epoch'] = best_epoch
        wandb.run.summary['best_time'] = best_time
        wandb.run.summary['flops'] = flops
        wandb.run.summary['max_memory'] = max_memory
        wandb.run.summary['no_params'] = no_params
        wandb.run.summary['no_params_trainable'] = no_params_trainable
        wandb.run.summary['class_deviation'] = class_deviation
    return 0


def stats_test(test_acc, class_deviation, flops, max_memory, no_params,
               no_params_trainable, time_total, num_images, offline=False):

    throughput = round(num_images / time_total, 4)
    no_params = round(no_params / (1e6), 4)  # millions of parameters
    no_params_trainable = round(no_params_trainable / (1e6), 4)  # millions
    max_memory = round(max_memory, 4)

    print('''Throughput (images / s): {}
          Test accuracy (%): {}
          Class deviation (%): {}
          Max VRAM consumption (GB): {}
          Total number of parameters in all modules (M): {}
          Trainable number of parameters in all modules (M): {}
          '''.format(throughput, test_acc, class_deviation, max_memory, no_params, no_params_trainable))

    if not offline:
        wandb.run.summary['test_acc'] = test_acc
        wandb.run.summary['class_deviation'] = class_deviation
        wandb.run.summary['throughput'] = throughput
        wandb.run.summary['flops'] = flops
        wandb.run.summary['max_memory'] = max_memory
        wandb.run.summary['no_params'] = no_params
        wandb.run.summary['no_params_trainable'] = no_params_trainable
    return 0


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions
    for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
