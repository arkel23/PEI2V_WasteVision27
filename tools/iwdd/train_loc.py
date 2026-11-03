# IWDD onset detection on whole videos: per-frame features of an FGIR-ViT backbone (any adapter or full fine-tuning
# flags; or precomputed frozen features) and their change from the video's first 2 s feed a temporal head
# that scores every step against [onset - 3 s, onset + 10 s]; the video score is the maximum step score.
# Usage: python -m tools.iwdd.train_loc --cfg configs/iwdd_ft.yaml <backbone flags> --loc_out <csv> [--loc_fold k] (GPU).
import ast
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from PIL import Image
from torchvision.transforms import v2
import wandb

from fgir_vit.other_utils.build_args import parse_train_args, adjust_config, add_adjust_common_dependent
from fgir_vit.data_utils.build_transform import get_mean_std
from fgir_vit.model_utils.build_model import build_model
from fgir_vit.model_utils.misc.configs import TEMPORAL_ADAPTERS
from fgir_vit.train_utils.misc_utils import set_random_seed
from tools.train import adjust_args_general

EARLY, LATE = 3.0, 10.0
CHANGE_S = 2.0  # the video's first seconds define the scene for the change cue
DISK_FPS = 5
CHUNK = 32  # frames per backbone pass; temporal adapters need --n_frames 32


def parse_args():
    parser = parse_train_args(ret_parser=True)
    parser.add_argument('--loc_out', type=str, required=True, help='csv of step scores: split, seed, video_id, t, score')
    parser.add_argument('--loc_fold', type=int, default=-1, help='held-out fold; -1 trains on all and scores test.csv')
    parser.add_argument('--loc_folds', type=str, default=None, help='csv with video_id and fold; default the fold column of train_val.csv')
    parser.add_argument('--loc_fps', type=float, default=2.0)
    parser.add_argument('--loc_head', type=str, default='transformer', choices=['transformer', 'conv'])
    parser.add_argument('--loc_depth', type=int, default=1, help='transformer blocks or residual convolutions')
    parser.add_argument('--loc_features', type=str, default=None, help='npz of frozen per-frame features: no backbone')
    parser.add_argument('--loc_accum', type=int, default=8, help='videos per optimizer step when training the backbone')
    parser.add_argument('--loc_backbone_lr', type=float, default=1e-5, help='lr of pretrained backbone weights that train')
    args = parser.parse_args()
    assert args.n_frames == CHUNK or args.adapter not in TEMPORAL_ADAPTERS, '--n_frames must be 32 with a temporal adapter'
    adjust_config(parser, args)
    args.video, args.video_model, args.num_classes = 'frames', 'temporal', 1
    return adjust_args_general(add_adjust_common_dependent(args))


class Head(nn.Module):
    # per-step features -> temporal context (attention or residual convolutions) -> one logit per step;
    # position enters only through a depthwise temporal convolution, so the head sees relative time only
    def __init__(self, in_dim, kind, depth, dim=256, kernel=5):
        super().__init__()
        self.kind = kind
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim), nn.GELU(), nn.Dropout(0.3))
        self.pos = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        if kind == 'transformer':
            self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(dim, 4, 4 * dim, 0.1, batch_first=True, norm_first=True)
                                        for _ in range(depth))
        else:
            self.blocks = nn.ModuleList(nn.Conv1d(dim, dim, kernel, padding=kernel // 2) for _ in range(depth))
        self.out = nn.Linear(dim, 1)

    def forward(self, x, mask):  # b t c, b t -> b t
        h = self.proj(x) * mask[..., None]
        h = h + self.pos(h.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            if self.kind == 'transformer':
                h = block(h, src_key_padding_mask=~mask)
            else:
                h = h + nn.functional.gelu(block(h.transpose(1, 2))).transpose(1, 2) * mask[..., None]
        return self.out(h).squeeze(-1)


class Detector(nn.Module):
    def __init__(self, args, feat_dim=None):
        super().__init__()
        self.backbone = None if args.loc_features else build_model(args).backbone
        if self.backbone is not None:
            cfg = self.backbone.cfg
            feat_dim = 2 * (cfg.hidden_size if hasattr(cfg, 'hidden_size') else self.backbone.model.embed_dim)
        self.head = Head(2 * feat_dim, args.loc_head, args.loc_depth)
        self.n_scene = max(1, round(CHANGE_S * args.loc_fps))

    def frame_features(self, frames):  # t c h w -> t d: class token and mean patch token
        tokens = self.backbone(frames)
        return torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], -1).float()

    def encode(self, frames):
        # CHUNK-frame passes; the last repeats its final frame up to CHUNK so temporal adapters see a fixed length
        trains = self.training and any(p.requires_grad for p in self.backbone.parameters())
        n = len(frames)
        frames = torch.cat([frames, frames[-1:].expand(-CHUNK * (-n // CHUNK) - n, *frames.shape[1:])])
        chunks = frames.split(CHUNK)
        if trains:
            return torch.cat([checkpoint(self.frame_features, c, use_reentrant=False) for c in chunks])[:n]
        with torch.no_grad():
            return torch.cat([self.frame_features(c) for c in chunks])[:n]

    def forward(self, feats, mask):  # b t d (features or frames encoded per video), b t -> b t
        scene = (feats * mask[..., None])[:, :self.n_scene].sum(1, keepdim=True) / mask[:, :self.n_scene].sum(1)[:, None, None]
        return self.head(torch.cat([feats, feats - scene], -1), mask)


def frame_paths(root, row, fps):
    # the 5 fps frame nearest to each 1/fps step
    folder = os.path.dirname(ast.literal_eval(row['dir'])[0][0])
    idx = np.unique(np.round(np.arange(0, (row['n_frames'] - 1) / DISK_FPS + 1e-6, 1 / fps) * DISK_FPS).astype(int))
    return [os.path.join(root, folder, f'{k + 1:05d}.jpg') for k in idx]


def load(paths, transform):
    # one transform for the whole video, so a random flip is the same in every frame
    return transform(torch.stack([v2.functional.pil_to_tensor(Image.open(p).convert('RGB')) for p in paths]))


def targets(n, fps, onset):
    t = np.arange(n) / fps
    return np.zeros(n, np.float32) if onset < 0 else ((t >= onset - EARLY) & (t <= onset + LATE)).astype(np.float32)


class Videos(torch.utils.data.Dataset):
    # one whole video: frames (or frozen features) and step targets
    def __init__(self, rows, root, fps, transform, features):
        self.rows, self.root, self.fps, self.transform, self.features = rows, root, fps, transform, features

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = (torch.from_numpy(self.features[r['video_id']].reshape(len(self.features[r['video_id']]), -1).astype(np.float32))
             if self.features is not None else load(frame_paths(self.root, r, self.fps), self.transform))
        return x, torch.from_numpy(targets(len(x), self.fps, r['timestamp'])), float(r['timestamp'] >= 0), r['video_id']


def pad(batch):
    n = max(len(x) for x, _, _, _ in batch)
    x = torch.zeros(len(batch), n, *batch[0][0].shape[1:])
    y, mask = torch.zeros(len(batch), n), torch.zeros(len(batch), n, dtype=torch.bool)
    for i, (xi, yi, _, _) in enumerate(batch):
        x[i, :len(xi)], y[i, :len(yi)], mask[i, :len(xi)] = xi, yi, True
    return x, y, mask, torch.tensor([b[2] for b in batch]), [b[3] for b in batch]


def run_batch(model, x, mask, device):
    # frozen features pass straight to the head; frames go through the backbone one video at a time
    x, mask = x.to(device), mask.to(device)
    with torch.autocast('cuda', torch.float16, enabled=device != 'cpu'):
        if model.backbone is not None:
            x = torch.stack([nn.functional.pad(model.encode(xi[:int(m.sum())]), (0, 0, 0, len(xi) - int(m.sum()))) for xi, m in zip(x, mask)])
        return model(x.float(), mask).float()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    root, fps = args.dataset_root_path, args.loc_fps
    videos = pd.read_csv(os.path.join(root, 'train_val.csv'))
    if args.loc_folds:
        folds = pd.read_csv(args.loc_folds)
        videos['fold'] = videos['video_id'].map(dict(zip(folds['video_id'], folds['fold'])))
    test = pd.read_csv(os.path.join(root, 'test.csv'))
    train_df, eval_df, split = ((videos, test, 'test') if args.loc_fold < 0 else
                                (videos[videos['fold'] != args.loc_fold], videos[videos['fold'] == args.loc_fold], 'oof'))
    if args.debugging:
        train_df, eval_df = train_df[:2], eval_df[:2]

    features = dict(np.load(args.loc_features)) if args.loc_features else None
    if features is not None:
        assert abs(float(features.pop('fps')) - fps) < 1e-6, '--loc_fps must match the features'
    mean, std = get_mean_std(args)
    size = (args.image_size, args.image_size)
    train_tf = v2.Compose([v2.Resize(size), v2.RandomHorizontalFlip(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])
    eval_tf = v2.Compose([v2.Resize(size), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])

    feat_dim = int(np.prod(next(iter(features.values())).shape[1:])) if features is not None else None
    model = Detector(args, feat_dim).to(args.device)
    pretrained = [p for p in (model.backbone.parameters() if model.backbone is not None else []) if p.requires_grad]
    # new modules (head, adapters) at --lr; pretrained weights, when they train, at --loc_backbone_lr
    new = [p for p in model.parameters() if p.requires_grad and (args.freeze_backbone or all(p is not q for q in pretrained))]
    groups = [{'params': new, 'lr': args.lr}]
    if not args.freeze_backbone and pretrained:
        groups.append({'params': pretrained, 'lr': args.loc_backbone_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    per_step = args.batch_size if features is not None else args.loc_accum
    steps = args.epochs * int(np.ceil(len(train_df) / per_step))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g['lr'] for g in groups], total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device != 'cpu')
    bce = nn.BCEWithLogitsLoss(reduction='none')
    print(f'trainable {sum(p.numel() for g in groups for p in g["params"]) / 1e6:.2f} M, {len(train_df)} train videos, '
          f'{split} {len(eval_df)}, {args.loc_head} head', flush=True)
    if args.wandb_log:
        wandb.init(config=args, project=args.project_name, entity=args.entity)
        wandb.run.name = args.run_name
        wandb.define_metric('epoch')
        wandb.define_metric('train_*', step_metric='epoch')
    time_start = time.time()

    loader_bs = args.batch_size if features is not None else 1
    loader = torch.utils.data.DataLoader(Videos(list(train_df.to_dict('records')), root, fps, train_tf, features), batch_size=loader_bs,
                                         shuffle=True, num_workers=0 if features is not None else args.cpu_workers, collate_fn=pad)
    accum = 1 if features is not None else args.loc_accum
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for i, (x, y, mask, label, _) in enumerate(loader):
            logits = run_batch(model, x, mask, args.device)
            y, mask, label = y.to(args.device), mask.to(args.device), label.float().to(args.device)
            loss = (bce(logits, y) * mask).sum() / mask.sum() + bce(logits.masked_fill(~mask, -1e4).amax(1), label).mean()
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or i + 1 == len(loader):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
            losses.append(loss.item())
        print(f'epoch {epoch + 1}/{args.epochs} loss {np.mean(losses):.4f}', flush=True)
        if args.wandb_log:
            wandb.log({'epoch': epoch + 1, 'train_loss': np.mean(losses)})

    model.eval()
    out = []
    eval_loader = torch.utils.data.DataLoader(Videos(list(eval_df.to_dict('records')), root, fps, eval_tf, features), batch_size=1,
                                              num_workers=0 if features is not None else args.cpu_workers, collate_fn=pad)
    with torch.no_grad():
        for x, _, mask, _, ids in eval_loader:
            scores = torch.sigmoid(run_batch(model, x, mask, args.device))[0].cpu().numpy()
            out += [{'split': split, 'seed': args.seed, 'video_id': ids[0], 't': round(i / fps, 2), 'score': round(float(s), 5)}
                    for i, s in enumerate(scores)]
    os.makedirs(os.path.dirname(os.path.abspath(args.loc_out)), exist_ok=True)
    pd.DataFrame(out).to_csv(args.loc_out, index=False)
    memory = torch.cuda.max_memory_allocated() / 2 ** 30 if args.device != 'cpu' else 0
    print(f'{args.loc_out}: {len(out)} step scores; peak memory {memory:.1f} GB')
    if args.wandb_log:
        wandb.run.summary['time_total'] = round((time.time() - time_start) / 60, 4)
        wandb.run.summary['max_memory'] = round(memory, 4)
        wandb.run.summary['no_params_trainable'] = round(sum(p.numel() for g in groups for p in g['params']) / 1e6, 4)
        wandb.run.summary['step_scores'] = len(out)
        wandb.finish()


if __name__ == '__main__':
    main()
