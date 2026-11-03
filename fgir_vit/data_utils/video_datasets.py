import os
import ast
import random
from functools import partial

import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange

import torch
import torch.utils.data as data

from .datasets import train_subset


def split_csv(args, split):
    # same split rule as DatasetImgTarget
    if split == 'train':
        return args.df_trainval if args.train_trainval else args.df_train
    if split == 'val':
        return args.df_test if args.train_trainval else args.df_val
    return args.df_test


# each returns n_frames indices into a clip of length t; short clips repeat their last frame
def uniform_indices(t, n_frames):
    # same indices as pytorchvideo's uniform_temporal_subsample
    return torch.linspace(0, t - 1, n_frames).long().tolist()


def center_indices(t, n_frames):
    start = max(t - n_frames, 0) // 2
    return pad_last(list(range(start, min(start + n_frames, t))), n_frames)


def random_indices(t, n_frames):
    start = random.randint(0, max(t - n_frames, 0))
    return pad_last(list(range(start, min(start + n_frames, t))), n_frames)


def stride_indices(t, n_frames, stride):
    return pad_last(list(range(0, t, stride))[:n_frames], n_frames)


def segment_indices(t, n_frames, jitter):
    # one frame per equal segment: random within it when training (jitter), its centre otherwise
    edges = np.linspace(0, t, n_frames + 1)
    if jitter:
        return [random.randint(int(a), max(int(a), int(np.ceil(b)) - 1)) for a, b in zip(edges[:-1], edges[1:])]
    return np.minimum(((edges[:-1] + edges[1:]) / 2).astype(int), t - 1).tolist()


def pad_last(indices, n_frames):
    return indices + indices[-1:] * (n_frames - len(indices))


SUBSAMPLE = {
    'uniform': uniform_indices,
    'center': center_indices,
    'random': random_indices,
    'stride': stride_indices,
    'segment': segment_indices,
}


class DatasetVideoTarget(data.Dataset):
    def __init__(self, args, split, transform=None):
        self.root = os.path.abspath(args.dataset_root_path)
        self.transform = transform
        self.frame_indices = partial(SUBSAMPLE[args.temporal_subsample], n_frames=args.n_frames)
        if args.temporal_subsample == 'stride':
            self.frame_indices = partial(self.frame_indices, stride=args.temporal_stride)
        if args.temporal_subsample == 'segment':
            self.frame_indices = partial(self.frame_indices, jitter=split == 'train')

        self.df_file_name = split_csv(args, split)
        assert os.path.isfile(os.path.join(self.root, self.df_file_name)), \
            f'{os.path.join(self.root, self.df_file_name)} is not a file.'

        self.df = pd.read_csv(os.path.join(self.root, self.df_file_name), sep=',')
        if split == 'train' and getattr(args, 'train_subset', None):
            self.df = train_subset(self.df, args.train_subset, args.seed)
        self.targets = self.df['class_id'].to_numpy()
        self.data = [ast.literal_eval(record) for record in self.df['dir']]

        self.num_classes = len(np.unique(self.targets))

    def load_clip(self, idx):
        # c t h w, uint8
        paths = self.data[idx]
        frames = []
        for i in self.frame_indices(len(paths)):
            img = Image.open(os.path.join(self.root, paths[i][0]))
            frames.append(np.array(img.convert('RGB')))
        return rearrange(torch.from_numpy(np.stack(frames)), 't h w c -> c t h w')

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        clip = self.load_clip(idx)

        if self.transform:
            clip = self.transform(clip)
        else:
            clip = clip.float() / 255.0

        return clip, self.targets[idx]

    def __len__(self):
        return len(self.targets)


class DatasetMP4VideoTarget(DatasetVideoTarget):
    def __init__(self, args, split, transform=None):
        super().__init__(args, split, transform)
        self.start_frame = self.df['start_frame'].to_numpy()
        self.end_frame = self.df['end_frame'].to_numpy()

    def load_clip(self, idx):
        import decord
        decord.bridge.set_bridge('torch')

        reader = decord.VideoReader(os.path.join(self.root, 'rgb', f'{self.data[idx][0]}.mp4'))
        start, t = self.start_frame[idx], self.end_frame[idx] - self.start_frame[idx] + 1
        frames = reader.get_batch([start + i for i in self.frame_indices(t)])
        return rearrange(frames, 't h w c -> c t h w')
