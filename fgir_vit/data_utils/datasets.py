import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile

import torch
import torch.utils.data as data


ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_set(args, split, transform=None):
    if args.video:
        from .video_datasets import DatasetVideoTarget, DatasetMP4VideoTarget
        video_dataset = DatasetVideoTarget if args.video == 'frames' else DatasetMP4VideoTarget
        ds = video_dataset(args, split=split, transform=transform)

    elif args.dataset_in_memory:
        ds = DatasetImgTargetInMemory(args, split=split, transform=transform)
    else:
        ds = DatasetImgTarget(args, split=split, transform=transform)

    if split == 'train':
        args.num_classes = ds.num_classes

    setattr(args, f'num_images_{split}', ds.__len__())
    print(f"{args.dataset_name} {split} split. N={ds.__len__()}, K={ds.num_classes}.")
    return ds


def train_subset(df, subset, seed):
    # per class: a fraction (subset < 1) or a number of samples, at least one; nested in subset for a fixed seed
    rng = np.random.RandomState(seed)
    keep = []
    for _, rows in df.groupby('class_id', sort=True):
        n = round(subset * len(rows)) if subset < 1 else int(subset)
        keep.extend(rng.permutation(rows.index)[:max(1, n)])
    return df.loc[sorted(keep)].reset_index(drop=True)


class DatasetImgTarget(data.Dataset):
    def __init__(self, args, split, transform=None):
        self.root = os.path.abspath(args.dataset_root_path)
        self.transform = transform
        self.dataset_name = args.dataset_name

        if split == 'train':
            if args.train_trainval:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_trainval
            else:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_train
        elif split == 'val':
            if args.train_trainval:
                self.images_folder = args.folder_test
                self.df_file_name = args.df_test
            else:
                self.images_folder = args.folder_val
                self.df_file_name = args.df_val
        else:
            self.images_folder = args.folder_test
            self.df_file_name = args.df_test

        assert os.path.isfile(os.path.join(self.root, self.df_file_name)), \
            f'{os.path.join(self.root, self.df_file_name)} is not a file.'

        self.df = pd.read_csv(os.path.join(self.root, self.df_file_name), sep=',')
        if split == 'train' and args.train_subset:
            self.df = train_subset(self.df, args.train_subset, args.seed)
        self.targets = self.df['class_id'].to_numpy()
        self.data = self.df['dir'].to_numpy()

        self.num_classes = len(np.unique(self.targets))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir, target = self.data[idx], self.targets[idx]
        full_img_dir = os.path.join(self.root, self.images_folder, img_dir)
        img = Image.open(full_img_dir)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.targets)



class DatasetImgTargetInMemory(DatasetImgTarget):
    def __init__(self, args, split, transform=None):
        super().__init__(args, split, transform)
        resize_size = args.resize_size if split == 'train' else args.test_resize_size

        self.images = [None] * len(self.data)

        def load_image(i, img_dir):
        # for i, img_dir in enumerate(self.data):
            full_img_dir = os.path.join(self.root, self.images_folder, img_dir)
            img = Image.open(full_img_dir)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            img = img.resize((resize_size, resize_size))

            self.images[i] = img.copy()
            img.close()

        max_workers = 4 if (not args.cpu_workers or args.cpu_workers < 4) else args.cpu_workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(load_image, i, img_dir) for i, img_dir in enumerate(self.data)]
            for f in futures:
                f.result()  # Wait for all threads to complete

        print(f'Loaded all {split} images on memory: {len(self.images)}')

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, target = self.images[idx], self.targets[idx]

        if self.transform:
            img = self.transform(img)

        return img, target
