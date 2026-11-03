# extract IWDD frames (5 fps, short side 256) and write the video and window csvs; skips extracted videos
# usage: python tools/iwdd/prepare_iwdd.py <iwdd_root> (ffmpeg on PATH)
import os
import sys
import json
import random
import subprocess
from glob import glob
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

FPS = 5
SHORT_SIDE = 256
# frames per window (3.2 s) and between window starts (1 s)
WINDOW = 16
STRIDE = 5
FOLDS = 5

root = sys.argv[1]


def extract(split, video):
    video_id = os.path.splitext(os.path.basename(video))[0]
    out = os.path.join(root, 'frames', split, video_id)
    if not glob(os.path.join(out, '*.jpg')):
        os.makedirs(out, exist_ok=True)
        cmd = ['ffmpeg', '-loglevel', 'error', '-i', video, '-vf', f'fps={FPS},scale=-2:{SHORT_SIDE}',
               '-q:v', '3', os.path.join(out, '%05d.jpg')]
        subprocess.run(cmd, check=True)
    return video_id, sorted(os.path.relpath(p, root) for p in glob(os.path.join(out, '*.jpg')))


def videos(split):
    rows = []
    video_paths = sorted(glob(os.path.join(root, split, 'videos', '*.mp4')))
    with ThreadPoolExecutor(max_workers=10) as pool:
        frames = dict(pool.map(lambda v: extract(split, v), video_paths))
    for path in sorted(glob(os.path.join(root, split, 'labels', '*.json'))):
        with open(path) as f:
            label = json.load(f)
        meta = label['metadata']
        video_id = os.path.splitext(meta['video_id'])[0]
        assert frames[video_id], video_id
        rows.append({'class_id': int(label['label'] == 'true'), 'dir': [(p,) for p in frames[video_id]],
                     'video_id': video_id, 'event_type': meta['event_type'], 'timestamp': label['timestamp'],
                     'time_of_day': meta['time_of_day'], 'lighting': meta['lighting'],
                     'duration': meta['duration'], 'n_frames': len(frames[video_id])})
    assert len(rows) == len(frames), (split, len(rows), len(frames))
    return pd.DataFrame(rows)


def assign_folds(df):
    # round-robin per label, event type and time of day
    rng = random.Random(0)
    fold = {}
    for _, group in df.groupby(['class_id', 'event_type', 'time_of_day']):
        ids = sorted(group['video_id'])
        rng.shuffle(ids)
        fold.update({vid: i % FOLDS for i, vid in enumerate(ids)})
    return df.assign(fold=df['video_id'].map(fold))


def windows(df):
    # dumping: ends at or after the onset
    rows = []
    for video in df.to_dict('records'):
        frames = video['dir']
        starts = range(0, max(len(frames) - WINDOW, 0) + 1, STRIDE)
        for start in starts:
            t_start = start / FPS
            t_end = (min(start + WINDOW, len(frames)) - 1) / FPS
            dumping = video['class_id'] == 1 and t_end >= video['timestamp']
            rows.append({**video, 'class_id': int(dumping), 'dir': frames[start:start + WINDOW],
                         't_start': t_start, 't_end': t_end})
    return pd.DataFrame(rows)


def write(df, name):
    df.assign(dir=df['dir'].map(str)).to_csv(os.path.join(root, name), index=False)
    print(name, len(df), 'rows,', df['class_id'].value_counts().sort_index().to_dict())


train = assign_folds(videos('train'))
test = videos('public_test').assign(fold=-1)
for prefix, convert in (('', lambda df: df), ('win_', windows)):
    write(convert(train), f'{prefix}train_val.csv')
    write(convert(test), f'{prefix}test.csv')
    for k in range(FOLDS):
        write(convert(train[train['fold'] != k]), f'{prefix}train_f{k}.csv')
        write(convert(train[train['fold'] == k]), f'{prefix}val_f{k}.csv')
