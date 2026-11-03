# Frozen per-frame features of every IWDD video: class token and mean patch token of a timm ViT, at fps frames per second.
# Usage: python tools/iwdd/extract_frame_features.py <iwdd_root> <timm_model> <out.npz> [fps] (GPU if available).
import os
import sys

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torchvision.transforms import v2

root, model_name, out = sys.argv[1:4]
fps = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0  # the 5 fps frame nearest to each 1/fps step
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(device)
cfg = model.pretrained_cfg
transform = v2.Compose([v2.Resize(cfg['input_size'][1:]), v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(mean=cfg['mean'], std=cfg['std'])])

videos = pd.concat([pd.read_csv(os.path.join(root, f)) for f in ('train_val.csv', 'test.csv')])
features = {}
with torch.inference_mode(), torch.autocast('cuda', torch.float16, enabled=device == 'cuda'):
    for i, (video_id, n_frames, split) in enumerate(zip(videos['video_id'], videos['n_frames'], videos['dir'])):
        folder = os.path.dirname(eval(split)[0][0])
        idx = np.unique(np.round(np.arange(0, (n_frames - 1) / 5 + 1e-6, 1 / fps) * 5).astype(int))
        frames = [os.path.join(root, folder, f'{k + 1:05d}.jpg') for k in idx]
        clip = torch.stack([transform(v2.functional.pil_to_tensor(Image.open(f).convert('RGB'))) for f in frames])
        tokens = model.forward_features(clip.to(device))
        patches = tokens[:, model.num_prefix_tokens:].mean(1)
        features[video_id] = torch.stack([tokens[:, 0], patches], 1).half().cpu().numpy()
        if i % 50 == 0:
            print(f'{i}/{len(videos)} {video_id}: {len(frames)} frames', flush=True)

np.savez(out, fps=fps, **features)
print(f'{out}: {len(features)} videos, feature shape {next(iter(features.values())).shape[1:]}')
