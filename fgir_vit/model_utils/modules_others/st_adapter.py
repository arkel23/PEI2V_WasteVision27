import torch
from torch import nn

from .frame2freq import split_prefix

# ST-Adapter (arXiv:2206.13559, github.com/linziyi96/st-adapter), outputs equal to the release's


class STAdapter(nn.Module):
    # down, depthwise conv over frames, up; identity at init
    def __init__(self, dim, adapter_dim, n_frames, kernel_size=(3, 1, 1)):
        super().__init__()
        self.n_frames = n_frames
        self.fc1 = nn.Linear(dim, adapter_dim)
        self.conv = nn.Conv3d(adapter_dim, adapter_dim, kernel_size, padding=tuple(k // 2 for k in kernel_size),
                              groups=adapter_dim)
        self.fc2 = nn.Linear(adapter_dim, dim)
        self.init_weights()

    def init_weights(self):
        # also called by the backbone's init
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        for p in (self.conv.weight, self.conv.bias, self.fc1.bias, self.fc2.bias):
            nn.init.zeros_(p)

    def forward(self, x):
        prefix, patches, h = split_prefix(x)
        z = self.fc1(patches)
        z = z.view(-1, self.n_frames, h, h, z.shape[-1]).permute(0, 4, 1, 2, 3)
        z = self.conv(z)
        z = z.permute(0, 2, 3, 4, 1).reshape(patches.shape[0], h * h, -1)
        return torch.cat([prefix, patches + self.fc2(z)], dim=1)
