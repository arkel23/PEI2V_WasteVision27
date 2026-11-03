from einops import rearrange

import torch
from torch import nn


class Adapter(nn.Module):
    def __init__(self, adapter, dim, adapter_dim, kernel_size, adapter_init='xavier', adapter_scale=1.0):
        super().__init__()

        self.adapter_down = nn.Linear(dim, adapter_dim)
        self.adapter_up = nn.Linear(adapter_dim, dim)
        self.act = nn.GELU()

        if adapter == 'convpass':
            padding = kernel_size // 2
            self.adapter_conv = nn.Conv2d(adapter_dim, adapter_dim, kernel_size, 1, padding)

        self.adapter_init = adapter_init
        # negative: learnable scale
        if adapter_scale >= 0:
            self.register_buffer('adapter_scale', torch.tensor(float(adapter_scale)))
        else:
            self.adapter_scale = nn.Parameter(torch.tensor(abs(float(adapter_scale))))

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        if hasattr(self, 'adapter_down'):
            nn.init.xavier_uniform_(self.adapter_down.weight)
            nn.init.zeros_(self.adapter_down.bias)
            if self.adapter_init == 'zero':
                nn.init.zeros_(self.adapter_up.weight)
            else:
                nn.init.xavier_uniform_(self.adapter_up.weight)
            nn.init.zeros_(self.adapter_up.bias)
        if hasattr(self, 'adapter_conv'):
            nn.init.xavier_uniform_(self.adapter_conv.weight)

    def forward(self, x):
        x = self.act(self.adapter_down(x))

        if hasattr(self, 'adapter_conv'):
            _, s, _ = x.shape
            h = int(s ** 0.5)
            cls = False if (h ** 2) == s else True

            if cls:
                x_cls, x_patches = torch.split(x, [1, s - 1], dim=1)
                x_cls = rearrange(x_cls, 'b 1 d -> b d 1 1')
                x_cls = self.adapter_conv(x_cls)
                x_cls = rearrange(x_cls, 'b d 1 1 -> b 1 d')

                x_patches = rearrange(x_patches, 'b (h w) d -> b d h w', h=h)
                x_patches = self.adapter_conv(x_patches)
                x_patches = rearrange(x_patches, 'b d h w -> b (h w) d')

                x = torch.cat([x_cls, x_patches], dim=1)
                x = self.act(x)
            else:
                x = rearrange(x, 'b (h w) d -> b d h w', h=h)
                x = self.adapter_conv(x)
                x = rearrange(x, 'b d h w -> b (h w) d')
                x = self.act(x)

        x = self.adapter_up(x)
        x = x * self.adapter_scale
        return x
