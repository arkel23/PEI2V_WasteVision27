from torch import nn

# AIM adapters (arXiv:2302.03024, github.com/taoyang1122/adapt-image-models), outputs equal to the release's


class AIMAdapter(nn.Module):
    # bottleneck mlp, zero-init fc2
    def __init__(self, dim, adapter_dim, skip=True):
        super().__init__()
        self.skip = skip
        self.fc1 = nn.Linear(dim, adapter_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(adapter_dim, dim)
        self.init_weights()

    def init_weights(self):
        # also called by the backbone's init
        nn.init.trunc_normal_(self.fc1.weight, std=.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = self.fc2(self.act(self.fc1(x)))
        return x + h if self.skip else h


class AIM(nn.Module):
    # temporal, spatial and mlp adapters
    def __init__(self, dim, adapter_dim, n_frames, scale, pre_attn=False):
        super().__init__()
        self.n_frames = n_frames
        self.scale = scale
        self.t_in = AIMAdapter(dim, adapter_dim) if pre_attn else nn.Identity()
        self.t = AIMAdapter(dim, adapter_dim, skip=False)
        self.s = AIMAdapter(dim, adapter_dim)
        self.mlp = AIMAdapter(dim, adapter_dim, skip=False)
