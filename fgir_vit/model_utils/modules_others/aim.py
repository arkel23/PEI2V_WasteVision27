from torch import nn

# AIM adapters (arXiv:2302.03024, github.com/taoyang1122/adapt-image-models), outputs equal to the release's.
# The block runs them around its frozen MHSA and MLP; every token, the class token included, is adapted.


class AIMAdapter(nn.Module):
    # fc1 -> GELU -> fc2, zero-init fc2; skip adds the input
    def __init__(self, dim, adapter_dim, skip=True):
        super().__init__()
        self.skip = skip
        self.fc1 = nn.Linear(dim, adapter_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(adapter_dim, dim)
        self.init_weights()

    def init_weights(self):
        # also called by the backbones' init after they reset every Linear
        nn.init.trunc_normal_(self.fc1.weight, std=.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = self.fc2(self.act(self.fc1(x)))
        return x + h if self.skip else h


class AIM(nn.Module):
    # t: on the MHSA output over each token's frames (t_in before it with pre_attn); s: on the MHSA output
    # over each frame's tokens; mlp: parallel to the MLP, times scale
    def __init__(self, dim, adapter_dim, n_frames, scale, pre_attn=False):
        super().__init__()
        self.n_frames = n_frames
        self.scale = scale
        self.t_in = AIMAdapter(dim, adapter_dim) if pre_attn else nn.Identity()
        self.t = AIMAdapter(dim, adapter_dim, skip=False)
        self.s = AIMAdapter(dim, adapter_dim)
        self.mlp = AIMAdapter(dim, adapter_dim, skip=False)
