import numpy as np

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F

from timm.layers import DropPath
from .adapters import Adapter


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class AllYouNeedAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout, proj_dim=None):
        super().__init__()
        if not proj_dim:
            proj_dim = dim

        self.proj_q = nn.Linear(dim, proj_dim)
        self.proj_k = nn.Linear(dim, proj_dim)
        self.proj_v = nn.Linear(dim, proj_dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.fused_attn = False

    def forward(self, x, mask=None):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)

        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        if self.fused_attn:
            # no attention maps: scores_soft and scores are None
            if mask is not None:
                mask = -10000.0 * (1.0 - mask[:, None, None, :].float())
            h = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.drop.p if self.training else 0.)
            return merge_last(h.transpose(1, 2).contiguous(), 2), None, None
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        # this is what's used to visualize attention
        scores_soft = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores_soft @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)

        return h, scores_soft, scores


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class BlockVanilla(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, hidden_dropout_prob,
                 attention_probs_dropout_prob, layer_norm_eps, sd=0,
                 adapter=None, adapter_dim=8, adapter_kernel_size=3,
                 adapter_path='pfeiffer', scale=1.0, adapter_pre_attn=False,
                 n_frames=16, adapter_scale=1.0):
        super().__init__()
        self.attn = AllYouNeedAttention(dim, num_heads, attention_probs_dropout_prob)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)

        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)

        if sd > 0:
            self.drop = DropPath(sd)
        else:
            self.drop = nn.Dropout(hidden_dropout_prob)

        if str(adapter).startswith('frame2freq'):
            # sequential on each clip's frames: after MHSA, and also before it with adapter_pre_attn
            from .frame2freq import build_frame2freq
            self.adapter_f2f = build_frame2freq(adapter, dim, adapter_dim, n_frames)
            if adapter_pre_attn:
                self.adapter_f2f_attn = build_frame2freq(adapter, dim, adapter_dim, n_frames)
        elif adapter == 'st_adapter':
            # before the MLP, and also before MHSA with adapter_pre_attn
            from .st_adapter import STAdapter
            kernel_size = (adapter_kernel_size, 1, 1)
            self.adapter_st = STAdapter(dim, adapter_dim, n_frames, kernel_size)
            if adapter_pre_attn:
                self.adapter_st_attn = STAdapter(dim, adapter_dim, n_frames, kernel_size)
        elif adapter == 'aim':
            from .aim import AIM
            self.adapter_aim = AIM(dim, adapter_dim, n_frames, adapter_scale, adapter_pre_attn)
            self.forward = self.forward_aim
        elif adapter:
            # https://arxiv.org/abs/2005.00247
            self.adapter_path = adapter_path
            self.scale = scale
            self.adapter_pwffn = Adapter(adapter, dim, adapter_dim, adapter_kernel_size)
            if self.adapter_path in ('convpass'):
                self.adapter_attn = Adapter(adapter, dim, adapter_dim, adapter_kernel_size)
 
    def forward(self, x):
        if hasattr(self, 'adapter_f2f_attn'):
            x = self.adapter_f2f_attn(x)
        if hasattr(self, 'adapter_st_attn'):
            x = self.adapter_st_attn(x)

        if hasattr(self, 'adapter_path') and self.adapter_path in ('convpass', 'ours'):
            res = x
            x = self.norm1(x)

            h, scores_soft, scores = self.attn(x)
            h = self.drop(self.proj(h))

            if hasattr(self, 'adapter_attn'):
                h2 = self.drop(self.adapter_attn(x))
                x = res + h + (self.scale * h2)
            else:
                x = res + h

            res = x
            x = self.norm2(x)

            h = self.drop(self.pwff(x))

            if hasattr(self, 'adapter_pwffn'):
                h2 = self.drop(self.adapter_pwffn(x))
                x = res + h + (self.scale * h2)
            else:
                x = res + h

        else:
            res = x

            h, scores_soft, scores = self.attn(self.norm1(x))
            h = self.drop(self.proj(h))

            if hasattr(self, 'adapter_attn'):
                h2 = self.drop(self.adapter_attn(res + h, scores_soft))
                x = res + h + (self.scale * h2)
            else:
                x = res + h

            if hasattr(self, 'adapter_f2f'):
                x = self.adapter_f2f(x)
            if hasattr(self, 'adapter_st'):
                x = self.adapter_st(x)

            res = x

            h = self.drop(self.pwff(self.norm2(x)))

            if hasattr(self, 'adapter_pwffn'):
                h2 = self.drop(self.adapter_pwffn(res + h))
                x = res + h + (self.scale * h2)
            else:
                x = res + h

        return x, scores_soft, scores

    def forward_aim(self, x):
        # the frozen MHSA over each token's frames, then over each frame's tokens; drop on the t and mlp adapters
        aim, n = self.adapter_aim, x.shape[1]
        xt = rearrange(x, '(b t) n d -> (b n) t d', t=aim.n_frames)
        xt = aim.t(self.proj(self.attn(aim.t_in(self.norm1(xt)))[0]))
        x = x + self.drop(rearrange(xt, '(b n) t d -> (b t) n d', n=n))
        h, scores_soft, scores = self.attn(self.norm1(x))
        x = x + aim.s(self.proj(h))
        xn = self.norm2(x)
        return x + self.pwff(xn) + self.drop(aim.scale * aim.mlp(xn)), scores_soft, scores


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""

    def __init__(
        self,
        num_layers=12,
        dim=768,
        num_heads=12,
        ff_dim=768*4,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-12,
        sd=0,

        adapter=None,
        adapter_dim=8,
        adapter_kernel_size=3,
        adapter_path='convpass',
        adapter_pre_attn=False,
        n_frames=16,
        adapter_scale=1.0,
        adapter_depth=None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            BlockVanilla(
                dim, num_heads, ff_dim, hidden_dropout_prob, attention_probs_dropout_prob,
                layer_norm_eps, sd, adapter if i < (adapter_depth or num_layers) else None, adapter_dim,adapter_kernel_size,
                adapter_path, adapter_pre_attn=adapter_pre_attn,
                n_frames=n_frames, adapter_scale=adapter_scale) for i in range(num_layers)])

    def forward(self, x):
        for block in self.blocks:
            x, _, _ = block(x)
        return x
