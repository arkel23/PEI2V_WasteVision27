import torch
import torch.nn as nn
from einops import rearrange, repeat

from .misc import TEMPORAL_ADAPTERS
from .modules_others.transformer import BlockVanilla


def sincos_1d(n, dim, temperature=10000):
    omega = 1.0 / temperature ** (torch.arange(dim // 2) / (dim // 2))
    angles = torch.arange(n)[:, None] * omega[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class TemporalConv(nn.Module):
    # conditional position: x + zero-init depthwise conv over time (dim 1), zero padding marks the clip ends
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        y = x.movedim(1, -2)
        shape = y.shape
        y = self.conv(y.reshape(-1, shape[-2], shape[-1]).transpose(1, 2)).transpose(1, 2)
        return x + y.reshape(shape).movedim(-2, 1)


def temporal_embedding(module, kind, shape, learned_init):
    # learned: a parameter (learned_init sets its start; learned_zero starts at 0); sincos: fixed;
    # none and tconv: zeros, tconv adding a TemporalConv after it
    if kind in ('learned', 'learned_zero'):
        module.temporal_embedding = nn.Parameter(torch.zeros(shape))
        if kind == 'learned':
            learned_init(module.temporal_embedding)
    elif kind == 'sincos':
        module.register_buffer('temporal_embedding', sincos_1d(shape[1], shape[-1]).reshape(shape), persistent=False)
    else:
        module.register_buffer('temporal_embedding', torch.zeros(shape), persistent=False)
    module.temporal_conv = TemporalConv(shape[-1]) if kind == 'tconv' else nn.Identity()


def block(dim):
    return BlockVanilla(dim, 8, dim * 4, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.0,
                        layer_norm_eps=1e-6)


class TemporalHead(nn.Module):
    # depth pre-norm ViT blocks over per-frame features, per-frame logits averaged
    def __init__(self, dim, temporal_dim, n_frames, num_classes, embedding='learned', depth=1):
        super().__init__()
        temporal_dim = temporal_dim or dim
        assert temporal_dim % 8 == 0, f'--temporal_dim {temporal_dim} must be divisible by 8 attention heads'
        self.proj = nn.Linear(dim, temporal_dim) if temporal_dim != dim else nn.Identity()
        temporal_embedding(self, embedding, (1, n_frames, temporal_dim), lambda p: nn.init.trunc_normal_(p, std=0.02))
        self.encoder = block(temporal_dim)
        self.encoder_rest = nn.ModuleList([block(temporal_dim) for _ in range(depth - 1)])
        self.norm = nn.LayerNorm(temporal_dim, eps=1e-6)
        self.classifier = nn.Linear(temporal_dim, num_classes)

    def forward(self, x):
        x, _, _ = self.encoder(self.temporal_conv(self.proj(x) + self.temporal_embedding))
        for encoder in self.encoder_rest:
            x, _, _ = encoder(x)
        return self.classifier(self.norm(x)).mean(dim=1)


class FrameMerge(nn.Module):
    # (b t) n d -> b (t n) d, each frame's tokens shifted by a temporal embedding (learned: zero-init)
    def __init__(self, dim, n_frames, video_token, embedding='learned'):
        super().__init__()
        self.n_frames = n_frames
        temporal_embedding(self, embedding, (1, n_frames, 1, dim), lambda p: None)
        if video_token:
            self.video_token = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        x = self.temporal_conv(rearrange(x, '(b t) n d -> b t n d', t=self.n_frames) + self.temporal_embedding)
        self.frame_len = x.shape[2]
        x = rearrange(x, 'b t n d -> b (t n) d')
        if hasattr(self, 'video_token'):
            x = torch.cat([repeat(self.video_token, '1 1 d -> b 1 d', b=x.shape[0]), x], dim=1)
        return x


class QKLoRA(nn.Module):
    # zero-init low-rank updates (A: kaiming, B: zeros) of an attention's query and key projections
    def __init__(self, dim, rank):
        super().__init__()
        self.q_a, self.k_a = nn.Linear(dim, rank, bias=False), nn.Linear(dim, rank, bias=False)
        self.q_b, self.k_b = nn.Linear(rank, dim, bias=False), nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.q_b.weight)
        nn.init.zeros_(self.k_b.weight)

    def add_q(self, module, inputs, output):
        return output + self.q_b(self.q_a(inputs[0]))

    def add_k(self, module, inputs, output):
        return output + self.k_b(self.k_a(inputs[0]))

    def add_qk(self, module, inputs, output):
        # fused qkv: q, k, v are the thirds of the last dim
        q, k, v = output.chunk(3, dim=-1)
        return torch.cat([self.add_q(module, inputs, q), self.add_k(module, inputs, k), v], dim=-1)


class VideoModel(nn.Module):
    """Image ViT on clips (b c t h w). temporal: per-frame backbone + TemporalHead.
    joint: blocks before merge_depth run per frame (with any temporal adapter), the rest on the frames' tokens
    as one sequence."""
    def __init__(self, args, backbone):
        super().__init__()
        self.backbone = backbone
        self.cfg = backbone.cfg
        vit = backbone.model
        assert hasattr(vit, 'blocks') or hasattr(vit, 'encoder'), 'video models need a ViT backbone'

        # the backbone returns every token: drop its pooling and classifier
        if hasattr(vit, 'reset_classifier'):
            use_cls = vit.global_pool == 'token'
            assert not args.video_model == 'joint' or vit.num_prefix_tokens == 1, \
                'joint needs exactly one prefix (class) token'
            dim, blocks = vit.embed_dim, vit.blocks
            vit.reset_classifier(0, '')
        else:
            use_cls = 'cls' in self.cfg.classifier
            dim, blocks = self.cfg.hidden_size, vit.encoder.blocks
            del vit.head
        n_frames = args.n_frames

        if args.video_model == 'temporal':
            self.frame_features = (lambda x: x[:, 0]) if use_cls else (lambda x: x.mean(dim=1))
            self.head = TemporalHead(dim, args.temporal_dim, n_frames, args.num_classes,
                                     args.temporal_embedding, args.temporal_depth)
            self.forward = self.forward_temporal
            self.add_qk_lora(args, dim, blocks, 0)
            return

        merge_depth = args.merge_depth
        assert use_cls, 'joint needs a class-token backbone'
        assert 0 <= merge_depth <= len(blocks), f'--merge_depth must be in [0, {len(blocks)}]'
        assert args.adapter != 'convpass', 'convpass adapters need the per-frame grid'
        assert args.adapter not in TEMPORAL_ADAPTERS or merge_depth > 0, \
            f'{args.adapter} adapters sit in the per-frame blocks before the merge: --merge_depth must be > 0'

        self.video_cls = args.video_cls
        self.aggregator = FrameMerge(dim, n_frames, args.video_cls in ('video', 'video_mean'),
                                     args.temporal_embedding)
        self.merge_at_end = merge_depth == len(blocks)
        if not self.merge_at_end:
            blocks[merge_depth].register_forward_pre_hook(
                lambda module, inputs: (self.aggregator(inputs[0]),) + tuple(inputs[1:]))

        head_dim = {'mean': dim, 'concat': dim * n_frames, 'video': dim, 'video_mean': dim * 2}
        self.head_drop = nn.Dropout(getattr(args, 'head_dropout', 0.0))
        self.head = nn.Linear(head_dim[args.video_cls], args.num_classes)
        self.forward = self.forward_joint
        self.add_qk_lora(args, dim, blocks, merge_depth)

    def add_qk_lora(self, args, dim, blocks, lora_from):
        if getattr(args, 'qk_lora', 0):
            assert lora_from < len(blocks), 'joint --qk_lora needs blocks after --merge_depth'
            for block in blocks[lora_from:]:
                attn = block.attn
                attn.qk_lora = QKLoRA(dim, args.qk_lora)
                if hasattr(attn, 'qkv'):
                    attn.qkv.register_forward_hook(attn.qk_lora.add_qk)
                else:
                    attn.proj_q.register_forward_hook(attn.qk_lora.add_q)
                    attn.proj_k.register_forward_hook(attn.qk_lora.add_k)

    def forward_temporal(self, clips, targets=None):
        b = clips.shape[0]
        x = self.backbone(rearrange(clips, 'b c t h w -> (b t) c h w'))
        x = rearrange(self.frame_features(x), '(b t) d -> b t d', b=b)
        return self.head(x)

    def forward_joint(self, clips, targets=None):
        x = self.backbone(rearrange(clips, 'b c t h w -> (b t) c h w'))
        if self.merge_at_end:
            x = self.aggregator(x)

        prefix = 0 if self.video_cls in ('mean', 'concat') else 1
        frame_cls = x[:, prefix::self.aggregator.frame_len]
        if self.video_cls == 'mean':
            x = frame_cls.mean(dim=1)
        elif self.video_cls == 'concat':
            x = rearrange(frame_cls, 'b t d -> b (t d)')
        elif self.video_cls == 'video':
            x = x[:, 0]
        else:
            x = torch.cat([x[:, 0], frame_cls.mean(dim=1)], dim=-1)
        return self.head(self.head_drop(x))
