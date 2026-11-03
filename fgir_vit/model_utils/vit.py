import torch
from torch import nn
from einops import repeat, rearrange

from .modules_others import Transformer, LearnedPositionalEmbedding1D
from .misc import load_pretrained_weights, TEMPORAL_ADAPTERS


class ViT(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        self.debugging = config.debugging

        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels, out_channels=config.hidden_size,
            kernel_size=(config.fh, config.fw), stride=(config.patch_stride, config.patch_stride))

        # Class token
        if 'cls' in config.classifier:
            self.class_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # Positional embedding
        if config.pos_embedding_type == 'learned':
            self.positional_embedding = LearnedPositionalEmbedding1D(config.seq_len, config.hidden_size)

        # Transformer encoder
        self.encoder = Transformer(
            num_layers=config.num_hidden_layers,
            dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            ff_dim=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            layer_norm_eps=config.layer_norm_eps,
            sd=config.sd,

            adapter=config.adapter,
            adapter_dim=config.adapter_dim,
            adapter_kernel_size=config.adapter_kernel_size,
            adapter_path=config.adapter_path,
            adapter_pre_attn=config.adapter_pre_attn,
            n_frames=config.n_frames,
            adapter_scale=config.adapter_scale,
            adapter_depth=config.adapter_depth,
        )

        if config.encoder_norm:
            self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Classifier head
        if config.load_fc_layer:
            self.head = nn.Linear(config.hidden_size, config.num_classes)

        if config.adapter == 'aim':
            # AIM's zero-init temporal embedding, added to every token of a frame before the encoder
            self.adapter_temporal_embedding = nn.Parameter(torch.zeros(1, config.n_frames, 1, config.hidden_size))

        # Initialize weights
        self.init_weights()
        if config.adapter in TEMPORAL_ADAPTERS:
            # their own init, which the xavier pass above overwrote
            for m in self.encoder.blocks.modules():
                if hasattr(m, 'init_weights'):
                    m.init_weights()
        if pretrained:
            load_pretrained_weights(self, config, config.model_name)

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        if hasattr(self, 'positional_embedding'):
            nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)
        if hasattr(self, 'class_token'):
            nn.init.constant_(self.class_token, 0)

    def forward(self, x):
        """
        x (tensor): b c fh fw -> b s d, or b num_classes with a head
        """
        x = self.patch_embedding(x)
        x = rearrange(x, 'b d gh gw -> b (gh gw) d')
        b, _, _ = x.shape

        if hasattr(self, 'class_token'):
            cls_tokens = repeat(self.class_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)  # b s+1 d

        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)
        if hasattr(self, 'adapter_temporal_embedding'):
            x = (x.unflatten(0, (-1, self.adapter_temporal_embedding.shape[1])) + self.adapter_temporal_embedding).flatten(0, 1)

        self.maybe_print('After tokenizing: ', x.shape)

        x = self.encoder(x)
        self.maybe_print('After encoder: ', x.shape)

        if hasattr(self, 'encoder_norm'):
            x = self.encoder_norm(x)

        if hasattr(self, 'head'):
            x = self.head(x[:, 0])
        self.maybe_print('After classifier head: ', x.shape)
        return x

    def maybe_print(self, *args):
        if self.debugging:
            print(*args)
