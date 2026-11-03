import torch
from torch import nn

# Frame2Freq temporal adapters (arXiv:2602.18977, github.com/th-nesh/Frame2Freq)
# _og: outputs equal to the release's (channels mixed by a view, center=False stft with center=True istft)


def split_prefix(x):
    n = x.shape[1]
    h = int((n - 1) ** 0.5)
    return x[:, :n - h * h], x[:, n - h * h:], h


class Frame2FreqMS(nn.Module):
    # down, temporal conv on half the channels and multi-scale spectral conv on the rest, up
    def __init__(self, dim, adapter_dim, n_frames, kernel_size=(3, 1, 1), og=False):
        super().__init__()
        self.n_frames = n_frames
        self.og = og
        self.windows = (n_frames, n_frames // 2, n_frames // 4)
        assert all(w > 0 and n_frames % w == 0 for w in self.windows), \
            f'frame2freq_ms windows {self.windows} must divide --n_frames {n_frames}'
        self.ct = adapter_dim // 2
        self.cf = adapter_dim - self.ct
        padding = tuple(k // 2 for k in kernel_size)

        self.fc1 = nn.Linear(dim, adapter_dim)
        self.temporal_conv = nn.Conv3d(self.ct, self.ct, kernel_size, padding=padding, groups=self.ct)
        self.freq_conv = nn.Conv3d(2 * self.cf, 2 * self.cf, kernel_size, padding=padding, groups=2 * self.cf)
        self.fc2 = nn.Linear(adapter_dim, dim)
        self.init_weights()

    def init_weights(self):
        # zero biases; also called by the backbone's init
        for m in (self.fc1, self.fc2, self.temporal_conv, self.freq_conv):
            m.reset_parameters()
            nn.init.zeros_(m.bias)

    def forward(self, x):
        prefix, patches, h = split_prefix(x)
        t = self.n_frames
        cf = self.cf
        z = self.fc1(patches)
        z = z.view(-1, t, h, h, z.shape[-1])
        b = z.shape[0]
        zt = z[..., :self.ct]
        zf = z[..., self.ct:]

        # temporal branch
        zt = self.temporal_conv(zt.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

        # spectral branch, averaged over window sizes

        out_f = torch.zeros_like(zf)
        for w in self.windows:
            chunks = t // w
            spec = torch.fft.rfft(zf.reshape(b, chunks, w, h, h, cf), dim=2)
            spec = torch.view_as_real(spec).contiguous()
            freqs = spec.shape[2]
            if self.og:
                spec = self.freq_conv(spec.view(b * chunks, 2 * cf, freqs, h, h))
                spec = spec.view(b, chunks, freqs, h, h, cf, 2)
            else:
                # real and imaginary parts per channel
                spec = spec.permute(0, 1, 5, 6, 2, 3, 4).reshape(b * chunks, 2 * cf, freqs, h, h)
                spec = self.freq_conv(spec)
                spec = spec.view(b, chunks, cf, 2, freqs, h, h).permute(0, 1, 4, 5, 6, 2, 3)
            spec = torch.complex(spec[..., 0], spec[..., 1])
            out_w = torch.fft.irfft(spec, n=w, dim=2).reshape(b, t, h, h, cf)
            out_f = out_f + out_w
        out_f = out_f / len(self.windows)

        z = torch.cat([zt, out_f], dim=-1).reshape(b * t, h * h, -1)
        return torch.cat([prefix, patches + self.fc2(z)], dim=1)


class Frame2FreqST(nn.Module):
    # down, stft over frames, two zero-init depthwise convs, istft, up; identity at init
    def __init__(self, dim, adapter_dim, n_frames, og=False):
        super().__init__()
        self.n_frames = n_frames
        self.og = og
        self.n_fft = min(32, n_frames // 2)
        self.hop = max(1, self.n_fft // 4)
        if (n_frames - self.n_fft) // self.hop + 1 < 2:
            self.n_fft = n_frames // 4
            self.hop = max(1, self.n_fft // 4)

        self.fc1 = nn.Linear(dim, adapter_dim)
        self.conv_temporal = nn.Conv3d(2 * adapter_dim, 2 * adapter_dim, (3, 1, 1), padding=(1, 0, 0),
                                       groups=2 * adapter_dim)
        self.conv_cross = nn.Conv3d(2 * adapter_dim, 2 * adapter_dim, (3, 3, 1), padding=(1, 1, 0),
                                    groups=2 * adapter_dim)
        self.fc2 = nn.Linear(adapter_dim, dim)
        self.init_weights()

    def init_weights(self):
        # zero convs and biases
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        for p in (self.fc1.bias, self.fc2.bias, self.conv_temporal.weight, self.conv_temporal.bias,
                  self.conv_cross.weight, self.conv_cross.bias):
            nn.init.zeros_(p)

    def forward(self, x):
        prefix, patches, h = split_prefix(x)
        t = self.n_frames
        z = self.fc1(patches)
        ca = z.shape[-1]
        z = z.view(-1, t, h, h, ca)
        b = z.shape[0]
        bhw = b * h * h

        window = torch.hann_window(self.n_fft, device=z.device, dtype=z.dtype)
        series = z.permute(0, 2, 3, 4, 1).reshape(-1, t)
        # center=True's reflection pad by hand: cuda's pad kernel takes at most 65,535 rows
        if not self.og:
            p = self.n_fft // 2
            series = torch.cat([series[:, 1:p + 1].flip(-1), series, series[:, -p - 1:-1].flip(-1)], -1)
        spec = torch.stft(series, n_fft=self.n_fft, hop_length=self.hop, window=window,
                          return_complex=True, center=False)
        freqs, steps = spec.shape[1:]
        spec = torch.view_as_real(spec.view(bhw, ca, freqs, steps)).contiguous()
        if self.og:
            spec = spec.view(bhw, 2 * ca, freqs, steps, 1)
            spec = self.conv_temporal(spec) + self.conv_cross(spec)
            spec = spec.permute(0, 2, 3, 1, 4).contiguous()[..., 0].view(bhw, freqs, steps, ca, 2)
            spec = torch.complex(spec[..., 0], spec[..., 1]).permute(0, 3, 1, 2)
        else:
            # real and imaginary parts per channel
            spec = spec.permute(0, 1, 4, 2, 3).reshape(bhw, 2 * ca, freqs, steps, 1)
            spec = self.conv_temporal(spec) + self.conv_cross(spec)
            spec = spec[..., 0].view(bhw, ca, 2, freqs, steps)
            spec = torch.complex(spec[:, :, 0], spec[:, :, 1])
        # istft with center=True in both
        spec = spec.reshape(-1, freqs, steps)
        y = torch.istft(spec, n_fft=self.n_fft, hop_length=self.hop, window=window, length=t)
        y = y.view(b, h, h, ca, t).permute(0, 4, 1, 2, 3).reshape(b * t, h * h, ca)
        return torch.cat([prefix, patches + self.fc2(y)], dim=1)


def build_frame2freq(adapter, dim, adapter_dim, n_frames):
    og = adapter.endswith('_og')
    if adapter.startswith('frame2freq_ms'):
        return Frame2FreqMS(dim, adapter_dim, n_frames, og=og)
    return Frame2FreqST(dim, adapter_dim, n_frames, og=og)
