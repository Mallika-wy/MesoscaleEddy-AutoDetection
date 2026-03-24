from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def _merge_batch_time(x: torch.Tensor) -> torch.Tensor:
    batch, time, channels, height, width = x.shape
    return x.reshape(batch * time, channels, height, width)


def _restore_batch_time(x: torch.Tensor, batch: int, time: int) -> torch.Tensor:
    channels, height, width = x.shape[1:]
    return x.reshape(batch, time, channels, height, width)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, reduced, kernel_size=1)
        self.bn = nn.BatchNorm2d(reduced)
        self.conv2 = nn.Conv2d(reduced, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.avg_pool(x)
        weights = self.conv1(weights)
        weights = self.bn(weights)
        weights = F.relu(weights, inplace=True)
        weights = torch.sigmoid(self.conv2(weights))
        return x * weights


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, use_ca: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout2d(dropout)
        self.channel_attention = ChannelAttention(out_channels) if use_ca else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.channel_attention(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = F.relu(out + residual, inplace=True)
        return out


class PositionAttentionFusion(nn.Module):
    def __init__(self, encoder_channels: int, decoder_channels: int) -> None:
        super().__init__()
        self.encoder_proj = nn.Conv2d(encoder_channels, decoder_channels, kernel_size=1)
        self.decoder_proj = nn.Conv2d(decoder_channels, decoder_channels, kernel_size=1)
        self.weight_proj = nn.Conv2d(decoder_channels, decoder_channels, kernel_size=1)

    def forward(self, encoder: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        batch, time = encoder.shape[:2]
        encoder_flat = _merge_batch_time(encoder)
        decoder_flat = _merge_batch_time(decoder)
        weights = self.encoder_proj(encoder_flat) + self.decoder_proj(decoder_flat)
        weights = F.relu(weights, inplace=True)
        weights = torch.sigmoid(self.weight_proj(weights))
        weighted_decoder = decoder_flat * weights
        fused = torch.cat([weighted_decoder, encoder_flat], dim=1)
        return _restore_batch_time(fused, batch, time)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(4 * hidden_channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = hidden
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.dropout(self.bn(self.gates(combined)))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, batch: int, spatial_shape: tuple[int, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = spatial_shape
        zeros = torch.zeros(batch, self.hidden_channels, height, width, device=device)
        return zeros, zeros.clone()


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dropout: float,
        return_sequence: bool,
    ) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size, dropout)
        self.return_sequence = return_sequence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, _, height, width = x.shape
        hidden = self.cell.init_hidden(batch, (height, width), x.device)
        outputs: list[torch.Tensor] = []
        for step in range(time):
            hidden = self.cell(x[:, step], hidden)
            outputs.append(hidden[0])
        if self.return_sequence:
            return torch.stack(outputs, dim=1)
        return outputs[-1]


@dataclass
class DecoderStageSpec:
    encoder_channels: int
    decoder_channels: int
    out_channels: int
    return_sequence: bool


class DecoderStage(nn.Module):
    def __init__(self, spec: DecoderStageSpec, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.position_attention = PositionAttentionFusion(spec.encoder_channels, spec.decoder_channels)
        self.block = ResidualBlock(spec.encoder_channels + spec.decoder_channels, spec.out_channels, dropout, use_ca=True)
        self.temporal = ConvLSTM(spec.out_channels, spec.out_channels, kernel_size, dropout, spec.return_sequence)

    def forward(self, encoder: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        batch, time = decoder.shape[:2]
        decoder_up = F.interpolate(
            _merge_batch_time(decoder),
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        decoder_up = _restore_batch_time(decoder_up, batch, time)
        fused = self.position_attention(encoder, decoder_up)
        fused_flat = _merge_batch_time(fused)
        out = self.block(fused_flat)
        out = _restore_batch_time(out, batch, time)
        return self.temporal(out)


class DualAttentionConvLSTMUNet(nn.Module):
    def __init__(self, in_channels: int = 3, seq_len: int = 3, num_classes: int = 3, channels: list[int] | None = None, dropout: float = 0.1, kernel_size: int = 3) -> None:
        super().__init__()
        self.seq_len = seq_len
        widths = channels or [16, 32, 48, 64, 96]
        if len(widths) != 5:
            raise ValueError("channels must contain 5 width values")

        self.enc1 = ResidualBlock(in_channels, widths[0], dropout)
        self.enc2 = ResidualBlock(widths[0], widths[1], dropout)
        self.enc3 = ResidualBlock(widths[1], widths[2], dropout)
        self.enc4 = ResidualBlock(widths[2], widths[3], dropout)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(widths[3], widths[4], dropout)

        self.dec4 = DecoderStage(DecoderStageSpec(widths[3], widths[4], widths[3], True), kernel_size, dropout)
        self.dec3 = DecoderStage(DecoderStageSpec(widths[2], widths[3], widths[2], True), kernel_size, dropout)
        self.dec2 = DecoderStage(DecoderStageSpec(widths[1], widths[2], widths[1], True), kernel_size, dropout)
        self.dec1 = DecoderStage(DecoderStageSpec(widths[0], widths[1], widths[0], False), kernel_size, dropout)
        self.head = nn.Conv2d(widths[0], num_classes, kernel_size=1)

    @classmethod
    def from_config(cls, config: ModelConfig, in_channels: int, seq_len: int) -> "DualAttentionConvLSTMUNet":
        return cls(
            in_channels=in_channels,
            seq_len=seq_len,
            num_classes=config.num_classes,
            channels=config.channels,
            dropout=config.dropout,
            kernel_size=config.kernel_size,
        )

    def _encode(self, x: torch.Tensor, block: nn.Module) -> torch.Tensor:
        batch, time = x.shape[:2]
        flat = _merge_batch_time(x)
        encoded = block(flat)
        return _restore_batch_time(encoded, batch, time)

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        batch, time = x.shape[:2]
        down = self.pool(_merge_batch_time(x))
        return _restore_batch_time(down, batch, time)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self._encode(x, self.enc1)
        e2 = self._encode(self._downsample(e1), self.enc2)
        e3 = self._encode(self._downsample(e2), self.enc3)
        e4 = self._encode(self._downsample(e3), self.enc4)
        b = self._encode(self._downsample(e4), self.bottleneck)

        d4 = self.dec4(e4, b)
        d3 = self.dec3(e3, d4)
        d2 = self.dec2(e2, d3)
        d1 = self.dec1(e1, d2)
        return self.head(d1)
