"""
Conv-TasNet Speech Enhancement / Denoiser.

Two modes (switchable):
    Phase 1 (current): Speech Enhancement — noisy audio → clean audio
    Phase 2 (future):  Target Speaker Extraction — noisy + enrollment → isolated voice

Architecture:
    Noisy Waveform → 1D Conv Encoder → Bottleneck → TCN Separator → Mask → Decoder → Clean Waveform

When speaker conditioning is enabled (Phase 2), FiLM injects the target
speaker's identity between the bottleneck and separator.

Configurations:
    - Nano:     ~0.5M params  (N=128, B=64,  H=128, X=6, R=2)
    - Tiny:     ~1.3M params  (N=256, B=128, H=256, X=7, R=2)  ← recommended
    - Small:    ~2.5M params  (N=256, B=128, H=512, X=8, R=3)
    - Standard: ~5.1M params  (N=512, B=128, H=512, X=8, R=3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation (FiLM) for speaker conditioning.
    
    Used in Phase 2 (personalization) to tell the model which speaker to extract.
    FiLM(x, emb) = γ(emb) * x + β(emb)
    """
    
    def __init__(self, speaker_dim: int, feature_dim: int):
        super().__init__()
        self.scale = nn.Linear(speaker_dim, feature_dim)
        self.shift = nn.Linear(speaker_dim, feature_dim)
        
        # Initialize as identity transform (γ=1, β=0)
        nn.init.ones_(self.scale.bias)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)
    
    def forward(self, features: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, T) — encoded audio features
            speaker_emb: (B, D) — speaker embedding
        Returns:
            Conditioned features: (B, C, T)
        """
        gamma = self.scale(speaker_emb).unsqueeze(-1)  # (B, C, 1)
        beta = self.shift(speaker_emb).unsqueeze(-1)    # (B, C, 1)
        return gamma * features + beta


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution — parameter-efficient."""
    
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int,
                 padding: int = 0, dilation: int = 1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=padding, dilation=dilation, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, hidden_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class TCNBlock(nn.Module):
    """Single Temporal Convolutional Network block with residual connection."""
    
    def __init__(self, in_channels: int, hidden_channels: int, 
                 kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            nn.PReLU(),
            nn.GroupNorm(1, hidden_channels),
            DepthwiseSeparableConv1d(
                hidden_channels, in_channels, kernel_size,
                padding=padding, dilation=dilation
            ),
            nn.PReLU(),
            nn.GroupNorm(1, in_channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TCNSeparator(nn.Module):
    """Temporal Convolutional Network mask generator."""
    
    def __init__(self, in_channels: int = 128, hidden_channels: int = 256,
                 kernel_size: int = 3, num_layers: int = 7, num_stacks: int = 2,
                 mask_channels: int = 256):
        super().__init__()
        
        blocks = []
        for r in range(num_stacks):
            for x in range(num_layers):
                dilation = 2 ** x
                blocks.append(
                    TCNBlock(in_channels, hidden_channels, kernel_size, dilation)
                )
        self.tcn = nn.Sequential(*blocks)
        
        self.mask_proj = nn.Sequential(
            nn.Conv1d(in_channels, mask_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tcn(x)
        return self.mask_proj(out)


class ConvTasNet(nn.Module):
    """Conv-TasNet for Speech Enhancement (Phase 1) and Target Speaker Extraction (Phase 2).
    
    Phase 1 — Speech Enhancement (current):
        forward(noisy_audio) → clean_audio
        No speaker embedding needed. Removes noise, keeps speech.
    
    Phase 2 — Target Speaker Extraction (future):
        forward(mixture, speaker_emb) → isolated_voice
        Speaker embedding activates FiLM conditioning.
    
    Args:
        enc_dim (N): Encoder output dimension
        enc_kernel_size (L): Encoder kernel size
        bottleneck_dim (B): Bottleneck dimension
        hidden_dim (H): Hidden dimension in TCN blocks
        tcn_kernel_size (P): Kernel size for TCN
        num_layers (X): TCN blocks per stack
        num_stacks (R): Number of stacks
        speaker_dim: Speaker embedding dim (0 = no speaker conditioning)
    """
    
    CONFIGS = {
        'nano':     {'enc_dim': 128, 'bottleneck_dim': 64,  'hidden_dim': 128, 'num_layers': 6, 'num_stacks': 2},
        'tiny':     {'enc_dim': 256, 'bottleneck_dim': 128, 'hidden_dim': 256, 'num_layers': 7, 'num_stacks': 2},
        'small':    {'enc_dim': 256, 'bottleneck_dim': 128, 'hidden_dim': 512, 'num_layers': 8, 'num_stacks': 3},
        'standard': {'enc_dim': 512, 'bottleneck_dim': 128, 'hidden_dim': 512, 'num_layers': 8, 'num_stacks': 3},
    }
    
    def __init__(
        self,
        enc_dim: int = 256,
        enc_kernel_size: int = 20,
        bottleneck_dim: int = 128,
        hidden_dim: int = 256,
        tcn_kernel_size: int = 3,
        num_layers: int = 7,
        num_stacks: int = 2,
        speaker_dim: int = 0,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.enc_dim = enc_dim
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2
        self.sample_rate = sample_rate
        self.speaker_dim = speaker_dim
        self.use_speaker_cond = speaker_dim > 0
        
        # ===== Encoder =====
        self.encoder = nn.Conv1d(
            1, enc_dim, enc_kernel_size,
            stride=self.enc_stride, bias=False
        )
        
        # ===== Bottleneck =====
        self.bottleneck = nn.Sequential(
            nn.GroupNorm(1, enc_dim),
            nn.Conv1d(enc_dim, bottleneck_dim, 1),
        )
        
        # ===== Speaker Conditioning (Phase 2, optional) =====
        if self.use_speaker_cond:
            self.film = FiLMConditioner(speaker_dim, bottleneck_dim)
        else:
            self.film = None
        
        # ===== Separator (TCN) =====
        self.separator = TCNSeparator(
            in_channels=bottleneck_dim,
            hidden_channels=hidden_dim,
            kernel_size=tcn_kernel_size,
            num_layers=num_layers,
            num_stacks=num_stacks,
            mask_channels=enc_dim,
        )
        
        # ===== Decoder =====
        self.decoder = nn.ConvTranspose1d(
            enc_dim, 1, enc_kernel_size,
            stride=self.enc_stride, bias=False
        )
        
        self._log_param_count()
    
    def _log_param_count(self):
        counts = {}
        for name, module in [('encoder', self.encoder), ('bottleneck', self.bottleneck),
                              ('separator', self.separator), ('decoder', self.decoder)]:
            counts[name] = sum(p.numel() for p in module.parameters())
        if self.film:
            counts['film'] = sum(p.numel() for p in self.film.parameters())
        total = sum(counts.values())
        self._param_counts = counts
        self._total_params = total
    
    @classmethod
    def from_config(cls, config_name: str = 'tiny', **kwargs):
        """Create model from preset configuration."""
        if config_name not in cls.CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Choose from {list(cls.CONFIGS.keys())}")
        config = cls.CONFIGS[config_name].copy()
        config.update(kwargs)
        return cls(**config)
    
    def forward(self, mixture: torch.Tensor, 
                speaker_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Denoise / extract speech from noisy audio.
        
        Phase 1 (Enhancement):
            model(noisy_audio) → clean_audio
        
        Phase 2 (TSE, future):
            model(noisy_audio, speaker_embedding) → target_speaker_audio
        
        Args:
            mixture: (B, 1, T) or (B, T) — noisy input waveform at 16kHz
            speaker_emb: (B, D) — OPTIONAL speaker embedding (Phase 2 only)
        
        Returns:
            clean: (B, 1, T) — denoised/extracted waveform
        """
        if mixture.dim() == 2:
            mixture = mixture.unsqueeze(1)
        
        input_length = mixture.shape[-1]
        
        # 1. Encode
        encoded = self.encoder(mixture)
        encoded = F.relu(encoded)
        
        # 2. Bottleneck
        bottleneck = self.bottleneck(encoded)
        
        # 3. Speaker conditioning (only if embedding provided AND model supports it)
        if self.use_speaker_cond and speaker_emb is not None:
            bottleneck = self.film(bottleneck, speaker_emb)
        
        # 4. Generate mask
        mask = self.separator(bottleneck)
        
        # 5. Apply mask
        masked = encoded * mask
        
        # 6. Decode
        clean = self.decoder(masked)
        
        # 7. Trim to input length
        clean = clean[:, :, :input_length]
        
        return clean
    
    def get_param_summary(self) -> str:
        lines = [f"ConvTasNet ({'Enhancement' if not self.use_speaker_cond else 'TSE'}) — Parameter Summary:"]
        for name, count in self._param_counts.items():
            lines.append(f"  {name:15s}: {count:>10,d}")
        lines.append(f"  {'TOTAL':15s}: {self._total_params:>10,d}")
        lines.append(f"  Model size (FP32): ~{self._total_params * 4 / 1024 / 1024:.1f} MB")
        if not self.use_speaker_cond:
            lines.append(f"  Mode: Speech Enhancement (no speaker conditioning)")
        else:
            lines.append(f"  Mode: Target Speaker Extraction (speaker_dim={self.speaker_dim})")
        return "\n".join(lines)


# Keep backward-compatible alias
ConvTasNetTSE = ConvTasNet


def build_model(config_name: str = 'tiny', speaker_dim: int = 0, **kwargs) -> ConvTasNet:
    """Build a ConvTasNet model.
    
    Args:
        config_name: 'nano', 'tiny', 'small', or 'standard'
        speaker_dim: 0 = speech enhancement, 192 = target speaker extraction
    
    Returns:
        Initialized ConvTasNet model
    """
    model = ConvTasNet.from_config(config_name, speaker_dim=speaker_dim, **kwargs)
    print(model.get_param_summary())
    return model
