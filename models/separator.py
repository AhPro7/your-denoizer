"""
Conv-TasNet Separator with Speaker Conditioning (FiLM).

Architecture:
    Mixture Waveform → 1D Conv Encoder → FiLM(speaker_emb) → TCN Mask Network → Decoder → Extracted Voice

The separator learns to produce a mask conditioned on the target speaker's embedding.
Only the masked (target) source is output — all other speakers and noise are suppressed.

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


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation (FiLM) for speaker conditioning.
    
    Transforms speaker embedding into scale (γ) and shift (β) parameters
    that modulate the separator's internal features. This is how the model
    knows WHICH speaker to extract.
    
    FiLM(x, emb) = γ(emb) * x + β(emb)
    """
    
    def __init__(self, speaker_dim: int, feature_dim: int):
        super().__init__()
        self.scale = nn.Linear(speaker_dim, feature_dim)
        self.shift = nn.Linear(speaker_dim, feature_dim)
        
        # Initialize scale near 1.0 and shift near 0.0 for stable start
        nn.init.ones_(self.scale.weight.data.mean(dim=1, keepdim=True).expand_as(self.scale.weight))
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)
    
    def forward(self, features: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, T) — encoded audio features
            speaker_emb: (B, D) — speaker embedding from ECAPA-TDNN
        
        Returns:
            Conditioned features: (B, C, T)
        """
        gamma = self.scale(speaker_emb).unsqueeze(-1)  # (B, C, 1)
        beta = self.shift(speaker_emb).unsqueeze(-1)    # (B, C, 1)
        return gamma * features + beta


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution — much more parameter-efficient
    than standard conv. This is what makes Conv-TasNet lightweight."""
    
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
    """Single Temporal Convolutional Network block with residual connection.
    
    Structure: 1x1 Conv → PReLU → Norm → DW-SepConv → PReLU → Norm → 1x1 Conv → Residual
    Uses exponentially increasing dilation for large receptive field.
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, 
                 kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        
        self.net = nn.Sequential(
            # Bottleneck expansion
            nn.Conv1d(in_channels, hidden_channels, 1),
            nn.PReLU(),
            nn.GroupNorm(1, hidden_channels),  # Global LayerNorm equivalent
            # Depthwise separable dilated conv
            DepthwiseSeparableConv1d(
                hidden_channels, in_channels, kernel_size,
                padding=padding, dilation=dilation
            ),
            nn.PReLU(),
            nn.GroupNorm(1, in_channels),
        )
        
        # Skip connection projection
        self.skip_proj = nn.Conv1d(in_channels, in_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            out: (B, C, T) — residual output
        """
        residual = self.net(x)
        return x + residual


class TCNSeparator(nn.Module):
    """Temporal Convolutional Network mask generator.
    
    Stacks multiple TCN blocks with exponentially increasing dilation rates
    to build a large receptive field. Multiple repeats of the stack further
    increase capacity.
    
    Args:
        in_channels (B): Number of input channels (bottleneck dim)
        hidden_channels (H): Hidden channels in TCN blocks
        kernel_size (P): Kernel size for depthwise convolutions
        num_layers (X): Number of TCN blocks per repeat
        num_stacks (R): Number of repeating stacks
        num_sources: Number of output sources (1 for target extraction)
        mask_channels (N): Number of channels for the output mask
    """
    
    def __init__(self, in_channels: int = 128, hidden_channels: int = 256,
                 kernel_size: int = 3, num_layers: int = 7, num_stacks: int = 2,
                 num_sources: int = 1, mask_channels: int = 256):
        super().__init__()
        self.num_sources = num_sources
        
        # Build TCN blocks
        blocks = []
        for r in range(num_stacks):
            for x in range(num_layers):
                dilation = 2 ** x
                blocks.append(
                    TCNBlock(in_channels, hidden_channels, kernel_size, dilation)
                )
        self.tcn = nn.Sequential(*blocks)
        
        # Output mask projection
        self.mask_proj = nn.Sequential(
            nn.Conv1d(in_channels, mask_channels * num_sources, 1),
            nn.Sigmoid()  # Mask values in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, B_channels, T) — bottleneck features
        Returns:
            mask: (B, num_sources, N, T) — estimated masks
        """
        B, _, T = x.shape
        out = self.tcn(x)
        mask = self.mask_proj(out)
        
        if self.num_sources == 1:
            return mask  # (B, N, T)
        else:
            return mask.view(B, self.num_sources, -1, T)


class ConvTasNetTSE(nn.Module):
    """Conv-TasNet for Target Speaker Extraction.
    
    Full pipeline:
        1. Encode mixture waveform → latent features
        2. Bottleneck projection
        3. FiLM conditioning with speaker embedding
        4. TCN separator generates mask
        5. Apply mask to encoded features
        6. Decode back to waveform
    
    The model extracts ONLY the target speaker (matched by enrollment embedding),
    suppressing all other speakers and noise.
    
    Args:
        enc_dim (N): Encoder output dimension / number of filters
        enc_kernel_size (L): Encoder kernel size (determines latency)
        bottleneck_dim (B): Bottleneck dimension for TCN input
        hidden_dim (H): Hidden dimension in TCN blocks
        tcn_kernel_size (P): Kernel size for TCN depthwise convs
        num_layers (X): Number of TCN blocks per stack
        num_stacks (R): Number of TCN stack repeats
        speaker_dim: Dimension of speaker embedding (192 for ECAPA-TDNN)
    """
    
    # Preset configurations
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
        speaker_dim: int = 192,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.enc_dim = enc_dim
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2  # 50% overlap
        self.sample_rate = sample_rate
        
        # ===== Encoder =====
        # 1D Conv that converts waveform to latent representation
        # Like a learnable STFT but in time domain
        self.encoder = nn.Conv1d(
            1, enc_dim, enc_kernel_size, 
            stride=self.enc_stride, bias=False
        )
        
        # ===== Bottleneck =====
        # Reduce dimensionality before TCN
        self.bottleneck = nn.Sequential(
            nn.GroupNorm(1, enc_dim),
            nn.Conv1d(enc_dim, bottleneck_dim, 1),
        )
        
        # ===== Speaker Conditioning (FiLM) =====
        self.film = FiLMConditioner(speaker_dim, bottleneck_dim)
        
        # ===== Separator (TCN) =====
        self.separator = TCNSeparator(
            in_channels=bottleneck_dim,
            hidden_channels=hidden_dim,
            kernel_size=tcn_kernel_size,
            num_layers=num_layers,
            num_stacks=num_stacks,
            num_sources=1,
            mask_channels=enc_dim,
        )
        
        # ===== Decoder =====
        # Transposed conv to reconstruct waveform from masked features
        self.decoder = nn.ConvTranspose1d(
            enc_dim, 1, enc_kernel_size,
            stride=self.enc_stride, bias=False
        )
        
        # Track parameter count
        self._log_param_count()
    
    def _log_param_count(self):
        """Log parameter counts for each component."""
        counts = {}
        for name, module in [('encoder', self.encoder), ('bottleneck', self.bottleneck),
                              ('film', self.film), ('separator', self.separator),
                              ('decoder', self.decoder)]:
            counts[name] = sum(p.numel() for p in module.parameters())
        total = sum(counts.values())
        self._param_counts = counts
        self._total_params = total
    
    @classmethod
    def from_config(cls, config_name: str = 'tiny', **kwargs):
        """Create model from preset configuration.
        
        Args:
            config_name: One of 'nano', 'tiny', 'small', 'standard'
        """
        if config_name not in cls.CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Choose from {list(cls.CONFIGS.keys())}")
        
        config = cls.CONFIGS[config_name].copy()
        config.update(kwargs)
        return cls(**config)
    
    def forward(self, mixture: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        """
        Extract target speaker from mixture audio.
        
        Args:
            mixture: (B, 1, T) or (B, T) — mixture waveform at 16kHz
            speaker_emb: (B, D) — speaker embedding of target speaker
        
        Returns:
            extracted: (B, 1, T) — extracted target speaker waveform
        """
        # Handle shape
        if mixture.dim() == 2:
            mixture = mixture.unsqueeze(1)  # (B, T) → (B, 1, T)
        
        input_length = mixture.shape[-1]
        
        # 1. Encode
        encoded = self.encoder(mixture)   # (B, N, T')
        encoded = F.relu(encoded)
        
        # 2. Bottleneck
        bottleneck = self.bottleneck(encoded)  # (B, B, T')
        
        # 3. Speaker conditioning via FiLM
        conditioned = self.film(bottleneck, speaker_emb)  # (B, B, T')
        
        # 4. Generate mask
        mask = self.separator(conditioned)  # (B, N, T')
        
        # 5. Apply mask to encoded mixture
        masked = encoded * mask  # (B, N, T')
        
        # 6. Decode back to waveform
        extracted = self.decoder(masked)  # (B, 1, T_out)
        
        # 7. Trim to match input length
        extracted = extracted[:, :, :input_length]
        
        return extracted
    
    def get_param_summary(self) -> str:
        """Get a human-readable parameter count summary."""
        lines = ["Parameter Summary:"]
        for name, count in self._param_counts.items():
            lines.append(f"  {name:15s}: {count:>10,d}")
        lines.append(f"  {'TOTAL':15s}: {self._total_params:>10,d}")
        lines.append(f"  Model size (FP32): ~{self._total_params * 4 / 1024 / 1024:.1f} MB")
        return "\n".join(lines)


def build_model(config_name: str = 'tiny', speaker_dim: int = 192, **kwargs) -> ConvTasNetTSE:
    """Build a ConvTasNet-TSE model from a preset config.
    
    Args:
        config_name: 'nano', 'tiny', 'small', or 'standard'
        speaker_dim: Dimension of speaker embeddings (192 for ECAPA-TDNN)
    
    Returns:
        Initialized ConvTasNetTSE model
    """
    model = ConvTasNetTSE.from_config(config_name, speaker_dim=speaker_dim, **kwargs)
    print(model.get_param_summary())
    return model
