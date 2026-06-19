"""
Loss functions for target speaker extraction training.

Primary loss: SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
The standard loss for speech separation — measures how well the extracted
signal matches the clean target, invariant to scaling.
"""

import torch
import torch.nn as nn


class SISnrLoss(nn.Module):
    """Scale-Invariant Signal-to-Noise Ratio loss.
    
    SI-SNR = 10 * log10(||s_target||² / ||e_noise||²)
    
    Where:
        s_target = (<estimate, target> / ||target||²) * target
        e_noise  = estimate - s_target
    
    Higher SI-SNR = better. We return negative SI-SNR for minimization.
    
    Args:
        eps: Small constant for numerical stability
        reduction: 'mean' or 'none'
    """
    
    def __init__(self, eps: float = 1e-8, reduction: str = 'mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            estimate: (B, 1, T) or (B, T) — estimated waveform
            target:   (B, 1, T) or (B, T) — clean target waveform
        
        Returns:
            loss: scalar (negative SI-SNR, for minimization)
        """
        # Flatten to (B, T)
        if estimate.dim() == 3:
            estimate = estimate.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        
        # Zero-mean normalization
        estimate = estimate - estimate.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        
        # s_target = (<estimate, target> / ||target||²) * target
        dot = torch.sum(estimate * target, dim=-1, keepdim=True)
        s_target_norm_sq = torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps
        s_target = (dot / s_target_norm_sq) * target
        
        # e_noise = estimate - s_target
        e_noise = estimate - s_target
        
        # SI-SNR = 10 * log10(||s_target||² / ||e_noise||²)
        si_snr = 10 * torch.log10(
            torch.sum(s_target ** 2, dim=-1) / 
            (torch.sum(e_noise ** 2, dim=-1) + self.eps)
            + self.eps
        )
        
        # Return negative for minimization (we want to maximize SI-SNR)
        if self.reduction == 'mean':
            return -si_snr.mean()
        elif self.reduction == 'none':
            return -si_snr
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss for better perceptual quality.
    
    Computes spectral convergence + log magnitude loss at multiple
    STFT resolutions. Helps the model produce cleaner, more natural audio.
    
    Used as an auxiliary loss alongside SI-SNR.
    """
    
    def __init__(self, fft_sizes=(512, 1024, 2048), hop_sizes=(128, 256, 512), 
                 win_sizes=(512, 1024, 2048)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
    
    def _stft_loss(self, estimate, target, fft_size, hop_size, win_size):
        """Compute STFT loss at a single resolution."""
        window = torch.hann_window(win_size, device=estimate.device)
        
        est_stft = torch.stft(estimate, fft_size, hop_size, win_size, 
                               window=window, return_complex=True)
        tgt_stft = torch.stft(target, fft_size, hop_size, win_size,
                               window=window, return_complex=True)
        
        est_mag = torch.abs(est_stft)
        tgt_mag = torch.abs(tgt_stft)
        
        # Spectral convergence loss
        sc_loss = torch.norm(tgt_mag - est_mag, p='fro') / (torch.norm(tgt_mag, p='fro') + 1e-8)
        
        # Log magnitude loss
        log_loss = torch.mean(torch.abs(torch.log(est_mag + 1e-8) - torch.log(tgt_mag + 1e-8)))
        
        return sc_loss + log_loss
    
    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            estimate: (B, T) or (B, 1, T) — estimated waveform
            target:   (B, T) or (B, 1, T) — clean target waveform
        
        Returns:
            loss: scalar — average multi-resolution STFT loss
        """
        if estimate.dim() == 3:
            estimate = estimate.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        
        total_loss = 0
        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            total_loss += self._stft_loss(estimate, target, fft_size, hop_size, win_size)
        
        return total_loss / len(self.fft_sizes)


class CombinedLoss(nn.Module):
    """Combined SI-SNR + Multi-Resolution STFT loss.
    
    Loss = si_snr_weight * (-SI-SNR) + stft_weight * MR-STFT
    
    SI-SNR optimizes signal fidelity. STFT loss improves perceptual quality.
    """
    
    def __init__(self, si_snr_weight: float = 1.0, stft_weight: float = 0.1):
        super().__init__()
        self.si_snr = SISnrLoss()
        self.stft = MultiResolutionSTFTLoss()
        self.si_snr_weight = si_snr_weight
        self.stft_weight = stft_weight
    
    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Returns:
            dict with 'loss' (total), 'si_snr', 'stft' components
        """
        si_snr_loss = self.si_snr(estimate, target)
        stft_loss = self.stft(estimate, target)
        
        total = self.si_snr_weight * si_snr_loss + self.stft_weight * stft_loss
        
        return {
            'loss': total,
            'si_snr': si_snr_loss,
            'stft': stft_loss,
        }
