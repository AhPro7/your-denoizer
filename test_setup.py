#!/usr/bin/env python3
"""
Quick test script — verify everything works before training.
Run this first on Colab to check the setup.

Usage:
    python test_setup.py
"""

import sys
import torch

def test_model():
    """Test model forward pass with all configurations."""
    from models.separator import ConvTasNet
    
    print("=" * 60)
    print("  Model Architecture Test")
    print("=" * 60)
    
    for config_name in ['nano', 'tiny', 'small', 'standard']:
        model = ConvTasNet.from_config(config_name, speaker_dim=0)
        
        batch_size = 2
        mixture = torch.randn(batch_size, 1, 64000)   # 4s at 16kHz
        
        with torch.no_grad():
            output = model(mixture)
        
        total_params = sum(p.numel() for p in model.parameters())
        size_mb = total_params * 4 / 1024 / 1024
        
        print(f"  {config_name:10s} | Params: {total_params:>10,} | "
              f"Size: {size_mb:.1f} MB | Output: {output.shape}")
    
    print("\n  ✅ All model configurations OK!\n")


def test_loss():
    """Test loss functions."""
    from training.losses import SISnrLoss, CombinedLoss
    
    print("=" * 60)
    print("  Loss Function Test")
    print("=" * 60)
    
    B, T = 4, 64000
    estimate = torch.randn(B, 1, T)
    target = torch.randn(B, 1, T)
    
    # SI-SNR
    si_snr = SISnrLoss()
    loss = si_snr(estimate, target)
    print(f"  SI-SNR loss: {loss.item():.4f}")
    
    # Combined
    combined = CombinedLoss()
    losses = combined(estimate, target)
    print(f"  Combined loss: {losses['loss'].item():.4f}")
    print(f"    SI-SNR component: {losses['si_snr'].item():.4f}")
    print(f"    STFT component:   {losses['stft'].item():.4f}")
    
    print("\n  ✅ Loss functions OK!\n")


def test_dataset():
    """Test dataset with a local dummy setup."""
    from training.dataset import (
        HFSpeechSource, HFNoiseSource, SpeechEnhancementDataset,
        LocalSpeechIndex, LocalNoiseIndex,
        rms_normalize, mix_at_snr, random_segment
    )
    
    print("=" * 60)
    print("  Dataset Utilities Test")
    print("=" * 60)
    
    # Test utilities
    wav = torch.randn(32000)  # 2 seconds
    normed = rms_normalize(wav)
    print(f"  rms_normalize: {wav.shape} → RMS={torch.sqrt(torch.mean(normed**2)):.4f}")
    
    seg = random_segment(wav, 64000)
    print(f"  random_segment: {wav.shape} → {seg.shape} (padded/looped)")
    
    noise = torch.randn(64000)
    mixed = mix_at_snr(seg, noise, 10.0)
    print(f"  mix_at_snr: target + noise at 10dB SNR → {mixed.shape}")
    
    print("\n  ✅ Dataset utilities OK!\n")


def test_hf_available():
    """Check if HuggingFace datasets library is available."""
    print("=" * 60)
    print("  HuggingFace Integration Check")
    print("=" * 60)
    
    try:
        import datasets
        print(f"  datasets version: {datasets.__version__}")
        print("  ✅ HuggingFace datasets available")
    except ImportError:
        print("  ⚠️ HuggingFace datasets not installed")
        print("  Run: pip install datasets")
    
    try:
        import torchaudio
        print(f"  torchaudio version: {torchaudio.__version__}")
        print("  ✅ torchaudio available")
    except ImportError:
        print("  ⚠️ torchaudio not installed")
    
    print()


def test_gpu():
    """Check GPU availability."""
    print("=" * 60)
    print("  Hardware Check")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"  ✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"  ✅ Apple MPS available")
    else:
        print(f"  ⚠️ No GPU detected — training will be slow on CPU")
    
    print(f"  PyTorch version: {torch.__version__}")
    print()


if __name__ == '__main__':
    test_gpu()
    test_hf_available()
    test_model()
    test_loss()
    test_dataset()
    
    print("=" * 60)
    print("  🎉 All tests passed! Ready to train.")
    print("=" * 60)
