"""
Evaluation script — Compute metrics on test mixtures.

Metrics:
    - SI-SNR / SI-SNRi (signal quality improvement)
    - PESQ (perceptual quality, MOS-like)
    - STOI (intelligibility)
    - Speaker Similarity (cosine similarity of ECAPA-TDNN embeddings)
    - RTF (real-time factor — inference speed)

Usage:
    python -m training.evaluate \
        --checkpoint checkpoints/best.pt \
        --test-dir data/test_mixtures/ \
        --enrollment data/enrollment/
"""

import argparse
import time
import sys
import os
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.separator import ConvTasNetTSE
from models.speaker_encoder import SpeakerEncoder
from training.losses import SISnrLoss


def compute_si_snr(estimate: torch.Tensor, target: torch.Tensor) -> float:
    """Compute SI-SNR in dB (higher = better)."""
    estimate = estimate - estimate.mean()
    target = target - target.mean()
    
    dot = torch.sum(estimate * target)
    s_target = (dot / (torch.sum(target ** 2) + 1e-8)) * target
    e_noise = estimate - s_target
    
    si_snr = 10 * torch.log10(
        torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + 1e-8) + 1e-8
    )
    return si_snr.item()


def compute_pesq(estimate: np.ndarray, target: np.ndarray, sr: int = 16000) -> float:
    """Compute PESQ (Perceptual Evaluation of Speech Quality).
    Returns MOS-LQO score (1.0 = bad, 4.5 = excellent)."""
    try:
        from pesq import pesq
        score = pesq(sr, target, estimate, 'wb')  # Wideband mode
        return score
    except Exception as e:
        print(f"  PESQ computation failed: {e}")
        return float('nan')


def compute_stoi(estimate: np.ndarray, target: np.ndarray, sr: int = 16000) -> float:
    """Compute STOI (Short-Time Objective Intelligibility).
    Returns score in [0, 1], higher = more intelligible."""
    try:
        from pystoi import stoi
        score = stoi(target, estimate, sr, extended=False)
        return score
    except Exception as e:
        print(f"  STOI computation failed: {e}")
        return float('nan')


def evaluate_sample(
    model: ConvTasNetTSE,
    mixture_path: str,
    target_path: str,
    enrollment_path: str,
    speaker_encoder: SpeakerEncoder,
    device: str = "cpu",
    sample_rate: int = 16000,
) -> Dict[str, float]:
    """Evaluate model on a single test sample.
    
    Args:
        model: Trained TSE model
        mixture_path: Path to mixture audio
        target_path: Path to clean target audio (ground truth)
        enrollment_path: Path to enrollment audio
        speaker_encoder: ECAPA-TDNN encoder
        device: Inference device
    
    Returns:
        Dict of metric name → value
    """
    # Load audio
    mixture, _ = torchaudio.load(mixture_path)
    target, _ = torchaudio.load(target_path)
    
    mixture = mixture.squeeze(0)
    target = target.squeeze(0)
    
    # Ensure same length
    min_len = min(len(mixture), len(target))
    mixture = mixture[:min_len]
    target = target[:min_len]
    
    # Get speaker embedding
    speaker_emb = speaker_encoder.encode_file(enrollment_path)
    
    # Run inference
    model.eval()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        mixture_input = mixture.unsqueeze(0).to(device)
        emb_input = speaker_emb.unsqueeze(0).to(device)
        estimated = model(mixture_input, emb_input)
        estimated = estimated.squeeze().cpu()
    
    elapsed = time.perf_counter() - start_time
    duration = len(mixture) / sample_rate
    
    # Ensure same length after model processing
    min_len = min(len(estimated), len(target), len(mixture))
    estimated = estimated[:min_len]
    target = target[:min_len]
    mixture = mixture[:min_len]
    
    # Compute metrics
    metrics = {}
    
    # SI-SNR
    metrics['si_snr_input'] = compute_si_snr(mixture, target)
    metrics['si_snr_output'] = compute_si_snr(estimated, target)
    metrics['si_snri'] = metrics['si_snr_output'] - metrics['si_snr_input']
    
    # PESQ
    metrics['pesq'] = compute_pesq(
        estimated.numpy(), target.numpy(), sample_rate
    )
    
    # STOI
    metrics['stoi'] = compute_stoi(
        estimated.numpy(), target.numpy(), sample_rate
    )
    
    # Speaker similarity
    with torch.no_grad():
        est_emb = speaker_encoder.encode_batch(estimated.unsqueeze(0))
        est_emb = est_emb.squeeze(0)
    metrics['speaker_similarity'] = SpeakerEncoder.compute_similarity(speaker_emb, est_emb)
    
    # RTF
    metrics['rtf'] = elapsed / duration
    metrics['inference_time_ms'] = elapsed * 1000
    
    return metrics


def generate_test_mixtures(
    target_dir: str,
    interferer_dir: str,
    output_dir: str,
    num_mixtures: int = 20,
    sir_range: tuple = (-5, 5),
    sample_rate: int = 16000,
    segment_length: int = 64000,
):
    """Generate test mixtures from target + interferer audio.
    
    Creates pairs of (mixture.wav, target_clean.wav) for evaluation.
    """
    import glob
    import random
    from training.dataset import load_audio, rms_normalize, mix_at_snr, random_segment
    
    os.makedirs(output_dir, exist_ok=True)
    
    target_files = sorted(glob.glob(os.path.join(target_dir, "**", "*.wav"), recursive=True))
    target_files += sorted(glob.glob(os.path.join(target_dir, "**", "*.flac"), recursive=True))
    
    int_files = sorted(glob.glob(os.path.join(interferer_dir, "**", "*.wav"), recursive=True))
    int_files += sorted(glob.glob(os.path.join(interferer_dir, "**", "*.flac"), recursive=True))
    
    if not target_files or not int_files:
        print(f"No audio files found in {target_dir} or {interferer_dir}")
        return
    
    for i in range(num_mixtures):
        target = load_audio(random.choice(target_files), sample_rate)
        interferer = load_audio(random.choice(int_files), sample_rate)
        
        target = random_segment(rms_normalize(target), segment_length)
        interferer = random_segment(rms_normalize(interferer), segment_length)
        
        sir = random.uniform(*sir_range)
        scaled_int = mix_at_snr(target, interferer, sir)
        mixture = target + scaled_int
        
        # Normalize
        max_val = torch.max(torch.abs(mixture))
        if max_val > 0.95:
            scale = 0.9 / max_val
            mixture *= scale
            target *= scale
        
        torchaudio.save(os.path.join(output_dir, f"mixture_{i:03d}.wav"),
                        mixture.unsqueeze(0), sample_rate)
        torchaudio.save(os.path.join(output_dir, f"target_{i:03d}.wav"),
                        target.unsqueeze(0), sample_rate)
    
    print(f"Generated {num_mixtures} test mixtures in {output_dir}")


def run_evaluation(
    checkpoint_path: str,
    test_dir: str,
    enrollment_path: str,
    model_size: str = "tiny",
    device: str = "cpu",
):
    """Run full evaluation on test set.
    
    Expects test_dir to contain pairs:
        mixture_000.wav, target_000.wav
        mixture_001.wav, target_001.wav
        ...
    """
    print(f"\n{'='*60}")
    print(f"  Your Denoizer — Evaluation")
    print(f"{'='*60}\n")
    
    # Load model
    model = ConvTasNetTSE.from_config(model_size)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    
    # Speaker encoder
    speaker_encoder = SpeakerEncoder(device=device)
    
    # Find test files
    test_path = Path(test_dir)
    mixture_files = sorted(test_path.glob("mixture_*.wav"))
    
    if not mixture_files:
        print(f"No mixture files found in {test_dir}")
        print("Expected format: mixture_000.wav, target_000.wav, ...")
        return
    
    all_metrics = []
    
    for mix_file in mixture_files:
        idx = mix_file.stem.replace("mixture_", "")
        target_file = test_path / f"target_{idx}.wav"
        
        if not target_file.exists():
            print(f"  Skipping {mix_file.name} — no matching target")
            continue
        
        print(f"  Evaluating {mix_file.name}...", end=" ")
        
        metrics = evaluate_sample(
            model=model,
            mixture_path=str(mix_file),
            target_path=str(target_file),
            enrollment_path=enrollment_path,
            speaker_encoder=speaker_encoder,
            device=device,
        )
        
        all_metrics.append(metrics)
        print(f"SI-SNRi: {metrics['si_snri']:+.1f} dB | "
              f"PESQ: {metrics['pesq']:.2f} | "
              f"STOI: {metrics['stoi']:.3f} | "
              f"SpkSim: {metrics['speaker_similarity']:.3f}")
    
    if not all_metrics:
        print("No samples evaluated!")
        return
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  Results ({len(all_metrics)} samples)")
    print(f"{'='*60}")
    
    for key in ['si_snr_input', 'si_snr_output', 'si_snri', 'pesq', 'stoi',
                'speaker_similarity', 'rtf', 'inference_time_ms']:
        values = [m[key] for m in all_metrics if not np.isnan(m.get(key, float('nan')))]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            print(f"  {key:25s}: {mean:.3f} ± {std:.3f}")
    
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate TSE model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test-dir', type=str, required=True)
    parser.add_argument('--enrollment', type=str, required=True)
    parser.add_argument('--model-size', type=str, default='tiny')
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    run_evaluation(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        enrollment_path=args.enrollment,
        model_size=args.model_size,
        device=args.device,
    )
