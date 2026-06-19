"""
Inference script — Extract target speaker from mixture audio.

Runs on M4 CPU. Supports both PyTorch and ONNX Runtime backends.
Uses overlap-add for processing long audio files in chunks.

Usage:
    python -m inference.extract \
        --noisy input_noisy.wav \
        --output clean_output.wav \
        --checkpoint checkpoints/best.pt
    
    # With ONNX (faster on CPU):
    python -m inference.extract \
        --noisy input_noisy.wav \
        --output clean_output.wav \
        --onnx checkpoints/exported/separator.onnx
"""

import argparse
import time
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.separator import ConvTasNet


def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load audio file and convert to mono 16kHz."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    return waveform.squeeze(0)  # (T,)


def overlap_add_process(model, noisy: torch.Tensor,
                         chunk_size: int = 64000, overlap: float = 0.5,
                         device: str = "cpu", use_onnx: bool = False,
                         onnx_session=None) -> torch.Tensor:
    """Process long audio with overlap-add chunking.
    
    Splits audio into overlapping chunks, processes each through the model,
    and reconstructs the full output using overlap-add with Hann windowing.
    
    Args:
        model: ConvTasNet model (or None if using ONNX)
        noisy: (T,) input noisy waveform
        chunk_size: Size of each processing chunk in samples
        overlap: Overlap ratio between chunks (0.0 - 0.9)
        device: PyTorch device
        use_onnx: Use ONNX Runtime instead of PyTorch
        onnx_session: ONNX InferenceSession (required if use_onnx=True)
    
    Returns:
        output: (T,) denoised waveform
    """
    T = noisy.shape[0]
    hop_size = int(chunk_size * (1 - overlap))
    
    # Pad to fit complete chunks
    num_chunks = max(1, (T - chunk_size) // hop_size + 1)
    padded_length = (num_chunks - 1) * hop_size + chunk_size
    
    if padded_length > T:
        noisy = F.pad(noisy, (0, padded_length - T))
    
    # Hann window for smooth overlap-add
    window = torch.hann_window(chunk_size)
    output = torch.zeros(padded_length)
    weight = torch.zeros(padded_length)
    
    # Process each chunk
    for i in range(num_chunks):
        start = i * hop_size
        end = start + chunk_size
        chunk = noisy[start:end]
        
        if use_onnx and onnx_session:
            # ONNX inference
            chunk_np = chunk.unsqueeze(0).unsqueeze(0).numpy()  # (1, 1, chunk_size)
            
            result = onnx_session.run(None, {
                'noisy': chunk_np,
            })
            extracted = torch.tensor(result[0]).squeeze()
        else:
            # PyTorch inference
            with torch.no_grad():
                chunk_input = chunk.unsqueeze(0).to(device)       # (1, T)
                
                extracted = model(chunk_input)
                extracted = extracted.squeeze().cpu()
        
        # Ensure correct length
        if extracted.shape[0] < chunk_size:
            extracted = F.pad(extracted, (0, chunk_size - extracted.shape[0]))
        extracted = extracted[:chunk_size]
        
        # Apply window and add
        output[start:end] += extracted * window
        weight[start:end] += window
    
    # Normalize by window sum
    output = output / (weight + 1e-8)
    
    # Trim to original length
    return output[:T]


def extract(
    noisy_path: str,
    output_path: str,
    checkpoint_path: str = None,
    onnx_path: str = None,
    model_size: str = "tiny",
    chunk_size: int = 64000,
    overlap: float = 0.5,
    device: str = "cpu",
):
    """Denoise audio using ConvTasNet.
    
    Args:
        noisy_path: Path to noisy audio file
        output_path: Path to save clean audio
        checkpoint_path: Path to PyTorch checkpoint (.pt)
        onnx_path: Path to ONNX model (overrides checkpoint if provided)
        model_size: Model config name
        chunk_size: Processing chunk size in samples
        overlap: Overlap ratio for overlap-add
        device: PyTorch device
    """
    print(f"\n{'='*60}")
    print(f"  Your Denoizer — Speech Enhancement")
    print(f"{'='*60}")
    
    # 1. Load and preprocess mixture
    print(f"\n[1/3] Loading noisy audio: {noisy_path}")
    noisy = load_audio(noisy_path)
    duration = len(noisy) / 16000
    print(f"  Duration: {duration:.1f}s ({len(noisy):,} samples)")
    
    # 2. Load model
    use_onnx = onnx_path is not None
    onnx_session = None
    model = None
    
    if use_onnx:
        print(f"\n[3/4] Loading ONNX model: {onnx_path}")
        import onnxruntime as ort
        onnx_session = ort.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider']
        )
        print(f"  ONNX Runtime loaded (CPU)")
    else:
        print(f"\n[2/3] Loading PyTorch model: {checkpoint_path}")
        model = ConvTasNet.from_config(model_size, speaker_dim=0)
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to(device)
        model.eval()
        print(f"  Model loaded ({model._total_params:,} params)")
    
    # 3. Extract
    print(f"\n[3/3] Denoising audio...")
    start_time = time.perf_counter()
    
    output = overlap_add_process(
        model=model,
        noisy=noisy,
        chunk_size=chunk_size,
        overlap=overlap,
        device=device,
        use_onnx=use_onnx,
        onnx_session=onnx_session,
    )
    
    elapsed = time.perf_counter() - start_time
    rtf = elapsed / duration
    
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  RTF: {rtf:.4f} ({'✅ Real-time' if rtf < 1.0 else '❌ Slower than real-time'})")
    
    # 5. Save output
    torchaudio.save(output_path, output.unsqueeze(0), 16000)
    print(f"\n  ✅ Saved: {output_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Denoise audio')
    parser.add_argument('--noisy', type=str, required=True, help='Path to noisy audio')
    parser.add_argument('--output', type=str, default='clean.wav', help='Output path')
    parser.add_argument('--checkpoint', type=str, default=None, help='PyTorch checkpoint path')
    parser.add_argument('--onnx', type=str, default=None, help='ONNX model path')
    parser.add_argument('--model-size', type=str, default='tiny', help='Model config name')
    parser.add_argument('--chunk-size', type=int, default=64000, help='Chunk size in samples')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap ratio')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    
    args = parser.parse_args()
    
    extract(
        noisy_path=args.noisy,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        onnx_path=args.onnx,
        model_size=args.model_size,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        device=args.device,
    )
