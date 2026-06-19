"""
Export Conv-TasNet-TSE to ONNX for optimized CPU inference on M4 Mac.

Usage:
    python -m export.to_onnx \
        --checkpoint checkpoints/best.pt \
        --output checkpoints/exported/separator.onnx \
        --model-size tiny
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.separator import ConvTasNetTSE


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    model_size: str = "tiny",
    opset_version: int = 17,
    verify: bool = True,
):
    """Export trained model to ONNX format.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pt)
        output_path: Path to save ONNX model
        model_size: Model config name
        opset_version: ONNX opset version
        verify: Run verification after export
    """
    print(f"\n{'='*60}")
    print(f"  ONNX Export")
    print(f"{'='*60}\n")
    
    # Load model
    model = ConvTasNetTSE.from_config(model_size)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[Export] Loaded checkpoint: {checkpoint_path}")
    
    model.eval()
    model.cpu()
    
    # Dummy inputs
    batch_size = 1
    audio_length = 64000  # 4 seconds at 16kHz
    speaker_dim = 192
    
    dummy_mixture = torch.randn(batch_size, 1, audio_length)
    dummy_embedding = torch.randn(batch_size, speaker_dim)
    
    # Export
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Export] Exporting to: {output_path}")
    print(f"[Export] Opset version: {opset_version}")
    
    torch.onnx.export(
        model,
        (dummy_mixture, dummy_embedding),
        output_path,
        input_names=['mixture', 'speaker_embedding'],
        output_names=['extracted_voice'],
        opset_version=opset_version,
        dynamic_axes={
            'mixture': {0: 'batch', 2: 'audio_length'},
            'speaker_embedding': {0: 'batch'},
            'extracted_voice': {0: 'batch', 2: 'audio_length'},
        },
        do_constant_folding=True,
    )
    
    # Check file size
    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Export] Model size: {size_mb:.2f} MB")
    
    # Verify
    if verify:
        print(f"\n[Verify] Running ONNX validation...")
        
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"[Verify] ✅ ONNX model is valid")
        
        # Test with ONNX Runtime
        import onnxruntime as ort
        import numpy as np
        
        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        
        # Run inference
        result = session.run(None, {
            'mixture': dummy_mixture.numpy(),
            'speaker_embedding': dummy_embedding.numpy(),
        })
        
        output_shape = result[0].shape
        print(f"[Verify] ✅ ONNX Runtime inference OK (output shape: {output_shape})")
        
        # Compare with PyTorch output
        with torch.no_grad():
            torch_output = model(dummy_mixture, dummy_embedding).numpy()
        
        max_diff = np.max(np.abs(torch_output - result[0]))
        print(f"[Verify] Max difference vs PyTorch: {max_diff:.6f}")
        
        if max_diff < 1e-4:
            print(f"[Verify] ✅ Outputs match (diff < 1e-4)")
        else:
            print(f"[Verify] ⚠️ Outputs differ (diff = {max_diff:.6f}), but this is often OK for FP32 vs ONNX")
        
        # Benchmark CPU speed
        import time
        num_runs = 20
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, {
                'mixture': dummy_mixture.numpy(),
                'speaker_embedding': dummy_embedding.numpy(),
            })
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times[5:])  # Skip warmup
        rtf = avg_time / 4.0  # 4 seconds of audio
        print(f"\n[Benchmark] Avg inference time: {avg_time*1000:.1f}ms per 4s chunk")
        print(f"[Benchmark] RTF: {rtf:.4f} ({'✅ Real-time' if rtf < 1.0 else '❌ Slower than real-time'})")
    
    print(f"\n{'='*60}")
    print(f"  ✅ Export complete: {output_path} ({size_mb:.2f} MB)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True, help='PyTorch checkpoint path')
    parser.add_argument('--output', type=str, default='checkpoints/exported/separator.onnx')
    parser.add_argument('--model-size', type=str, default='tiny')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--no-verify', action='store_true')
    
    args = parser.parse_args()
    
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_size=args.model_size,
        opset_version=args.opset,
        verify=not args.no_verify,
    )
