"""
Enrollment CLI — Register a speaker's voice for extraction.

Usage:
    # Enroll from audio files
    python -m inference.enroll \
        --audio clip1.wav clip2.wav clip3.wav \
        --output my_voiceprint.npy

    # Enroll and verify (plays back similarity score)
    python -m inference.enroll \
        --audio clip1.wav clip2.wav \
        --output my_voiceprint.npy \
        --verify clip3.wav
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.speaker_encoder import SpeakerEncoder


def enroll(
    audio_paths: list,
    output_path: str,
    verify_path: str = None,
    device: str = "cpu",
):
    """Enroll a speaker and save their voiceprint.
    
    Args:
        audio_paths: Paths to enrollment audio clips (3-10 recommended)
        output_path: Path to save the embedding (.npy)
        verify_path: Optional audio to verify against the enrollment
        device: Device for computation
    """
    print(f"\n{'='*60}")
    print(f"  Your Denoizer — Speaker Enrollment")
    print(f"{'='*60}\n")
    
    encoder = SpeakerEncoder(device=device)
    
    print(f"[Enrollment] Processing {len(audio_paths)} clip(s)...")
    
    # Compute individual embeddings
    embeddings = []
    for i, path in enumerate(audio_paths):
        emb = encoder.encode_file(path)
        embeddings.append(emb)
        print(f"  [{i+1}/{len(audio_paths)}] {Path(path).name} ✅")
    
    # Average embeddings
    avg_embedding = torch.stack(embeddings).mean(dim=0)
    avg_embedding = torch.nn.functional.normalize(avg_embedding, p=2, dim=0)
    
    # Show pairwise similarities (consistency check)
    if len(embeddings) > 1:
        print(f"\n[Consistency] Pairwise similarities between enrollment clips:")
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = SpeakerEncoder.compute_similarity(embeddings[i], embeddings[j])
                status = "✅" if sim > 0.7 else "⚠️ low"
                print(f"  Clip {i+1} ↔ Clip {j+1}: {sim:.3f} {status}")
    
    # Save
    np.save(output_path, avg_embedding.cpu().numpy())
    print(f"\n[Saved] Voiceprint → {output_path}")
    print(f"  Embedding dimension: {avg_embedding.shape[0]}")
    
    # Verify against held-out clip
    if verify_path:
        print(f"\n[Verify] Checking against: {verify_path}")
        verify_emb = encoder.encode_file(verify_path)
        similarity = SpeakerEncoder.compute_similarity(avg_embedding, verify_emb)
        
        if similarity > 0.8:
            verdict = "✅ Strong match — same speaker"
        elif similarity > 0.6:
            verdict = "⚠️ Moderate match — probably same speaker"
        else:
            verdict = "❌ Weak match — possibly different speaker"
        
        print(f"  Similarity: {similarity:.3f}")
        print(f"  Verdict: {verdict}")
    
    print(f"\n{'='*60}")
    print(f"  Enrollment complete! Use this voiceprint for extraction:")
    print(f"  python -m inference.extract --mixture input.wav \\")
    print(f"    --enrollment {output_path} --output extracted.wav")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enroll speaker and create voiceprint')
    parser.add_argument('--audio', type=str, nargs='+', required=True,
                        help='Path(s) to enrollment audio clips')
    parser.add_argument('--output', type=str, default='voiceprint.npy',
                        help='Output path for voiceprint (.npy)')
    parser.add_argument('--verify', type=str, default=None,
                        help='Optional audio clip to verify enrollment quality')
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    enroll(args.audio, args.output, args.verify, args.device)
