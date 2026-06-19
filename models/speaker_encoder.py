"""
Speaker Encoder — ECAPA-TDNN wrapper for extracting speaker embeddings.

Uses SpeechBrain's pretrained ECAPA-TDNN model trained on VoxCeleb (7000+ speakers).
This model is FROZEN during TSE training — it only provides the speaker identity signal.

The embedding is computed once at enrollment time and saved. During inference,
the saved embedding is loaded directly — no speaker encoder needed in the inference path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np


class SpeakerEncoder(nn.Module):
    """ECAPA-TDNN speaker encoder wrapper.
    
    Extracts 192-dimensional speaker embeddings from audio clips.
    Uses SpeechBrain's pretrained model — always frozen, never fine-tuned.
    
    Usage:
        encoder = SpeakerEncoder()
        embedding = encoder.encode_file("enrollment.wav")  # (192,)
        embeddings = encoder.encode_batch(waveforms)        # (B, 192)
    """
    
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self._model = None
        self._loaded = False
    
    def _load_model(self):
        """Lazy-load the SpeechBrain ECAPA-TDNN model."""
        if self._loaded:
            return
        
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            
            self._model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device},
            )
            self._loaded = True
            print(f"[SpeakerEncoder] ECAPA-TDNN loaded on {self.device}")
            
        except ImportError:
            raise ImportError(
                "SpeechBrain is required for the speaker encoder. "
                "Install it with: pip install speechbrain"
            )
    
    @torch.no_grad()
    def encode_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Encode a batch of waveforms into speaker embeddings.
        
        Args:
            waveforms: (B, T) — batch of waveforms at 16kHz
        
        Returns:
            embeddings: (B, 192) — L2-normalized speaker embeddings
        """
        self._load_model()
        
        waveforms = waveforms.to(self.device)
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)
        
        embeddings = self._model.encode_batch(waveforms)
        embeddings = embeddings.squeeze(1)  # (B, 1, 192) → (B, 192)
        
        # L2 normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    @torch.no_grad()
    def encode_file(self, audio_path: str) -> torch.Tensor:
        """Encode a single audio file into a speaker embedding.
        
        Args:
            audio_path: Path to WAV file (any sample rate, will be resampled)
        
        Returns:
            embedding: (192,) — L2-normalized speaker embedding
        """
        self._load_model()
        
        import torchaudio
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        embedding = self.encode_batch(waveform)
        return embedding.squeeze(0)  # (192,)
    
    @torch.no_grad()
    def encode_enrollment(self, audio_paths: list, output_path: str = None) -> torch.Tensor:
        """Encode multiple enrollment clips and average the embeddings.
        
        This produces a more robust voiceprint by averaging over multiple
        utterances from the same speaker.
        
        Args:
            audio_paths: List of paths to enrollment WAV files
            output_path: Optional path to save the embedding as .npy
        
        Returns:
            embedding: (192,) — averaged, L2-normalized speaker embedding
        """
        embeddings = []
        for path in audio_paths:
            emb = self.encode_file(path)
            embeddings.append(emb)
        
        # Average and re-normalize
        avg_embedding = torch.stack(embeddings).mean(dim=0)
        avg_embedding = F.normalize(avg_embedding, p=2, dim=-1)
        
        if output_path:
            np.save(output_path, avg_embedding.cpu().numpy())
            print(f"[SpeakerEncoder] Saved enrollment embedding to {output_path}")
        
        return avg_embedding
    
    @staticmethod
    def load_embedding(path: str) -> torch.Tensor:
        """Load a saved speaker embedding from .npy file.
        
        Args:
            path: Path to .npy file
        
        Returns:
            embedding: (192,) tensor
        """
        emb = np.load(path)
        return torch.from_numpy(emb).float()
    
    @staticmethod
    def compute_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings.
        
        Useful for quality filtering: compare enrollment vs extracted voice
        to verify speaker identity is preserved.
        
        Args:
            emb1, emb2: (192,) speaker embeddings
        
        Returns:
            similarity: float in [-1, 1], higher = more similar
        """
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
