"""
Full TSE Model — Combines separator + speaker encoder into one pipeline.

This is the high-level API. It handles:
    - Enrollment: audio → speaker embedding
    - Extraction: mixture + embedding → isolated voice
    - End-to-end: mixture + enrollment audio → isolated voice

For training, use the separator directly (speaker encoder is frozen).
For inference, use this class for the complete pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Union, List, Optional

from models.separator import ConvTasNetTSE
from models.speaker_encoder import SpeakerEncoder


class TSEPipeline:
    """Complete Target Speaker Extraction pipeline.
    
    Combines the speaker encoder (ECAPA-TDNN) and separator (Conv-TasNet)
    into a single, easy-to-use interface.
    
    Example:
        # Initialize
        pipeline = TSEPipeline.from_checkpoint("checkpoints/best.pt")
        
        # Enroll a speaker
        pipeline.enroll(["enrollment1.wav", "enrollment2.wav"])
        
        # Extract their voice from a mixture
        extracted = pipeline.extract("noisy_meeting.wav", output_path="my_voice.wav")
        
        # Or do it in one call
        extracted = pipeline.extract_with_enrollment(
            mixture="noisy_meeting.wav",
            enrollment=["enrollment1.wav"],
            output="my_voice.wav"
        )
    """
    
    def __init__(
        self,
        separator: ConvTasNetTSE,
        speaker_encoder: Optional[SpeakerEncoder] = None,
        device: str = "cpu",
        sample_rate: int = 16000,
        chunk_size: int = 64000,
        overlap: float = 0.5,
    ):
        self.separator = separator.to(device).eval()
        self.speaker_encoder = speaker_encoder or SpeakerEncoder(device=device)
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Current enrolled speaker embedding
        self._speaker_embedding: Optional[torch.Tensor] = None
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, model_size: str = "tiny",
                         device: str = "cpu", **kwargs) -> "TSEPipeline":
        """Load pipeline from a training checkpoint.
        
        Args:
            checkpoint_path: Path to .pt checkpoint file
            model_size: Model config name ('nano', 'tiny', 'small', 'standard')
            device: Inference device ('cpu' for M4 Mac)
        """
        # Load model
        separator = ConvTasNetTSE.from_config(model_size)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        separator.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"[TSEPipeline] Loaded checkpoint: {checkpoint_path}")
        print(f"[TSEPipeline] Model: {model_size} ({separator._total_params:,} params)")
        
        return cls(separator=separator, device=device, **kwargs)
    
    @classmethod
    def from_onnx(cls, onnx_path: str, device: str = "cpu", **kwargs) -> "TSEPipeline":
        """Load pipeline with ONNX Runtime separator (faster CPU inference).
        
        Args:
            onnx_path: Path to .onnx model file
            device: Device for speaker encoder ('cpu')
        """
        import onnxruntime as ort
        
        # We wrap the ONNX session in a mock module
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Create a wrapper that mimics the separator interface
        separator = _ONNXSeparatorWrapper(session)
        
        print(f"[TSEPipeline] Loaded ONNX model: {onnx_path}")
        
        pipeline = cls(separator=separator, device=device, **kwargs)
        pipeline._use_onnx = True
        return pipeline
    
    def enroll(self, audio_paths: Union[str, List[str]], 
               save_path: Optional[str] = None) -> torch.Tensor:
        """Enroll a speaker from one or more audio clips.
        
        Computes the speaker embedding and stores it for subsequent extractions.
        
        Args:
            audio_paths: Path(s) to enrollment audio files
            save_path: Optional path to save embedding as .npy
        
        Returns:
            embedding: (192,) speaker embedding tensor
        """
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
        
        if len(audio_paths) == 1:
            embedding = self.speaker_encoder.encode_file(audio_paths[0])
        else:
            embedding = self.speaker_encoder.encode_enrollment(audio_paths)
        
        self._speaker_embedding = embedding
        
        if save_path:
            np.save(save_path, embedding.cpu().numpy())
            print(f"[TSEPipeline] Saved embedding to {save_path}")
        
        print(f"[TSEPipeline] Speaker enrolled (embedding dim: {embedding.shape[0]})")
        return embedding
    
    def load_embedding(self, path: str):
        """Load a previously saved speaker embedding.
        
        Args:
            path: Path to .npy embedding file
        """
        self._speaker_embedding = SpeakerEncoder.load_embedding(path)
        print(f"[TSEPipeline] Loaded embedding from {path}")
    
    @torch.no_grad()
    def extract(self, mixture_path: str, output_path: Optional[str] = None,
                speaker_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract the enrolled speaker's voice from a mixture.
        
        Args:
            mixture_path: Path to mixture audio file
            output_path: Optional path to save extracted audio
            speaker_embedding: Override the enrolled embedding
        
        Returns:
            extracted: (T,) waveform tensor at 16kHz
        """
        import torchaudio
        
        emb = speaker_embedding if speaker_embedding is not None else self._speaker_embedding
        if emb is None:
            raise ValueError("No speaker enrolled! Call .enroll() first or provide speaker_embedding.")
        
        # Load mixture
        waveform, sr = torchaudio.load(mixture_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        mixture = waveform.squeeze(0)
        
        # Process with overlap-add
        from inference.extract import overlap_add_process
        
        extracted = overlap_add_process(
            model=self.separator,
            mixture=mixture,
            speaker_emb=emb.cpu(),
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            device=self.device,
        )
        
        if output_path:
            torchaudio.save(output_path, extracted.unsqueeze(0), self.sample_rate)
            print(f"[TSEPipeline] Saved extracted audio to {output_path}")
        
        return extracted
    
    def extract_with_enrollment(self, mixture: str, enrollment: Union[str, List[str]],
                                 output: Optional[str] = None) -> torch.Tensor:
        """One-call extraction: enroll + extract in a single step.
        
        Args:
            mixture: Path to mixture audio
            enrollment: Path(s) to enrollment clips
            output: Optional output path
        
        Returns:
            extracted: (T,) waveform tensor
        """
        self.enroll(enrollment)
        return self.extract(mixture, output_path=output)
    
    def check_speaker_similarity(self, audio_path: str) -> float:
        """Check how similar an audio file sounds to the enrolled speaker.
        
        Useful for verifying extraction quality.
        
        Returns:
            similarity: float in [0, 1], higher = more similar
        """
        if self._speaker_embedding is None:
            raise ValueError("No speaker enrolled!")
        
        audio_emb = self.speaker_encoder.encode_file(audio_path)
        similarity = SpeakerEncoder.compute_similarity(
            self._speaker_embedding, audio_emb
        )
        return max(0, similarity)  # Clamp to [0, 1]


class _ONNXSeparatorWrapper(nn.Module):
    """Wraps an ONNX Runtime session to mimic the ConvTasNetTSE interface."""
    
    def __init__(self, session):
        super().__init__()
        self._session = session
        self._total_params = 0  # Unknown for ONNX
    
    def forward(self, mixture: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        import numpy as np
        
        if mixture.dim() == 2:
            mixture = mixture.unsqueeze(1)
        
        result = self._session.run(None, {
            'mixture': mixture.cpu().numpy(),
            'speaker_embedding': speaker_emb.cpu().numpy(),
        })
        
        return torch.tensor(result[0])
    
    def eval(self):
        return self
    
    def to(self, device):
        return self
