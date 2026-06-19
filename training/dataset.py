"""
On-the-fly mixture generation dataset for Target Speaker Extraction.

Creates training triplets dynamically:
    (mixture, clean_target, enrollment_embedding)

Each mixture is generated fresh every time — the model never sees the
exact same mixture twice. This provides effectively infinite data augmentation.

Supports:
    - HuggingFace datasets (streaming + download) with configurable audio columns
    - Local datasets (LibriSpeech, VoxCeleb, flat directories)
    - Noise from HuggingFace (MUSAN, DEMAND, UrbanSound, FSD50K, etc.)
    - Local noise (MUSAN, DEMAND, custom WAV directories)
    - Room impulse responses (RIR: reverb simulation)
    - Speed/pitch perturbation
    - Dynamic SIR/SNR mixing
    - Multilingual: Arabic, English, Mandarin, and any HF speech dataset
"""

import os
import random
import glob
import math
import io
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np

# ============================================================================
# Audio Utilities
# ============================================================================

def load_audio(path: str, target_sr: int = 16000, mono: bool = True) -> torch.Tensor:
    """Load audio file and resample if needed.
    
    Returns:
        waveform: (T,) tensor at target sample rate
    """
    waveform, sr = torchaudio.load(path)
    
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    return waveform.squeeze(0)  # (T,)


def decode_hf_audio(sample: dict, audio_column: str, target_sr: int = 16000) -> torch.Tensor:
    """Decode audio from a HuggingFace dataset sample.
    
    HuggingFace audio columns can be in various formats:
        - {'array': np.ndarray, 'sampling_rate': int}  (decoded)
        - {'path': str, 'bytes': bytes}                 (raw)
        - str (just a path)
    
    Args:
        sample: HF dataset row
        audio_column: Name of the audio column
        target_sr: Target sample rate
    
    Returns:
        waveform: (T,) tensor at target_sr
    """
    audio_data = sample[audio_column]
    
    # Case 1: Already decoded by HF (most common with .cast_column('audio', Audio(sr=16000)))
    if isinstance(audio_data, dict) and 'array' in audio_data:
        waveform = torch.tensor(audio_data['array'], dtype=torch.float32)
        sr = audio_data.get('sampling_rate', target_sr)
        
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
        
        return waveform
    
    # Case 2: Raw bytes
    if isinstance(audio_data, dict) and 'bytes' in audio_data and audio_data['bytes']:
        audio_bytes = audio_data['bytes']
        buffer = io.BytesIO(audio_bytes)
        waveform, sr = torchaudio.load(buffer)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        return waveform.squeeze(0)
    
    # Case 3: File path
    if isinstance(audio_data, dict) and 'path' in audio_data:
        return load_audio(audio_data['path'], target_sr)
    
    if isinstance(audio_data, str):
        return load_audio(audio_data, target_sr)
    
    raise ValueError(f"Cannot decode audio from column '{audio_column}': {type(audio_data)}")


def rms_normalize(waveform: torch.Tensor, target_rms: float = 0.1) -> torch.Tensor:
    """Normalize waveform to target RMS level."""
    rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-8)
    if rms < 1e-10:
        return waveform
    return waveform * (target_rms / rms)


def mix_at_snr(signal: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Scale noise to achieve target SNR relative to signal.
    
    Returns:
        scaled_noise: (T,) noise scaled to target SNR
    """
    signal_power = torch.mean(signal ** 2) + 1e-8
    noise_power = torch.mean(noise ** 2) + 1e-8
    snr_linear = 10 ** (snr_db / 10)
    scale = torch.sqrt(signal_power / (noise_power * snr_linear))
    return noise * scale


def convolve_rir(waveform: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    """Apply room impulse response (reverb) to waveform."""
    rir = rir / (torch.max(torch.abs(rir)) + 1e-8)
    
    waveform_3d = waveform.unsqueeze(0).unsqueeze(0)
    rir_3d = rir.flip(0).unsqueeze(0).unsqueeze(0)
    
    convolved = F.conv1d(waveform_3d, rir_3d, padding=rir.shape[0] - 1)
    result = convolved.squeeze()[:waveform.shape[0]]
    
    orig_rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-8)
    new_rms = torch.sqrt(torch.mean(result ** 2) + 1e-8)
    result = result * (orig_rms / (new_rms + 1e-8))
    
    return result


def random_segment(waveform: torch.Tensor, segment_length: int) -> torch.Tensor:
    """Extract a random segment of fixed length from waveform.
    Loops if too short, crops randomly if too long."""
    T = waveform.shape[0]
    if T >= segment_length:
        offset = random.randint(0, T - segment_length)
        return waveform[offset:offset + segment_length]
    else:
        repeats = math.ceil(segment_length / T)
        return waveform.repeat(repeats)[:segment_length]


# ============================================================================
# HuggingFace Dataset Source
# ============================================================================

class HFSpeechSource:
    """A speech data source backed by a HuggingFace dataset.
    
    Supports streaming and downloaded modes. Handles any audio column name.
    Groups samples by speaker_id for enrollment pairing.
    
    Example usage:
        # Arabic Common Voice
        source = HFSpeechSource(
            dataset_name="mozilla-foundation/common_voice_17_0",
            subset="ar",
            split="train",
            audio_column="audio",
            speaker_column="client_id",
            streaming=True,
        )
        
        # LibriSpeech from HF
        source = HFSpeechSource(
            dataset_name="openslr/librispeech_asr",
            subset="train.clean.100",
            split="train",
            audio_column="audio",
            speaker_column="speaker_id",
        )
    """
    
    def __init__(
        self,
        dataset_name: str,
        subset: str = None,
        split: str = "train",
        audio_column: str = "audio",
        speaker_column: str = None,
        text_column: str = None,
        streaming: bool = False,
        max_samples: int = None,
        trust_remote_code: bool = True,
        sample_rate: int = 16000,
    ):
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.audio_column = audio_column
        self.speaker_column = speaker_column
        self.text_column = text_column
        self.streaming = streaming
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        
        self._dataset = None
        self._speaker_indices: Dict[str, List[int]] = {}
        self._all_speakers: List[str] = []
        self._loaded = False
        self._samples_buffer: List[dict] = []  # For streaming mode
        self._buffer_size = 5000  # How many samples to buffer in streaming mode
        
        self.trust_remote_code = trust_remote_code
    
    def _load(self):
        """Load the HuggingFace dataset."""
        if self._loaded:
            return
        
        from datasets import load_dataset, Audio
        
        print(f"[HFSpeechSource] Loading {self.dataset_name}"
              f"{f'/{self.subset}' if self.subset else ''} "
              f"(split={self.split}, streaming={self.streaming})...")
        
        kwargs = {
            'split': self.split,
            'streaming': self.streaming,
            'trust_remote_code': self.trust_remote_code,
        }
        if self.subset:
            kwargs['name'] = self.subset
        
        self._dataset = load_dataset(self.dataset_name, **kwargs)
        
        # Cast audio column to desired sample rate
        if not self.streaming:
            try:
                self._dataset = self._dataset.cast_column(
                    self.audio_column, Audio(sampling_rate=self.sample_rate)
                )
            except Exception:
                pass  # Column might not exist yet or format differs
            
            # Limit samples
            if self.max_samples and len(self._dataset) > self.max_samples:
                self._dataset = self._dataset.select(range(self.max_samples))
            
            # Build speaker index
            if self.speaker_column:
                self._build_speaker_index()
            
            print(f"[HFSpeechSource] Loaded {len(self._dataset)} samples"
                  f" ({len(self._all_speakers)} speakers)")
        else:
            # For streaming, buffer samples lazily
            print(f"[HFSpeechSource] Streaming mode — will buffer on first access")
        
        self._loaded = True
    
    def _build_speaker_index(self):
        """Build speaker → sample indices mapping."""
        self._speaker_indices = {}
        for idx, sample in enumerate(self._dataset):
            speaker = str(sample.get(self.speaker_column, f"spk_{idx}"))
            if speaker not in self._speaker_indices:
                self._speaker_indices[speaker] = []
            self._speaker_indices[speaker].append(idx)
        
        self._all_speakers = list(self._speaker_indices.keys())
    
    def _fill_streaming_buffer(self):
        """Fill the sample buffer from streaming dataset."""
        if self._samples_buffer:
            return
        
        from datasets import Audio
        
        print(f"[HFSpeechSource] Filling streaming buffer ({self._buffer_size} samples)...")
        
        count = 0
        for sample in self._dataset:
            self._samples_buffer.append(sample)
            count += 1
            if count >= self._buffer_size:
                break
        
        # Build speaker index from buffer
        if self.speaker_column:
            self._speaker_indices = {}
            for idx, sample in enumerate(self._samples_buffer):
                speaker = str(sample.get(self.speaker_column, f"stream_spk_{idx}"))
                if speaker not in self._speaker_indices:
                    self._speaker_indices[speaker] = []
                self._speaker_indices[speaker].append(idx)
            self._all_speakers = list(self._speaker_indices.keys())
        
        print(f"[HFSpeechSource] Buffered {len(self._samples_buffer)} samples "
              f"({len(self._all_speakers)} speakers)")
    
    def get_sample(self, idx: int = None) -> dict:
        """Get a sample from the dataset."""
        self._load()
        
        if self.streaming:
            self._fill_streaming_buffer()
            if idx is None:
                idx = random.randint(0, len(self._samples_buffer) - 1)
            return self._samples_buffer[idx % len(self._samples_buffer)]
        else:
            if idx is None:
                idx = random.randint(0, len(self._dataset) - 1)
            return self._dataset[idx]
    
    def get_audio(self, idx: int = None) -> torch.Tensor:
        """Get decoded audio from a sample."""
        sample = self.get_sample(idx)
        return decode_hf_audio(sample, self.audio_column, self.sample_rate)
    
    def get_random_speaker(self, exclude: str = None) -> str:
        """Get a random speaker ID."""
        self._load()
        if self.streaming:
            self._fill_streaming_buffer()
        
        candidates = [s for s in self._all_speakers if s != exclude] if exclude else self._all_speakers
        if not candidates:
            return self._all_speakers[0] if self._all_speakers else "unknown"
        return random.choice(candidates)
    
    def get_random_file_for_speaker(self, speaker_id: str, exclude_idx: int = None) -> Tuple[int, dict]:
        """Get a random sample from a specific speaker.
        
        Returns:
            (index, sample) tuple
        """
        indices = self._speaker_indices.get(speaker_id, [])
        if exclude_idx and len(indices) > 1:
            indices = [i for i in indices if i != exclude_idx]
        
        if not indices:
            # Fallback: return any random sample
            idx = random.randint(0, (len(self._samples_buffer) if self.streaming 
                                     else len(self._dataset)) - 1)
        else:
            idx = random.choice(indices)
        
        return idx, self.get_sample(idx)
    
    def get_speakers_with_multiple(self, min_files: int = 2) -> List[str]:
        """Get speakers with at least min_files recordings."""
        self._load()
        if self.streaming:
            self._fill_streaming_buffer()
        return [s for s, indices in self._speaker_indices.items() if len(indices) >= min_files]
    
    @property
    def num_samples(self) -> int:
        self._load()
        if self.streaming:
            self._fill_streaming_buffer()
            return len(self._samples_buffer)
        return len(self._dataset)
    
    @property
    def num_speakers(self) -> int:
        self._load()
        if self.streaming:
            self._fill_streaming_buffer()
        return len(self._all_speakers)


class HFNoiseSource:
    """A noise data source backed by a HuggingFace dataset.
    
    Example:
        # MUSAN from HuggingFace
        noise = HFNoiseSource(
            dataset_name="flozi00/MUSAN-Noise",
            audio_column="audio",
            streaming=True,
        )
        
        # Environmental sounds
        noise = HFNoiseSource(
            dataset_name="danavery/urbansound8K",
            audio_column="audio",
            label_column="class",
            streaming=True,
        )
    """
    
    def __init__(
        self,
        dataset_name: str,
        subset: str = None,
        split: str = "train",
        audio_column: str = "audio",
        label_column: str = None,
        streaming: bool = False,
        max_samples: int = None,
        trust_remote_code: bool = True,
        sample_rate: int = 16000,
        category: str = "noise",
    ):
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.audio_column = audio_column
        self.label_column = label_column
        self.streaming = streaming
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        self.category = category
        self.trust_remote_code = trust_remote_code
        
        self._dataset = None
        self._loaded = False
        self._buffer: List[dict] = []
        self._buffer_size = 2000
    
    def _load(self):
        if self._loaded:
            return
        
        from datasets import load_dataset, Audio
        
        print(f"[HFNoiseSource] Loading {self.dataset_name} "
              f"(category={self.category}, streaming={self.streaming})...")
        
        kwargs = {
            'split': self.split,
            'streaming': self.streaming,
            'trust_remote_code': self.trust_remote_code,
        }
        if self.subset:
            kwargs['name'] = self.subset
        
        self._dataset = load_dataset(self.dataset_name, **kwargs)
        
        if not self.streaming:
            try:
                self._dataset = self._dataset.cast_column(
                    self.audio_column, Audio(sampling_rate=self.sample_rate)
                )
            except Exception:
                pass
            
            if self.max_samples and len(self._dataset) > self.max_samples:
                self._dataset = self._dataset.select(range(self.max_samples))
            
            print(f"[HFNoiseSource] Loaded {len(self._dataset)} noise samples")
        else:
            print(f"[HFNoiseSource] Streaming mode — will buffer on first access")
        
        self._loaded = True
    
    def _fill_buffer(self):
        if self._buffer:
            return
        
        self._load()
        print(f"[HFNoiseSource] Filling noise buffer ({self._buffer_size} samples)...")
        
        count = 0
        for sample in self._dataset:
            self._buffer.append(sample)
            count += 1
            if count >= self._buffer_size:
                break
        
        print(f"[HFNoiseSource] Buffered {len(self._buffer)} noise samples")
    
    def get_random_noise(self) -> torch.Tensor:
        """Get a random noise waveform."""
        self._load()
        
        if self.streaming:
            self._fill_buffer()
            sample = random.choice(self._buffer)
        else:
            idx = random.randint(0, len(self._dataset) - 1)
            sample = self._dataset[idx]
        
        return decode_hf_audio(sample, self.audio_column, self.sample_rate)


# ============================================================================
# Local File Indexers (for downloaded data)
# ============================================================================

class LocalSpeechIndex:
    """Indexes local audio files by speaker for efficient sampling."""
    
    def __init__(self):
        self.speaker_files: Dict[str, List[str]] = {}
        self.all_files: List[str] = []
        self.all_speakers: List[str] = []
    
    def add_librispeech(self, root_dir: str):
        """Index LibriSpeech: root/speaker_id/chapter_id/*.flac"""
        root = Path(root_dir)
        if not root.exists():
            print(f"[LocalSpeechIndex] Warning: {root_dir} not found, skipping")
            return 0
        
        count = 0
        for speaker_dir in sorted(root.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker_id = f"ls_{speaker_dir.name}"
            files = sorted(glob.glob(str(speaker_dir / "**" / "*.flac"), recursive=True))
            files += sorted(glob.glob(str(speaker_dir / "**" / "*.wav"), recursive=True))
            if files:
                self.speaker_files[speaker_id] = files
                count += len(files)
        
        self._rebuild()
        print(f"[LocalSpeechIndex] LibriSpeech: {len(self.speaker_files)} speakers, {count} files")
        return count
    
    def add_voxceleb(self, root_dir: str):
        """Index VoxCeleb: root/id*/utterance_dir/*.wav"""
        root = Path(root_dir)
        if not root.exists():
            print(f"[LocalSpeechIndex] Warning: {root_dir} not found, skipping")
            return 0
        
        count = 0
        for speaker_dir in sorted(root.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker_id = f"vc_{speaker_dir.name}"
            files = sorted(glob.glob(str(speaker_dir / "**" / "*.wav"), recursive=True))
            files += sorted(glob.glob(str(speaker_dir / "**" / "*.m4a"), recursive=True))
            if files:
                self.speaker_files[speaker_id] = files
                count += len(files)
        
        self._rebuild()
        print(f"[LocalSpeechIndex] VoxCeleb: {len(self.speaker_files)} speakers, {count} files")
        return count
    
    def add_directory(self, root_dir: str, prefix: str = "custom"):
        """Index a flat directory. Each file treated as separate speaker."""
        root = Path(root_dir)
        if not root.exists():
            print(f"[LocalSpeechIndex] Warning: {root_dir} not found, skipping")
            return 0
        
        files = []
        for ext in ['*.wav', '*.flac', '*.mp3', '*.ogg']:
            files.extend(sorted(glob.glob(str(root / "**" / ext), recursive=True)))
        
        for f in files:
            speaker_id = f"{prefix}_{Path(f).stem}"
            self.speaker_files[speaker_id] = [f]
        
        self._rebuild()
        print(f"[LocalSpeechIndex] {prefix}: {len(files)} files from {root_dir}")
        return len(files)
    
    def _rebuild(self):
        self.all_files = []
        self.all_speakers = list(self.speaker_files.keys())
        for files in self.speaker_files.values():
            self.all_files.extend(files)
    
    def get_random_speaker(self, exclude: str = None) -> str:
        candidates = [s for s in self.all_speakers if s != exclude] if exclude else self.all_speakers
        return random.choice(candidates) if candidates else self.all_speakers[0]
    
    def get_random_file(self, speaker_id: str, exclude_file: str = None) -> str:
        files = self.speaker_files[speaker_id]
        if exclude_file and len(files) > 1:
            files = [f for f in files if f != exclude_file]
        return random.choice(files)
    
    def get_speakers_with_multiple(self, min_files: int = 2) -> List[str]:
        return [s for s, files in self.speaker_files.items() if len(files) >= min_files]


class LocalNoiseIndex:
    """Indexes local noise files by category."""
    
    def __init__(self):
        self.noise_files: Dict[str, List[str]] = {}
        self.rir_files: List[str] = []
    
    def add_musan(self, root_dir: str):
        """Index MUSAN: root/{music,speech,noise}/**/*.wav"""
        root = Path(root_dir)
        if not root.exists():
            print(f"[LocalNoiseIndex] Warning: {root_dir} not found, skipping")
            return
        
        for category in ['music', 'speech', 'noise']:
            cat_dir = root / category
            if cat_dir.exists():
                files = sorted(glob.glob(str(cat_dir / "**" / "*.wav"), recursive=True))
                self.noise_files.setdefault(category, []).extend(files)
                print(f"[LocalNoiseIndex] MUSAN/{category}: {len(files)} files")
    
    def add_demand(self, root_dir: str):
        """Index DEMAND: root/ENV_NAME/*.wav — car, street, cafe, metro noise."""
        root = Path(root_dir)
        if not root.exists():
            print(f"[LocalNoiseIndex] Warning: {root_dir} not found, skipping")
            return
        
        car_keys = ['DCAR', 'OCAR', 'SCAR', 'TCAR', 'NCAR', 'PCAR']
        street_keys = ['STRAFFIC', 'SCAFE', 'SPSQUARE', 'TMETRO', 'PRESTO']
        
        for env_dir in sorted(root.iterdir()):
            if not env_dir.is_dir():
                continue
            files = sorted(glob.glob(str(env_dir / "*.wav")))
            name = env_dir.name.upper()
            
            if any(name.startswith(k) for k in car_keys):
                self.noise_files.setdefault('car', []).extend(files)
            elif any(name.startswith(k) for k in street_keys):
                self.noise_files.setdefault('street', []).extend(files)
            
            self.noise_files.setdefault('environment', []).extend(files)
        
        car_n = len(self.noise_files.get('car', []))
        street_n = len(self.noise_files.get('street', []))
        env_n = len(self.noise_files.get('environment', []))
        print(f"[LocalNoiseIndex] DEMAND: car={car_n}, street={street_n}, total={env_n}")
    
    def add_rir(self, root_dir: str):
        """Index Room Impulse Response files."""
        root = Path(root_dir)
        if not root.exists():
            print(f"[LocalNoiseIndex] Warning: {root_dir} not found, skipping")
            return
        
        files = sorted(glob.glob(str(root / "**" / "*.wav"), recursive=True))
        self.rir_files.extend(files)
        print(f"[LocalNoiseIndex] RIR: {len(files)} impulse responses")
    
    def add_directory(self, root_dir: str, category: str = 'noise'):
        """Add any directory of noise WAV files."""
        root = Path(root_dir)
        if not root.exists():
            return
        
        files = []
        for ext in ['*.wav', '*.flac', '*.mp3']:
            files.extend(sorted(glob.glob(str(root / "**" / ext), recursive=True)))
        
        self.noise_files.setdefault(category, []).extend(files)
        print(f"[LocalNoiseIndex] {category}: {len(files)} files from {root_dir}")
    
    def get_random_noise(self, categories: List[str] = None) -> Optional[torch.Tensor]:
        """Get a random noise waveform from local files."""
        if categories is None:
            categories = list(self.noise_files.keys())
        
        available = []
        for cat in categories:
            available.extend(self.noise_files.get(cat, []))
        
        if not available:
            return None
        
        path = random.choice(available)
        try:
            return load_audio(path, 16000)
        except Exception:
            return None
    
    def get_random_rir(self) -> Optional[torch.Tensor]:
        if not self.rir_files:
            return None
        path = random.choice(self.rir_files)
        try:
            return load_audio(path, 16000)
        except Exception:
            return None
    
    def has_noise(self) -> bool:
        return any(len(v) > 0 for v in self.noise_files.values())
    
    def has_rir(self) -> bool:
        return len(self.rir_files) > 0


# ============================================================================
# Main TSE Dataset
# ============================================================================

class TSEDataset(Dataset):
    """On-the-fly mixture generation dataset for Target Speaker Extraction.
    
    For each sample, generates a training triplet:
        1. mixture: target voice + interferer(s) + noise
        2. clean_target: the clean target voice (ground truth)
        3. enrollment: a DIFFERENT clip of the same target speaker
    
    The SAME speaker appears in both input and output — the model learns
    to extract the voice matching the enrollment embedding.
    
    Supports mixing data from multiple sources:
        - HuggingFace datasets (any language, streaming or downloaded)
        - Local files (LibriSpeech, VoxCeleb, custom directories)
        - HuggingFace noise datasets + local noise (MUSAN, DEMAND, etc.)
    
    Args:
        hf_speech_sources: List of HFSpeechSource objects
        local_speech_index: Optional LocalSpeechIndex for local files
        hf_noise_sources: List of HFNoiseSource objects
        local_noise_index: Optional LocalNoiseIndex for local noise
        speaker_encoder: Optional SpeakerEncoder for real embeddings
        segment_length: Audio segment length in samples (4s = 64000 at 16kHz)
        sample_rate: Audio sample rate
        num_interferers: Max number of interfering speakers (1-3)
        sir_range: (min, max) Signal-to-Interference Ratio in dB
        snr_range: (min, max) Signal-to-Noise Ratio in dB
        noise_prob: Probability of adding noise
        rir_prob: Probability of adding reverb
        speed_perturb_prob: Probability of speed perturbation
        samples_per_epoch: Virtual epoch size
    """
    
    def __init__(
        self,
        hf_speech_sources: List[HFSpeechSource] = None,
        local_speech_index: LocalSpeechIndex = None,
        hf_noise_sources: List[HFNoiseSource] = None,
        local_noise_index: LocalNoiseIndex = None,
        speaker_encoder=None,
        segment_length: int = 64000,
        sample_rate: int = 16000,
        num_interferers: int = 2,
        sir_range: Tuple[float, float] = (-5.0, 5.0),
        snr_range: Tuple[float, float] = (5.0, 20.0),
        noise_prob: float = 0.7,
        rir_prob: float = 0.3,
        speed_perturb_prob: float = 0.3,
        speed_perturb_range: Tuple[float, float] = (0.9, 1.1),
        samples_per_epoch: int = 10000,
    ):
        self.hf_sources = hf_speech_sources or []
        self.local_index = local_speech_index
        self.hf_noise_sources = hf_noise_sources or []
        self.local_noise = local_noise_index
        self.speaker_encoder = speaker_encoder
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.num_interferers = num_interferers
        self.sir_range = sir_range
        self.snr_range = snr_range
        self.noise_prob = noise_prob
        self.rir_prob = rir_prob
        self.speed_perturb_prob = speed_perturb_prob
        self.speed_perturb_range = speed_perturb_range
        self.samples_per_epoch = samples_per_epoch
        
        # Weighted source selection (HF sources weighted by number of samples)
        self._source_weights: List[float] = []
        self._all_sources: List[Union[HFSpeechSource, str]] = []  # str = 'local'
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        
        self._init_sources()
    
    def _init_sources(self):
        """Initialize and weight all speech sources."""
        total_speakers = 0
        
        # Add HF sources
        for src in self.hf_sources:
            n = max(src.num_speakers, 1)
            self._all_sources.append(src)
            self._source_weights.append(float(n))
            total_speakers += n
        
        # Add local source
        if self.local_index and self.local_index.all_speakers:
            n = len(self.local_index.all_speakers)
            self._all_sources.append('local')
            self._source_weights.append(float(n))
            total_speakers += n
        
        # Normalize weights
        if self._source_weights:
            total = sum(self._source_weights)
            self._source_weights = [w / total for w in self._source_weights]
        
        print(f"\n[TSEDataset] === Data Summary ===")
        print(f"  Speech sources: {len(self._all_sources)}")
        print(f"  Total speakers: ~{total_speakers}")
        print(f"  Noise sources (HF): {len(self.hf_noise_sources)}")
        print(f"  Noise sources (local): {'yes' if self.local_noise and self.local_noise.has_noise() else 'no'}")
        print(f"  RIR available: {'yes' if self.local_noise and self.local_noise.has_rir() else 'no'}")
        print(f"  Samples/epoch: {self.samples_per_epoch}")
        print(f"  Segment length: {self.segment_length/self.sample_rate:.1f}s")
        print()
    
    def __len__(self):
        return self.samples_per_epoch
    
    def _pick_source(self) -> Union[HFSpeechSource, str]:
        """Pick a random speech source weighted by speaker count."""
        if not self._all_sources:
            raise ValueError("No speech sources configured!")
        return random.choices(self._all_sources, weights=self._source_weights, k=1)[0]
    
    def _get_target_and_enrollment(self, source) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get target audio + enrollment audio from the SAME speaker.
        
        Returns:
            (target_waveform, enrollment_waveform, speaker_id)
        """
        if isinstance(source, HFSpeechSource):
            # Get speakers with multiple recordings
            multi_speakers = source.get_speakers_with_multiple(min_files=2)
            
            if multi_speakers:
                speaker = random.choice(multi_speakers)
                idx1, sample1 = source.get_random_file_for_speaker(speaker)
                idx2, sample2 = source.get_random_file_for_speaker(speaker, exclude_idx=idx1)
                
                target = decode_hf_audio(sample1, source.audio_column, self.sample_rate)
                enrollment = decode_hf_audio(sample2, source.audio_column, self.sample_rate)
            else:
                # Fallback: same audio for target and enrollment (suboptimal but works)
                speaker = source.get_random_speaker()
                _, sample = source.get_random_file_for_speaker(speaker)
                audio = decode_hf_audio(sample, source.audio_column, self.sample_rate)
                target = audio
                enrollment = audio
            
            return target, enrollment, speaker
        
        else:  # 'local'
            multi_speakers = self.local_index.get_speakers_with_multiple(min_files=2)
            
            if multi_speakers:
                speaker = random.choice(multi_speakers)
            else:
                speaker = self.local_index.get_random_speaker()
            
            target_file = self.local_index.get_random_file(speaker)
            enrollment_file = self.local_index.get_random_file(speaker, exclude_file=target_file)
            
            target = load_audio(target_file, self.sample_rate)
            enrollment = load_audio(enrollment_file, self.sample_rate)
            
            return target, enrollment, speaker
    
    def _get_interferer(self, exclude_speaker: str = None) -> torch.Tensor:
        """Get an interfering speaker's audio from any source."""
        source = self._pick_source()
        
        if isinstance(source, HFSpeechSource):
            speaker = source.get_random_speaker(exclude=exclude_speaker)
            _, sample = source.get_random_file_for_speaker(speaker)
            return decode_hf_audio(sample, source.audio_column, self.sample_rate)
        else:
            speaker = self.local_index.get_random_speaker(exclude=exclude_speaker)
            filepath = self.local_index.get_random_file(speaker)
            return load_audio(filepath, self.sample_rate)
    
    def _get_noise(self) -> Optional[torch.Tensor]:
        """Get a random noise sample from HF or local sources."""
        all_noise_sources = []
        
        # HF noise sources
        for ns in self.hf_noise_sources:
            all_noise_sources.append(('hf', ns))
        
        # Local noise
        if self.local_noise and self.local_noise.has_noise():
            all_noise_sources.append(('local', self.local_noise))
        
        if not all_noise_sources:
            return None
        
        source_type, source = random.choice(all_noise_sources)
        
        try:
            if source_type == 'hf':
                return source.get_random_noise()
            else:
                return source.get_random_noise()
        except Exception:
            return None
    
    def _get_rir(self) -> Optional[torch.Tensor]:
        """Get a random room impulse response."""
        if self.local_noise and self.local_noise.has_rir():
            return self.local_noise.get_random_rir()
        return None
    
    def _get_speaker_embedding(self, enrollment_waveform: torch.Tensor, 
                                speaker_id: str) -> torch.Tensor:
        """Compute or generate speaker embedding."""
        if self.speaker_encoder is not None:
            cache_key = f"{speaker_id}_{hash(enrollment_waveform.data_ptr())}"
            if cache_key not in self._embedding_cache:
                emb = self.speaker_encoder.encode_batch(enrollment_waveform.unsqueeze(0))
                self._embedding_cache[cache_key] = emb.squeeze(0).cpu()
            return self._embedding_cache[cache_key]
        else:
            # Deterministic random embedding per speaker (for testing without encoder)
            if speaker_id not in self._embedding_cache:
                gen = torch.Generator()
                gen.manual_seed(hash(speaker_id) % (2**32))
                emb = torch.randn(192, generator=gen)
                emb = F.normalize(emb, p=2, dim=0)
                self._embedding_cache[speaker_id] = emb
            return self._embedding_cache[speaker_id]
    
    def _apply_speed_perturbation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random speed perturbation."""
        speed_factor = random.uniform(*self.speed_perturb_range)
        if abs(speed_factor - 1.0) < 0.01:
            return waveform
        
        try:
            effects = [['speed', str(speed_factor)], ['rate', str(self.sample_rate)]]
            modified, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform.unsqueeze(0), self.sample_rate, effects
            )
            return modified.squeeze(0)
        except Exception:
            return waveform
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Generate one training triplet.
        
        Returns:
            dict with:
                'mixture': (segment_length,) — mixed audio
                'target': (segment_length,) — clean target speaker
                'enrollment': (192,) — speaker embedding
        """
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return self._generate_sample()
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[TSEDataset] Failed after {max_retries} retries: {e}")
                    # Return silence as last resort
                    return {
                        'mixture': torch.zeros(self.segment_length),
                        'target': torch.zeros(self.segment_length),
                        'enrollment': torch.randn(192),
                    }
    
    def _generate_sample(self) -> Dict[str, torch.Tensor]:
        """Core sample generation logic."""
        # 1. Pick source and get target + enrollment from SAME speaker
        source = self._pick_source()
        target_wav, enrollment_wav, speaker_id = self._get_target_and_enrollment(source)
        
        # 2. Get segments
        target = random_segment(target_wav, self.segment_length)
        target = rms_normalize(target)
        
        enrollment_seg = random_segment(enrollment_wav, self.segment_length)
        
        # 3. Speed perturbation (optional)
        if random.random() < self.speed_perturb_prob:
            target = self._apply_speed_perturbation(target)
            target = random_segment(target, self.segment_length)
        
        # 4. Compute speaker embedding from enrollment
        embedding = self._get_speaker_embedding(enrollment_seg, speaker_id)
        
        # 5. Build mixture: start with target
        mixture = target.clone()
        
        # 6. Add interfering speakers
        num_int = random.randint(1, self.num_interferers)
        for _ in range(num_int):
            try:
                interferer = self._get_interferer(exclude_speaker=speaker_id)
                interferer = random_segment(interferer, self.segment_length)
                interferer = rms_normalize(interferer)
                
                sir_db = random.uniform(*self.sir_range)
                scaled_int = mix_at_snr(target, interferer, sir_db)
                mixture = mixture + scaled_int
            except Exception:
                continue
        
        # 7. Add reverb (optional)
        if random.random() < self.rir_prob:
            rir = self._get_rir()
            if rir is not None:
                try:
                    mixture = convolve_rir(mixture, rir)
                    target_rev = convolve_rir(target, rir)
                    # Use reverbed target as ground truth (model should output reverbed version)
                    target = target_rev
                except Exception:
                    pass
        
        # 8. Add environmental noise (optional)
        if random.random() < self.noise_prob:
            noise = self._get_noise()
            if noise is not None:
                try:
                    noise = random_segment(noise, self.segment_length)
                    snr_db = random.uniform(*self.snr_range)
                    scaled_noise = mix_at_snr(mixture, noise, snr_db)
                    mixture = mixture + scaled_noise
                except Exception:
                    pass
        
        # 9. Peak normalization to prevent clipping
        max_val = torch.max(torch.abs(mixture))
        if max_val > 0.95:
            scale = 0.9 / max_val
            mixture = mixture * scale
            target = target * scale
        
        # Ensure correct length
        mixture = random_segment(mixture, self.segment_length)
        target = random_segment(target, self.segment_length)
        
        return {
            'mixture': mixture,
            'target': target,
            'enrollment': embedding,
        }


# ============================================================================
# Factory Function — Easy Dataset Creation
# ============================================================================

def create_tse_dataloader(config: dict, speaker_encoder=None) -> DataLoader:
    """Create a TSE DataLoader from a config dict.
    
    Config format:
    ```yaml
    data:
      sample_rate: 16000
      segment_length: 64000
      batch_size: 16
      num_workers: 4
      samples_per_epoch: 10000
      
      # Mixing parameters
      num_interferers: 2
      sir_range: [-5, 5]
      snr_range: [5, 20]
      noise_prob: 0.7
      rir_prob: 0.3
      speed_perturb_prob: 0.3
      
      # HuggingFace speech datasets
      hf_speech:
        - name: "openslr/librispeech_asr"
          subset: "train.clean.100"
          audio_column: "audio"
          speaker_column: "speaker_id"
          streaming: false
          max_samples: 50000
        
        - name: "mozilla-foundation/common_voice_17_0"
          subset: "ar"
          audio_column: "audio"
          speaker_column: "client_id"
          streaming: true
      
      # HuggingFace noise datasets
      hf_noise:
        - name: "flozi00/MUSAN-Noise"
          audio_column: "audio"
          category: "noise"
          streaming: true
      
      # Local speech directories
      local_speech:
        librispeech: "/data/LibriSpeech/train-clean-100"
        voxceleb: "/data/VoxCeleb1/wav"
      
      # Local noise directories
      local_noise:
        musan: "/data/musan"
        demand: "/data/DEMAND"
        rir: "/data/RIRS_NOISES/simulated_rirs"
    ```
    """
    data_cfg = config.get('data', config)
    
    sample_rate = data_cfg.get('sample_rate', 16000)
    segment_length = data_cfg.get('segment_length', 64000)
    batch_size = data_cfg.get('batch_size', 16)
    num_workers = data_cfg.get('num_workers', 4)
    samples_per_epoch = data_cfg.get('samples_per_epoch', 10000)
    
    # === Build HF Speech Sources ===
    hf_speech_sources = []
    for src_cfg in data_cfg.get('hf_speech', []):
        source = HFSpeechSource(
            dataset_name=src_cfg['name'],
            subset=src_cfg.get('subset'),
            split=src_cfg.get('split', 'train'),
            audio_column=src_cfg.get('audio_column', 'audio'),
            speaker_column=src_cfg.get('speaker_column'),
            streaming=src_cfg.get('streaming', False),
            max_samples=src_cfg.get('max_samples'),
            sample_rate=sample_rate,
        )
        hf_speech_sources.append(source)
    
    # === Build HF Noise Sources ===
    hf_noise_sources = []
    for ns_cfg in data_cfg.get('hf_noise', []):
        noise_src = HFNoiseSource(
            dataset_name=ns_cfg['name'],
            subset=ns_cfg.get('subset'),
            split=ns_cfg.get('split', 'train'),
            audio_column=ns_cfg.get('audio_column', 'audio'),
            label_column=ns_cfg.get('label_column'),
            streaming=ns_cfg.get('streaming', False),
            max_samples=ns_cfg.get('max_samples'),
            sample_rate=sample_rate,
            category=ns_cfg.get('category', 'noise'),
        )
        hf_noise_sources.append(noise_src)
    
    # === Build Local Speech Index ===
    local_speech = None
    local_speech_cfg = data_cfg.get('local_speech', {})
    if local_speech_cfg:
        local_speech = LocalSpeechIndex()
        for fmt, path in local_speech_cfg.items():
            if fmt == 'librispeech':
                local_speech.add_librispeech(path)
            elif fmt == 'voxceleb':
                local_speech.add_voxceleb(path)
            else:
                local_speech.add_directory(path, prefix=fmt)
    
    # === Build Local Noise Index ===
    local_noise = None
    local_noise_cfg = data_cfg.get('local_noise', {})
    if local_noise_cfg:
        local_noise = LocalNoiseIndex()
        if 'musan' in local_noise_cfg:
            local_noise.add_musan(local_noise_cfg['musan'])
        if 'demand' in local_noise_cfg:
            local_noise.add_demand(local_noise_cfg['demand'])
        if 'rir' in local_noise_cfg:
            local_noise.add_rir(local_noise_cfg['rir'])
        for key, path in local_noise_cfg.items():
            if key not in ('musan', 'demand', 'rir'):
                local_noise.add_directory(path, category=key)
    
    # === Create Dataset ===
    dataset = TSEDataset(
        hf_speech_sources=hf_speech_sources,
        local_speech_index=local_speech,
        hf_noise_sources=hf_noise_sources,
        local_noise_index=local_noise,
        speaker_encoder=speaker_encoder,
        segment_length=segment_length,
        sample_rate=sample_rate,
        num_interferers=data_cfg.get('num_interferers', 2),
        sir_range=tuple(data_cfg.get('sir_range', [-5, 5])),
        snr_range=tuple(data_cfg.get('snr_range', [5, 20])),
        noise_prob=data_cfg.get('noise_prob', 0.7),
        rir_prob=data_cfg.get('rir_prob', 0.3),
        speed_perturb_prob=data_cfg.get('speed_perturb_prob', 0.3),
        samples_per_epoch=samples_per_epoch,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
