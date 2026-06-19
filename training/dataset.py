"""
On-the-fly noisy/clean pair generation for Speech Enhancement.

Phase 1: Speech Enhancement
    Creates (noisy, clean) pairs dynamically:
        noisy = clean_speech + noise (noise ALWAYS quieter than speech)
    
    Noise types: MUSAN, DEMAND (car/street), UrbanSound, HuggingFace, reverb

Phase 2 (future): Target Speaker Extraction
    Creates (mixture, clean_target, enrollment) triplets

Supports:
    - HuggingFace datasets (streaming + download) with configurable audio columns
    - Local datasets (LibriSpeech, Common Voice, VoxCeleb, flat directories)
    - HuggingFace noise datasets
    - Local noise (MUSAN, DEMAND car/street, custom)
    - Room impulse responses (reverb)
    - Speed/pitch perturbation
    - Noise is ALWAYS at lower volume than speech (positive SNR)
"""

import os
import random
import glob
import math
import io
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np


# ============================================================================
# Audio Utilities
# ============================================================================

def load_audio(path: str, target_sr: int = 16000, mono: bool = True) -> torch.Tensor:
    """Load audio file → (T,) tensor at target_sr."""
    waveform, sr = torchaudio.load(path)
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    return waveform.squeeze(0)


def decode_hf_audio(sample: dict, audio_column: str, target_sr: int = 16000) -> torch.Tensor:
    """Decode audio from a HuggingFace dataset sample.
    
    Handles all HF audio formats:
        - {'array': np.ndarray, 'sampling_rate': int}  (decoded)
        - {'path': str, 'bytes': bytes}                 (raw)
        - str (path)
    """
    audio_data = sample[audio_column]
    
    if isinstance(audio_data, dict) and 'array' in audio_data:
        waveform = torch.tensor(audio_data['array'], dtype=torch.float32)
        sr = audio_data.get('sampling_rate', target_sr)
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform.unsqueeze(0)).squeeze(0)
        return waveform
    
    if isinstance(audio_data, dict) and 'bytes' in audio_data and audio_data['bytes']:
        buffer = io.BytesIO(audio_data['bytes'])
        waveform, sr = torchaudio.load(buffer)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        return waveform.squeeze(0)
    
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
    
    Positive SNR = noise is QUIETER than signal (what we want!)
    Example: SNR=10dB means signal is 10dB louder than noise.
    
    Returns:
        scaled_noise: (T,) noise at correct volume
    """
    signal_power = torch.mean(signal ** 2) + 1e-8
    noise_power = torch.mean(noise ** 2) + 1e-8
    snr_linear = 10 ** (snr_db / 10)
    scale = torch.sqrt(signal_power / (noise_power * snr_linear))
    return noise * scale


def convolve_rir(waveform: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    """Apply room impulse response (reverb)."""
    rir = rir / (torch.max(torch.abs(rir)) + 1e-8)
    waveform_3d = waveform.unsqueeze(0).unsqueeze(0)
    rir_3d = rir.flip(0).unsqueeze(0).unsqueeze(0)
    convolved = F.conv1d(waveform_3d, rir_3d, padding=rir.shape[0] - 1)
    result = convolved.squeeze()[:waveform.shape[0]]
    
    orig_rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-8)
    new_rms = torch.sqrt(torch.mean(result ** 2) + 1e-8)
    return result * (orig_rms / (new_rms + 1e-8))


def random_segment(waveform: torch.Tensor, length: int) -> torch.Tensor:
    """Extract a random segment. Loops if too short."""
    T = waveform.shape[0]
    if T >= length:
        offset = random.randint(0, T - length)
        return waveform[offset:offset + length]
    else:
        repeats = math.ceil(length / T)
        return waveform.repeat(repeats)[:length]


# ============================================================================
# HuggingFace Speech Source
# ============================================================================

class HFSpeechSource:
    """Speech data from a HuggingFace dataset. Configurable audio column."""
    
    def __init__(self, dataset_name: str, subset: str = None, split: str = "train",
                 audio_column: str = "audio", speaker_column: str = None,
                 streaming: bool = False, max_samples: int = None,
                 trust_remote_code: bool = True, sample_rate: int = 16000):
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.audio_column = audio_column
        self.speaker_column = speaker_column
        self.streaming = streaming
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        self.trust_remote_code = trust_remote_code
        
        self._dataset = None
        self._loaded = False
        self._buffer: List[dict] = []
        self._buffer_size = 50
    
    def _load(self):
        if self._loaded:
            return
        from datasets import load_dataset, Audio
        
        print(f"[HFSpeech] Loading {self.dataset_name}"
              f"{f'/{self.subset}' if self.subset else ''} "
              f"(streaming={self.streaming})...")
        
        kwargs = {'split': self.split, 'streaming': self.streaming,
                  'trust_remote_code': self.trust_remote_code}
        if self.subset:
            kwargs['name'] = self.subset
        
        self._dataset = load_dataset(self.dataset_name, **kwargs)
        
        if not self.streaming:
            try:
                self._dataset = self._dataset.cast_column(
                    self.audio_column, Audio(sampling_rate=self.sample_rate))
            except Exception:
                pass
            if self.max_samples and len(self._dataset) > self.max_samples:
                self._dataset = self._dataset.select(range(self.max_samples))
            print(f"[HFSpeech] Loaded {len(self._dataset)} samples")
        else:
            print(f"[HFSpeech] Streaming mode — buffering on first access")
        
        self._loaded = True
    
    def _fill_buffer(self):
        if self._buffer:
            return
        self._load()
        print(f"[HFSpeech] Filling buffer ({self._buffer_size} samples)...")
        count = 0
        for sample in self._dataset:
            self._buffer.append(sample)
            count += 1
            if count >= self._buffer_size:
                break
        print(f"[HFSpeech] Buffered {len(self._buffer)} samples")
    
    def get_random_audio(self) -> torch.Tensor:
        """Get a random audio waveform."""
        self._load()
        if self.streaming:
            self._fill_buffer()
            sample = random.choice(self._buffer)
        else:
            idx = random.randint(0, len(self._dataset) - 1)
            sample = self._dataset[idx]
        return decode_hf_audio(sample, self.audio_column, self.sample_rate)
    
    @property
    def num_samples(self) -> int:
        self._load()
        if self.streaming:
            return 10000
        return len(self._dataset)


class HFNoiseSource:
    """Noise data from a HuggingFace dataset."""
    
    def __init__(self, dataset_name: str, subset: str = None, split: str = "train",
                 audio_column: str = "audio", label_column: str = None,
                 streaming: bool = False, max_samples: int = None,
                 trust_remote_code: bool = True, sample_rate: int = 16000,
                 category: str = "noise"):
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
        self._buffer_size = 50
    
    def _load(self):
        if self._loaded:
            return
        from datasets import load_dataset, Audio
        
        print(f"[HFNoise] Loading {self.dataset_name} (category={self.category})...")
        kwargs = {'split': self.split, 'streaming': self.streaming,
                  'trust_remote_code': self.trust_remote_code}
        if self.subset:
            kwargs['name'] = self.subset
        
        self._dataset = load_dataset(self.dataset_name, **kwargs)
        
        if not self.streaming:
            try:
                self._dataset = self._dataset.cast_column(
                    self.audio_column, Audio(sampling_rate=self.sample_rate))
            except Exception:
                pass
            if self.max_samples and len(self._dataset) > self.max_samples:
                self._dataset = self._dataset.select(range(self.max_samples))
            print(f"[HFNoise] Loaded {len(self._dataset)} noise samples")
        else:
            print(f"[HFNoise] Streaming — buffering on first access")
        self._loaded = True
    
    def _fill_buffer(self):
        if self._buffer:
            return
        self._load()
        print(f"[HFNoise] Filling buffer ({self._buffer_size})...")
        count = 0
        for sample in self._dataset:
            self._buffer.append(sample)
            count += 1
            if count >= self._buffer_size:
                break
        print(f"[HFNoise] Buffered {len(self._buffer)} noise samples")
    
    def get_random_noise(self) -> torch.Tensor:
        self._load()
        if self.streaming:
            self._fill_buffer()
            sample = random.choice(self._buffer)
        else:
            idx = random.randint(0, len(self._dataset) - 1)
            sample = self._dataset[idx]
        return decode_hf_audio(sample, self.audio_column, self.sample_rate)


# ============================================================================
# Local File Indexers
# ============================================================================

class LocalSpeechIndex:
    """Indexes local audio files."""
    
    def __init__(self):
        self.all_files: List[str] = []
    
    def add_librispeech(self, root_dir: str):
        root = Path(root_dir)
        if not root.exists():
            print(f"[LocalSpeech] Warning: {root_dir} not found")
            return
        files = sorted(glob.glob(str(root / "**" / "*.flac"), recursive=True))
        files += sorted(glob.glob(str(root / "**" / "*.wav"), recursive=True))
        self.all_files.extend(files)
        print(f"[LocalSpeech] LibriSpeech: {len(files)} files from {root_dir}")
    
    def add_directory(self, root_dir: str, prefix: str = "custom"):
        root = Path(root_dir)
        if not root.exists():
            print(f"[LocalSpeech] Warning: {root_dir} not found")
            return
        files = []
        for ext in ['*.wav', '*.flac', '*.mp3', '*.ogg']:
            files.extend(sorted(glob.glob(str(root / "**" / ext), recursive=True)))
        self.all_files.extend(files)
        print(f"[LocalSpeech] {prefix}: {len(files)} files")
    
    def add_voxceleb(self, root_dir: str):
        root = Path(root_dir)
        if not root.exists():
            return
        files = sorted(glob.glob(str(root / "**" / "*.wav"), recursive=True))
        self.all_files.extend(files)
        print(f"[LocalSpeech] VoxCeleb: {len(files)} files")
    
    def get_random_audio(self) -> torch.Tensor:
        path = random.choice(self.all_files)
        return load_audio(path, 16000)


class LocalNoiseIndex:
    """Indexes local noise files by category."""
    
    def __init__(self):
        self.noise_files: Dict[str, List[str]] = {}
        self.rir_files: List[str] = []
    
    def add_musan(self, root_dir: str):
        root = Path(root_dir)
        if not root.exists():
            print(f"[LocalNoise] Warning: {root_dir} not found")
            return
        for category in ['music', 'speech', 'noise']:
            cat_dir = root / category
            if cat_dir.exists():
                files = sorted(glob.glob(str(cat_dir / "**" / "*.wav"), recursive=True))
                self.noise_files.setdefault(category, []).extend(files)
                print(f"[LocalNoise] MUSAN/{category}: {len(files)} files")
    
    def add_demand(self, root_dir: str):
        root = Path(root_dir)
        if not root.exists():
            print(f"[LocalNoise] Warning: {root_dir} not found")
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
        
        print(f"[LocalNoise] DEMAND: car={len(self.noise_files.get('car', []))}, "
              f"street={len(self.noise_files.get('street', []))}")
    
    def add_rir(self, root_dir: str):
        root = Path(root_dir)
        if not root.exists():
            return
        files = sorted(glob.glob(str(root / "**" / "*.wav"), recursive=True))
        self.rir_files.extend(files)
        print(f"[LocalNoise] RIR: {len(files)} impulse responses")
    
    def add_directory(self, root_dir: str, category: str = 'noise'):
        root = Path(root_dir)
        if not root.exists():
            return
        files = []
        for ext in ['*.wav', '*.flac', '*.mp3']:
            files.extend(sorted(glob.glob(str(root / "**" / ext), recursive=True)))
        self.noise_files.setdefault(category, []).extend(files)
        print(f"[LocalNoise] {category}: {len(files)} files")
    
    def get_random_noise(self) -> Optional[torch.Tensor]:
        all_files = []
        for files in self.noise_files.values():
            all_files.extend(files)
        if not all_files:
            return None
        try:
            return load_audio(random.choice(all_files), 16000)
        except Exception:
            return None
    
    def get_random_rir(self) -> Optional[torch.Tensor]:
        if not self.rir_files:
            return None
        try:
            return load_audio(random.choice(self.rir_files), 16000)
        except Exception:
            return None
    
    def has_noise(self) -> bool:
        return any(len(v) > 0 for v in self.noise_files.values())
    
    def has_rir(self) -> bool:
        return len(self.rir_files) > 0


# ============================================================================
# Speech Enhancement Dataset
# ============================================================================

class SpeechEnhancementDataset(Dataset):
    """On-the-fly noisy/clean pair generation for speech enhancement.
    
    For each sample:
        clean = random speech utterance (ground truth)
        noisy = clean + noise at POSITIVE SNR (noise always quieter than speech!)
    
    The model learns: noisy → clean
    
    Noise sources (all optional, uses whatever is available):
        - MUSAN (babble, music, machinery)
        - DEMAND (car, street, cafe, metro)  
        - UrbanSound (car horns, sirens, engines)
        - HuggingFace noise datasets
        - Room reverb (RIR convolution)
    
    Args:
        hf_speech_sources: HuggingFace speech datasets
        local_speech_index: Local speech files
        hf_noise_sources: HuggingFace noise datasets
        local_noise_index: Local noise files
        segment_length: Audio segment length in samples
        snr_range: (min, max) SNR in dB — ALWAYS POSITIVE (noise quieter than speech)
        noise_prob: Probability of adding noise (remaining are clean passthrough)
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
        segment_length: int = 64000,
        sample_rate: int = 16000,
        snr_range: Tuple[float, float] = (3.0, 20.0),
        noise_prob: float = 0.8,
        rir_prob: float = 0.3,
        speed_perturb_prob: float = 0.2,
        speed_perturb_range: Tuple[float, float] = (0.95, 1.05),
        samples_per_epoch: int = 10000,
    ):
        self.hf_sources = hf_speech_sources or []
        self.local_index = local_speech_index
        self.hf_noise = hf_noise_sources or []
        self.local_noise = local_noise_index
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        self.noise_prob = noise_prob
        self.rir_prob = rir_prob
        self.speed_perturb_prob = speed_perturb_prob
        self.speed_perturb_range = speed_perturb_range
        self.samples_per_epoch = samples_per_epoch
        
        # Validate SNR range
        assert snr_range[0] >= 0, (
            f"SNR range must be positive (noise quieter than speech)! "
            f"Got snr_range={snr_range}. Use e.g. (3, 20)."
        )
        
        self._speech_sources = []
        self._speech_weights = []
        self._init_sources()
    
    def _init_sources(self):
        for src in self.hf_sources:
            self._speech_sources.append(('hf', src))
            self._speech_weights.append(max(1, src.num_samples))
        
        if self.local_index and self.local_index.all_files:
            self._speech_sources.append(('local', self.local_index))
            self._speech_weights.append(len(self.local_index.all_files))
        
        if self._speech_weights:
            total = sum(self._speech_weights)
            self._speech_weights = [w / total for w in self._speech_weights]
        
        noise_count = len(self.hf_noise)
        if self.local_noise and self.local_noise.has_noise():
            noise_count += 1
        
        print(f"\n[Dataset] ═══ Speech Enhancement Data ═══")
        print(f"  Speech sources : {len(self._speech_sources)}")
        print(f"  Noise sources  : {noise_count}")
        print(f"  RIR available  : {'yes' if self.local_noise and self.local_noise.has_rir() else 'no'}")
        print(f"  SNR range      : {self.snr_range[0]:.0f} – {self.snr_range[1]:.0f} dB (noise ALWAYS quieter)")
        print(f"  Noise prob     : {self.noise_prob:.0%}")
        print(f"  Reverb prob    : {self.rir_prob:.0%}")
        print(f"  Segment        : {self.segment_length/self.sample_rate:.1f}s")
        print(f"  Samples/epoch  : {self.samples_per_epoch}")
        print()
    
    def __len__(self):
        return self.samples_per_epoch
    
    def _get_speech(self) -> torch.Tensor:
        """Get a random speech waveform from any source."""
        src_type, src = random.choices(self._speech_sources, 
                                        weights=self._speech_weights, k=1)[0]
        if src_type == 'hf':
            return src.get_random_audio()
        else:
            return src.get_random_audio()
    
    def _get_noise(self) -> Optional[torch.Tensor]:
        """Get random noise from HF or local sources."""
        sources = []
        for ns in self.hf_noise:
            sources.append(('hf', ns))
        if self.local_noise and self.local_noise.has_noise():
            sources.append(('local', self.local_noise))
        
        if not sources:
            return None
        
        src_type, src = random.choice(sources)
        try:
            if src_type == 'hf':
                return src.get_random_noise()
            else:
                return src.get_random_noise()
        except Exception:
            return None
    
    def _get_rir(self) -> Optional[torch.Tensor]:
        if self.local_noise and self.local_noise.has_rir():
            return self.local_noise.get_random_rir()
        return None
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Generate one (noisy, clean) pair.
        
        Returns:
            dict with:
                'noisy': (segment_length,) — degraded audio (input to model)
                'clean': (segment_length,) — original clean speech (target)
        """
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return self._generate_sample()
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[Dataset] Failed after {max_retries} retries: {e}")
                    return {
                        'noisy': torch.zeros(self.segment_length),
                        'clean': torch.zeros(self.segment_length),
                    }
    
    def _generate_sample(self) -> Dict[str, torch.Tensor]:
        """Core sample generation."""
        
        # 1. Get clean speech
        speech = self._get_speech()
        clean = random_segment(speech, self.segment_length)
        clean = rms_normalize(clean)
        
        # 2. Optional speed perturbation (apply to both clean and noisy)
        if random.random() < self.speed_perturb_prob:
            try:
                speed = random.uniform(*self.speed_perturb_range)
                effects = [['speed', str(speed)], ['rate', str(self.sample_rate)]]
                modified, _ = torchaudio.sox_effects.apply_effects_tensor(
                    clean.unsqueeze(0), self.sample_rate, effects)
                clean = random_segment(modified.squeeze(0), self.segment_length)
            except Exception:
                pass
        
        # 3. Start noisy version
        noisy = clean.clone()
        
        # 4. Add reverb BEFORE noise (more realistic — reverb affects speech, not noise)
        if random.random() < self.rir_prob:
            rir = self._get_rir()
            if rir is not None:
                try:
                    noisy = convolve_rir(noisy, rir)
                    # Target remains the clean (non-reverbed) speech
                    # The model learns to remove reverb too!
                except Exception:
                    pass
        
        # 5. Add noise at POSITIVE SNR (noise is ALWAYS quieter than speech)
        if random.random() < self.noise_prob:
            noise = self._get_noise()
            if noise is not None:
                try:
                    noise = random_segment(noise, self.segment_length)
                    
                    # Random SNR in positive range — noise always quieter!
                    snr_db = random.uniform(*self.snr_range)
                    scaled_noise = mix_at_snr(noisy, noise, snr_db)
                    noisy = noisy + scaled_noise
                except Exception:
                    pass
        
        # 6. Peak normalization (prevent clipping)
        max_val = torch.max(torch.abs(noisy))
        if max_val > 0.95:
            scale = 0.9 / max_val
            noisy = noisy * scale
            clean = clean * scale
        
        return {
            'noisy': noisy,
            'clean': clean,
        }


# ============================================================================
# Factory Function
# ============================================================================

def create_dataloader(config: dict) -> DataLoader:
    """Create a DataLoader from config dict.
    
    Config format (in finetune.yaml):
    ```yaml
    data:
      sample_rate: 16000
      segment_length: 64000
      batch_size: 16
      num_workers: 4
      samples_per_epoch: 20000
      snr_range: [3, 20]        # Noise ALWAYS quieter than speech
      noise_prob: 0.8
      rir_prob: 0.3
      
      hf_speech:
        - name: "mozilla-foundation/common_voice_17_0"
          subset: "ar"
          audio_column: "audio"
          streaming: true
      
      hf_noise:
        - name: "flozi00/MUSAN-Noise"
          audio_column: "audio"
          category: "noise"
          streaming: true
      
      local_speech: {}
      local_noise: {}
    ```
    """
    data_cfg = config.get('data', config)
    
    sample_rate = data_cfg.get('sample_rate', 16000)
    segment_length = data_cfg.get('segment_length', 64000)
    batch_size = data_cfg.get('batch_size', 16)
    num_workers = data_cfg.get('num_workers', 4)
    samples_per_epoch = data_cfg.get('samples_per_epoch', 10000)
    
    # Build HF Speech Sources
    hf_speech = []
    for src_cfg in data_cfg.get('hf_speech', []):
        hf_speech.append(HFSpeechSource(
            dataset_name=src_cfg['name'],
            subset=src_cfg.get('subset'),
            split=src_cfg.get('split', 'train'),
            audio_column=src_cfg.get('audio_column', 'audio'),
            speaker_column=src_cfg.get('speaker_column'),
            streaming=src_cfg.get('streaming', False),
            max_samples=src_cfg.get('max_samples'),
            sample_rate=sample_rate,
        ))
    
    # Build HF Noise Sources
    hf_noise = []
    for ns_cfg in data_cfg.get('hf_noise', []):
        hf_noise.append(HFNoiseSource(
            dataset_name=ns_cfg['name'],
            subset=ns_cfg.get('subset'),
            split=ns_cfg.get('split', 'train'),
            audio_column=ns_cfg.get('audio_column', 'audio'),
            label_column=ns_cfg.get('label_column'),
            streaming=ns_cfg.get('streaming', False),
            max_samples=ns_cfg.get('max_samples'),
            sample_rate=sample_rate,
            category=ns_cfg.get('category', 'noise'),
        ))
    
    # Build Local Speech Index
    local_speech = None
    for fmt, path in data_cfg.get('local_speech', {}).items():
        if local_speech is None:
            local_speech = LocalSpeechIndex()
        if fmt == 'librispeech':
            local_speech.add_librispeech(path)
        elif fmt == 'voxceleb':
            local_speech.add_voxceleb(path)
        else:
            local_speech.add_directory(path, prefix=fmt)
    
    # Build Local Noise Index
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
    
    # Create Dataset
    dataset = SpeechEnhancementDataset(
        hf_speech_sources=hf_speech,
        local_speech_index=local_speech,
        hf_noise_sources=hf_noise,
        local_noise_index=local_noise,
        segment_length=segment_length,
        sample_rate=sample_rate,
        snr_range=tuple(data_cfg.get('snr_range', [3, 20])),
        noise_prob=data_cfg.get('noise_prob', 0.8),
        rir_prob=data_cfg.get('rir_prob', 0.3),
        speed_perturb_prob=data_cfg.get('speed_perturb_prob', 0.2),
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


# Backward compatibility alias
create_tse_dataloader = create_dataloader
