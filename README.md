<div align="center">

# 🎤 Your Denoizer

### Personalized Voice Isolation via Target Speaker Extraction

*Give it a 10-second clip of your voice. It extracts only YOU from any recording — suppressing all other speakers, car noise, street sounds, crowd chatter, and music.*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/datasets)

---

</div>

## 🧠 How It Works

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │   Enrollment (once)                                              │
  │   ──────────────────                                             │
  │   Your voice clips ──→ ECAPA-TDNN ──→ Speaker Embedding (192d)  │
  │   (10-30 seconds)       (frozen)        (your voiceprint)        │
  │                                              │                   │
  │   Extraction (per file)                      ▼                   │
  │   ─────────────────────                                          │
  │   Noisy mixture ──→ Encoder ──→ FiLM ──→ TCN ──→ Decoder ──→ 🔊│
  │   (any length)       (1D Conv)  (cond)  (mask)   (1D Conv)      │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
```

The system uses **Conv-TasNet** with **FiLM conditioning** — the speaker embedding tells the separator *which* voice to extract. Everything else (other speakers, car engines, street noise, music) is suppressed.

## ⚡ Key Features

| Feature | Details |
|---|---|
| **🔬 Tiny Model** | Conv-TasNet-Tiny: 1.3M params, ~5MB, real-time on M4 CPU |
| **🌍 Multilingual** | Arabic-focused + English + extensible to any language via HuggingFace |
| **🚗 Noise-Robust** | Trained with MUSAN, DEMAND (car/street/cafe), UrbanSound8K, RIR reverb |
| **🎯 Speaker-Conditioned** | FiLM conditioning on ECAPA-TDNN embeddings — extracts *only* the target |
| **📊 TensorBoard Audio** | Listen to mixture/target/estimated during training — hear your model learn! |
| **🤗 HuggingFace Native** | Stream any speech/noise dataset — configurable audio column names |
| **💾 Robust Checkpoints** | Atomic saves, best-loss + best-SI-SNR + periodic + last checkpoints |
| **⚙️ ONNX Export** | Export to ONNX for optimized CPU inference, with benchmark |

## 🏗️ Architecture

```
Model Configurations:
┌────────────┬──────────┬────────┬──────────────────────────┐
│ Config     │ Params   │ Size   │ Use Case                 │
├────────────┼──────────┼────────┼──────────────────────────┤
│ nano       │ 0.5M     │ ~2 MB  │ Edge / embedded          │
│ tiny ★     │ 1.3M     │ ~5 MB  │ M4 CPU real-time (rec.)  │
│ small      │ 2.5M     │ ~10 MB │ GPU inference            │
│ standard   │ 5.1M     │ ~20 MB │ Maximum quality          │
└────────────┴──────────┴────────┴──────────────────────────┘
```

## 🚀 Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
# Verify setup first
python test_setup.py

# Train (GPU recommended, uses HuggingFace datasets automatically)
python -m training.train --config configs/finetune.yaml

# Resume from checkpoint
python -m training.train --config configs/finetune.yaml --resume experiments/*/checkpoints/last.pt
```

### Monitor (TensorBoard)

```bash
# Launch TensorBoard — go to Audio tab to HEAR the model improve!
tensorboard --logdir experiments/
```

### Enroll & Extract

```bash
# Step 1: Create your voiceprint
python -m inference.enroll \
    --audio my_clip1.wav my_clip2.wav my_clip3.wav \
    --output my_voiceprint.npy

# Step 2: Extract your voice from any recording
python -m inference.extract \
    --mixture noisy_meeting.wav \
    --enrollment my_clip1.wav \
    --output my_voice_only.wav \
    --checkpoint experiments/*/checkpoints/best_loss.pt
```

### Export to ONNX (faster CPU inference)

```bash
python -m export.to_onnx \
    --checkpoint experiments/*/checkpoints/best_loss.pt \
    --output model.onnx
```

## 📁 Project Structure

```
your-denoizer/
├── configs/
│   └── finetune.yaml          # Training config (datasets, model, hyperparams)
├── models/
│   ├── separator.py           # Conv-TasNet + FiLM conditioning
│   ├── speaker_encoder.py     # ECAPA-TDNN wrapper (frozen)
│   └── tse_model.py           # High-level pipeline API
├── training/
│   ├── dataset.py             # HuggingFace-integrated on-the-fly mixer
│   ├── losses.py              # SI-SNR + Multi-Resolution STFT
│   ├── train.py               # Training loop + TensorBoard audio logging
│   └── evaluate.py            # SI-SNR, PESQ, STOI, speaker similarity
├── inference/
│   ├── extract.py             # CPU inference with overlap-add
│   └── enroll.py              # Speaker enrollment + verification
├── export/
│   └── to_onnx.py             # ONNX export + validation + benchmark
├── scripts/
│   └── discover_hf_datasets.py # HuggingFace dataset catalog
├── test_setup.py              # Smoke test (run first!)
├── colab_quickstart.py        # Colab copy-paste guide
└── requirements.txt
```

## 📊 Datasets

### Speech (configurable via YAML)

| Dataset | Language | Hours | Speakers | Status |
|---|---|---|---|---|
| Common Voice 17 | Arabic (MSA) | ~150+ | Many | ✅ Active |
| LibriSpeech clean-100 | English | 100 | 251 | ✅ Active |
| FLEURS | Arabic (Egyptian) | ~10 | — | 📋 Ready |
| FLEURS | Arabic (Saudi) | ~10 | — | 📋 Ready |
| Common Voice 17 | French | ~1000 | Many | 📋 Ready |
| Common Voice 17 | German | ~1200 | Many | 📋 Ready |
| Common Voice 17 | Chinese | ~200 | Many | 📋 Ready |
| VoxCeleb1 | Multi | ~350 | 1211 | 📋 Ready |
| VCTK | English | ~44 | 109 | 📋 Ready |

### Noise

| Dataset | Content | Status |
|---|---|---|
| **MUSAN** | Music, speech babble, technical noise | ✅ Active |
| **UrbanSound8K** | Car horns, sirens, engines, street music | 📋 Ready |
| **DEMAND** | 18 real environments (car, street, cafe, metro) | 📋 Ready (local) |
| **FSD50K** | 50K Freesound clips (everything) | 📋 Ready |
| **RIR** | Room impulse responses (reverb) | 📋 Ready (local) |

> **Adding a new dataset is 4 lines in the YAML config.** Run `python scripts/discover_hf_datasets.py` to see all available datasets with their column names.

## 🧪 Training Details

- **Loss**: Combined SI-SNR + Multi-Resolution STFT
- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.01)
- **Scheduler**: Cosine annealing with 3-epoch warmup
- **Mixed Precision**: FP16 on CUDA (~2x speedup)
- **Augmentation**: On-the-fly mixing with:
  - 1-2 interfering speakers at -5 to +5 dB SIR
  - Environmental noise at 5 to 20 dB SNR (car, street, babble)
  - Room reverb (RIR convolution)
  - Speed/pitch perturbation (0.9x-1.1x)

## 📈 TensorBoard

The training script logs:

| Tab | What You See |
|---|---|
| **Scalars** | Train/val loss, SI-SNR, learning rate, gradient norms |
| **Audio** 🔊 | Mixture → Target → Estimated (hear the model improve!) |
| **Graphs** | Model architecture visualization |
| **HParams** | Hyperparameter comparison across runs |

## 🔧 Experiment Output Structure

```
experiments/tse_multilingual_arabic_20260620_001200/
├── config.yaml              # Saved experiment config
├── metrics.json             # Full training metrics (for plotting)
├── checkpoints/
│   ├── last.pt              # Latest checkpoint (resume from here)
│   ├── best_loss.pt         # Best validation loss
│   ├── best_sisnr.pt        # Best validation SI-SNR
│   └── epoch_0010.pt        # Periodic checkpoints
├── logs/                    # TensorBoard logs
│   └── events.out.tfevents...
└── audio_samples/           # (reserved for audio exports)
```

## 🗺️ Roadmap

- [x] Conv-TasNet-Tiny with FiLM speaker conditioning
- [x] HuggingFace dataset pipeline (speech + noise, streaming)
- [x] TensorBoard audio logging
- [x] ONNX export for M4 CPU
- [ ] Phase 2: Personal voice cloning (F5-TTS)
- [ ] Phase 3: Per-user fine-tuning (3-5 min of your voice)
- [ ] Phase 4: CoreML export for Apple Neural Engine

## 📜 License

MIT

---

<div align="center">

*Built for scholarship-winning research.* 🎓

</div>
