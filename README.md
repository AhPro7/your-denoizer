# Your Denoizer 🎤

**Personalized voice isolation via target speaker extraction.**

A speaker-conditioned speech separation system that isolates a specific person's voice from multi-speaker audio. Give it a short enrollment clip of the target speaker, and it extracts only their voice — suppressing all other speakers, background noise, car sounds, street noise, and music.

## Architecture

```
Enrollment Audio (5-15s) → ECAPA-TDNN → Speaker Embedding (192-dim)
                                              ↓
Mixture Audio → Audio Encoder → [FiLM Conditioning] → TCN Separator → Audio Decoder → Isolated Voice
```

- **Conv-TasNet-Tiny** (~1.3M params) — lightweight, CPU-friendly, real-time on M4
- **ECAPA-TDNN** — pretrained speaker encoder (frozen), runs only at enrollment
- **FiLM Conditioning** — injects speaker identity into the separator
- **Multilingual** — trained on Arabic + English + more
- **Noise-robust** — trained with MUSAN, DEMAND (car/street), RIR reverb

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets
bash scripts/download_librispeech.sh
bash scripts/download_musan.sh
bash scripts/download_demand.sh

# Train
python -m training.train --config configs/finetune.yaml

# Inference (M4 CPU)
python -m inference.extract --mixture mixture.wav --enrollment enrollment.wav --output output.wav
```

## Project Structure

```
your-denoizer/
├── configs/           # Training & inference configs
├── models/            # Model architecture
├── training/          # Training loop, dataset, losses
├── inference/         # CPU inference + enrollment
├── export/            # ONNX / CoreML export
├── scripts/           # Data download scripts
└── checkpoints/       # Saved models
```

## License

MIT
