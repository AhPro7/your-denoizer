# ============================================================
# Colab Quick Start — Your Denoizer
# ============================================================
# Copy-paste these cells into a Colab notebook.
# Make sure to set Runtime → GPU (T4 is fine).
# ============================================================

# ---- Cell 1: Clone repo & install ----
# !git clone https://github.com/YOUR_USERNAME/your-denoizer.git
# %cd your-denoizer
# OR upload the folder to Colab / Google Drive

# !pip install torch torchaudio speechbrain soundfile librosa
# !pip install datasets huggingface-hub
# !pip install pesq pystoi tensorboard
# !pip install onnx onnxruntime pyyaml tqdm

# ---- Cell 2: Verify setup ----
# !python test_setup.py

# ---- Cell 3: Train with HuggingFace datasets ----
# Simplest: just use the config file
# !python -m training.train --config configs/finetune.yaml

# ---- Cell 4: Or train with custom config inline ----
"""
import yaml
from training.train import train

# You can modify the config programmatically:
config = yaml.safe_load(open('configs/finetune.yaml'))

# Add more Arabic datasets
config['data']['hf_speech'].append({
    'name': 'google/fleurs',
    'subset': 'ar_eg',       # Egyptian Arabic
    'audio_column': 'audio',
    'speaker_column': 'id',
    'streaming': True,
})

# Add noise from HuggingFace
config['data']['hf_noise'] = [{
    'name': 'flozi00/MUSAN-Noise',
    'audio_column': 'audio',
    'category': 'noise',
    'streaming': True,
}]

# Adjust for Colab T4
config['data']['batch_size'] = 16
config['training']['epochs'] = 50
config['training']['mixed_precision'] = True

# Save and train
yaml.dump(config, open('configs/my_config.yaml', 'w'))
# Then run: !python -m training.train --config configs/my_config.yaml
"""

# ---- Cell 5: Export to ONNX ----
# !python -m export.to_onnx --checkpoint checkpoints/tse_multilingual_arabic_*/best.pt --output model.onnx

# ---- Cell 6: Test inference ----
# !python -m inference.extract --mixture test_mixture.wav --enrollment test_enrollment.wav --output result.wav --checkpoint checkpoints/tse_multilingual_arabic_*/best.pt
