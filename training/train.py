"""
Training script for Target Speaker Extraction.

Supports:
    - CUDA GPU training (primary)
    - MPS (Apple Silicon) fallback
    - Mixed precision (FP16) for speed
    - TensorBoard logging with AUDIO samples (hear your model improve!)
    - Robust checkpoint save/resume (last, best, periodic)
    - Config-driven via YAML
    - Automatic experiment directory structure
    
Usage:
    python -m training.train --config configs/finetune.yaml
    python -m training.train --config configs/finetune.yaml --resume checkpoints/last.pt
"""

import os
import sys
import time
import math
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.separator import ConvTasNetTSE, build_model
from training.losses import SISnrLoss, CombinedLoss
from training.dataset import create_tse_dataloader


# ============================================================================
# Device & Config
# ============================================================================

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name()
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[Train] Using CUDA: {gpu_name} ({vram:.1f} GB)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Train] Using Apple MPS (Metal)")
    else:
        device = torch.device("cpu")
        print("[Train] Using CPU (⚠️ training will be slow)")
    return device


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def count_parameters(model: nn.Module) -> dict:
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {'trainable': trainable, 'total': total}


# ============================================================================
# Checkpointing (robust)
# ============================================================================

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step,
                     best_loss, best_si_snr, config, metrics_history, path):
    """Save a comprehensive training checkpoint."""
    checkpoint = {
        # Training state
        'epoch': epoch,
        'global_step': global_step,
        'best_loss': best_loss,
        'best_si_snr': best_si_snr,
        # Model + optimizer
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        # Config & history
        'config': config,
        'metrics_history': metrics_history,
        # Metadata
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
    }
    
    # Atomic save: write to tmp then rename (prevents corruption on crash)
    tmp_path = str(path) + '.tmp'
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, str(path))


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    """Load training checkpoint.
    
    Returns:
        (epoch, global_step, best_loss, best_si_snr, metrics_history)
    """
    print(f"[Resume] Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"[Resume] Warning: Could not load optimizer state: {e}")
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            print(f"[Resume] Warning: Could not load scheduler state: {e}")
    
    if scaler and checkpoint.get('scaler_state_dict'):
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        except Exception as e:
            print(f"[Resume] Warning: Could not load scaler state: {e}")
    
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    best_loss = checkpoint.get('best_loss', float('inf'))
    best_si_snr = checkpoint.get('best_si_snr', float('-inf'))
    metrics_history = checkpoint.get('metrics_history', [])
    
    print(f"[Resume] Epoch {epoch}, Step {global_step}, "
          f"Best Loss: {best_loss:.4f}, Best SI-SNR: {best_si_snr:.2f} dB")
    
    return epoch, global_step, best_loss, best_si_snr, metrics_history


# ============================================================================
# TensorBoard Audio Logging
# ============================================================================

def log_audio_samples(writer, model, val_loader, device, epoch, 
                       sample_rate=16000, num_samples=4, use_amp=True):
    """Log audio samples to TensorBoard so you can HEAR the model improve.
    
    For each sample, logs:
        - mixture (input)
        - target (ground truth)
        - estimated (model output)
    
    Listen to these in TensorBoard → Audio tab to track progress!
    """
    if writer is None:
        return
    
    model.eval()
    logged = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if logged >= num_samples:
                break
            
            mixture = batch['noisy'].to(device)
            target = batch['clean'].to(device)
            
            if use_amp and device.type == 'cuda':
                from torch.cuda.amp import autocast
                with autocast():
                    estimated = model(mixture)
            else:
                estimated = model(mixture)
            
            # Log each sample in the batch
            batch_size = min(mixture.shape[0], num_samples - logged)
            for i in range(batch_size):
                mix_wav = mixture[i].cpu().float()
                tgt_wav = target[i].cpu().float()
                est_wav = estimated[i].squeeze().cpu().float()
                
                # Normalize audio for TensorBoard playback
                def safe_normalize(wav):
                    max_val = wav.abs().max()
                    if max_val > 0:
                        wav = wav / max_val * 0.9
                    return wav.unsqueeze(0)  # (1, T) for TensorBoard
                
                mix_wav = safe_normalize(mix_wav)
                tgt_wav = safe_normalize(tgt_wav)
                est_wav = safe_normalize(est_wav)
                
                tag = f"audio/sample_{logged}"
                writer.add_audio(f"{tag}/1_mixture", mix_wav, epoch, sample_rate)
                writer.add_audio(f"{tag}/2_target_clean", tgt_wav, epoch, sample_rate)
                writer.add_audio(f"{tag}/3_estimated", est_wav, epoch, sample_rate)
                
                logged += 1
    
    model.train()


def log_gradient_stats(writer, model, global_step):
    """Log gradient statistics to TensorBoard for debugging."""
    if writer is None:
        return
    
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            # Log per-layer gradient norms (key layers only)
            if any(k in name for k in ['encoder.weight', 'decoder.weight', 
                                         'film.scale.weight', 'mask_proj']):
                writer.add_scalar(f'gradients/{name}', param_norm, global_step)
    
    total_norm = total_norm ** 0.5
    writer.add_scalar('gradients/total_norm', total_norm, global_step)


# ============================================================================
# Training & Validation
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler,
                     scaler, device, epoch, global_step, use_amp=True,
                     writer=None, log_interval=50):
    """Train for one epoch with detailed logging."""
    model.train()
    
    total_loss = 0
    total_si_snr = 0
    num_batches = 0
    epoch_start = time.time()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
    
    for batch_idx, batch in enumerate(pbar):
        mixture = batch['noisy'].to(device)        # (B, T)
        target = batch['clean'].to(device)           # (B, T)
        
        optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
        
        # Forward pass with optional mixed precision
        if use_amp and device.type == 'cuda':
            from torch.cuda.amp import autocast
            with autocast():
                estimated = model(mixture)
                
                if isinstance(criterion, CombinedLoss):
                    losses = criterion(estimated, target)
                    loss = losses['loss']
                    si_snr_val = -losses['si_snr'].item()
                else:
                    loss = criterion(estimated, target)
                    si_snr_val = -loss.item()
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            estimated = model(mixture)
            
            if isinstance(criterion, CombinedLoss):
                losses = criterion(estimated, target)
                loss = losses['loss']
                si_snr_val = -losses['si_snr'].item()
            else:
                loss = criterion(estimated, target)
                si_snr_val = -loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        global_step += 1
        total_loss += loss.item()
        total_si_snr += si_snr_val
        num_batches += 1
        
        # Per-step TensorBoard logging
        if writer and global_step % log_interval == 0:
            writer.add_scalar('train_step/loss', loss.item(), global_step)
            writer.add_scalar('train_step/si_snr', si_snr_val, global_step)
            writer.add_scalar('train_step/lr', optimizer.param_groups[0]['lr'], global_step)
            
            # Log gradient stats periodically
            if global_step % (log_interval * 5) == 0:
                log_gradient_stats(writer, model, global_step)
        
        # Update progress bar
        avg_loss = total_loss / num_batches
        avg_si_snr = total_si_snr / num_batches
        elapsed = time.time() - epoch_start
        samples_per_sec = (num_batches * mixture.shape[0]) / elapsed
        
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'SI-SNR': f'{avg_si_snr:.2f}dB',
            'lr': f'{optimizer.param_groups[0]["lr"]:.1e}',
            'samp/s': f'{samples_per_sec:.0f}',
        })
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'si_snr': total_si_snr / max(num_batches, 1),
    }, global_step


@torch.no_grad()
def validate(model, dataloader, criterion, device, use_amp=True):
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    total_si_snr = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Validating", leave=False):
        mixture = batch['noisy'].to(device)
        target = batch['clean'].to(device)
        
        if use_amp and device.type == 'cuda':
            from torch.cuda.amp import autocast
            with autocast():
                estimated = model(mixture)
                if isinstance(criterion, CombinedLoss):
                    losses = criterion(estimated, target)
                    loss = losses['loss']
                    si_snr_val = -losses['si_snr'].item()
                else:
                    loss = criterion(estimated, target)
                    si_snr_val = -loss.item()
        else:
            estimated = model(mixture)
            if isinstance(criterion, CombinedLoss):
                losses = criterion(estimated, target)
                loss = losses['loss']
                si_snr_val = -losses['si_snr'].item()
            else:
                loss = criterion(estimated, target)
                si_snr_val = -loss.item()
        
        total_loss += loss.item()
        total_si_snr += si_snr_val
        num_batches += 1
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'si_snr': total_si_snr / max(num_batches, 1),
    }


# ============================================================================
# Main Training Loop
# ============================================================================

def train(config_path: str, resume_path: str = None):
    """Main training loop with full TensorBoard logging and checkpointing."""
    
    # Load config
    config = load_config(config_path)
    train_cfg = config.get('training', {})
    model_cfg = config.get('model', {})
    
    # Setup
    device = get_device()
    use_amp = train_cfg.get('mixed_precision', True) and device.type == 'cuda'
    
    # Create experiment directory structure
    exp_name = config.get('experiment_name', 'tse_experiment')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(train_cfg.get('output_dir', 'checkpoints')) / f"{exp_name}_{timestamp}"
    
    # Subdirectories
    ckpt_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    audio_dir = output_dir / 'audio_samples'
    
    for d in [ckpt_dir, log_dir, audio_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Save config to experiment dir
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n{'='*60}")
    print(f"  🎤 Your Denoizer — Training")
    print(f"{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Output     : {output_dir}")
    print(f"  Device     : {device}")
    print(f"  Mixed Prec : {use_amp}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  TensorBoard: {log_dir}")
    print(f"{'='*60}\n")
    
    # ===== Build Model =====
    model_size = model_cfg.get('size', 'tiny')
    speaker_dim = model_cfg.get('speaker_dim', 0)
    
    model = build_model(config_name=model_size, speaker_dim=speaker_dim)
    model = model.to(device)
    
    params = count_parameters(model)
    print(f"\n[Model] {model_size.upper()} — "
          f"Trainable: {params['trainable']:,} / Total: {params['total']:,}\n")
    
    # ===== Build DataLoaders =====
    print("[Data] Building training dataloader...")
    train_loader = create_tse_dataloader(config, speaker_encoder=None)
    
    val_config = config.copy()
    val_data = val_config.get('data', {}).copy()
    val_data['samples_per_epoch'] = val_data.get('val_samples', 1000)
    val_config['data'] = val_data
    
    print("[Data] Building validation dataloader...")
    val_loader = create_tse_dataloader(val_config, speaker_encoder=None)
    
    # ===== Loss =====
    loss_type = train_cfg.get('loss', 'si_snr')
    if loss_type == 'combined':
        criterion = CombinedLoss(
            si_snr_weight=train_cfg.get('si_snr_weight', 1.0),
            stft_weight=train_cfg.get('stft_weight', 0.1),
        )
        print(f"[Loss] Combined (SI-SNR × {train_cfg.get('si_snr_weight', 1.0)} + "
              f"STFT × {train_cfg.get('stft_weight', 0.1)})")
    else:
        criterion = SISnrLoss()
        print(f"[Loss] SI-SNR")
    
    # ===== Optimizer =====
    lr = float(train_cfg.get('learning_rate', 3e-4))
    weight_decay = float(train_cfg.get('weight_decay', 0.01))
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # ===== Scheduler (cosine with warmup) =====
    num_epochs = train_cfg.get('epochs', 50)
    warmup_epochs = train_cfg.get('warmup_epochs', 3)
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))  # min LR = 1% of peak
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # ===== Mixed Precision Scaler =====
    scaler = None
    if use_amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    
    # ===== Resume =====
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')
    best_si_snr = float('-inf')
    metrics_history = []
    
    if resume_path and Path(resume_path).exists():
        start_epoch, global_step, best_loss, best_si_snr, metrics_history = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler
        )
    
    # ===== TensorBoard =====
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(log_dir))
        
        # Log model graph
        try:
            dummy_mix = torch.randn(1, 1, 64000).to(device)
            if speaker_dim > 0:
                dummy_emb = torch.randn(1, speaker_dim).to(device)
                writer.add_graph(model, (dummy_mix, dummy_emb))
            else:
                writer.add_graph(model, dummy_mix)
        except Exception:
            pass  # Graph logging may fail for complex models
        
        # Log hyperparameters
        hparams = {
            'model_size': model_size,
            'lr': lr,
            'batch_size': config.get('data', {}).get('batch_size', 16),
            'weight_decay': weight_decay,
            'loss_type': loss_type,
            'num_interferers': config.get('data', {}).get('num_interferers', 2),
            'noise_prob': config.get('data', {}).get('noise_prob', 0.7),
        }
        writer.add_text('config/hyperparameters', 
                        '\n'.join(f'**{k}**: {v}' for k, v in hparams.items()), 0)
        
        print(f"[TensorBoard] Logging to {log_dir}")
        print(f"[TensorBoard] Run: tensorboard --logdir {log_dir}")
        
    except ImportError:
        print("[TensorBoard] Not available — install with: pip install tensorboard")
    
    # ===== Audio sample logging config =====
    audio_log_interval = train_cfg.get('audio_log_every', 5)  # Log audio every N epochs
    num_audio_samples = train_cfg.get('num_audio_samples', 4)  # Number of samples to log
    
    # ===== Training Loop =====
    patience = train_cfg.get('patience', 10)
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"  🚀 Starting training")
    print(f"  Epochs        : {start_epoch + 1} → {num_epochs}")
    print(f"  Steps/epoch   : {steps_per_epoch}")
    print(f"  Total steps   : {total_steps:,}")
    print(f"  Warmup        : {warmup_epochs} epochs ({warmup_steps} steps)")
    print(f"  Early stop    : {patience} epochs patience")
    print(f"  Audio logging : every {audio_log_interval} epochs ({num_audio_samples} samples)")
    print(f"{'='*60}\n")
    
    training_start = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        # ---- Train ----
        train_metrics, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch + 1, global_step, use_amp,
            writer=writer, log_interval=train_cfg.get('log_interval', 50)
        )
        
        # ---- Validate ----
        val_metrics = validate(model, val_loader, criterion, device, use_amp)
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - training_start
        
        # ---- Epoch metrics ----
        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_si_snr': train_metrics['si_snr'],
            'val_loss': val_metrics['loss'],
            'val_si_snr': val_metrics['si_snr'],
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time,
            'global_step': global_step,
        }
        metrics_history.append(epoch_record)
        
        # ---- Console logging ----
        print(f"\n  ┌─ Epoch {epoch+1}/{num_epochs} ({epoch_time:.0f}s, total {total_time/60:.0f}min)")
        print(f"  │  Train — Loss: {train_metrics['loss']:.4f}  SI-SNR: {train_metrics['si_snr']:.2f} dB")
        print(f"  │  Val   — Loss: {val_metrics['loss']:.4f}  SI-SNR: {val_metrics['si_snr']:.2f} dB")
        print(f"  │  LR: {optimizer.param_groups[0]['lr']:.2e}  Step: {global_step}")
        
        # ---- TensorBoard epoch logging ----
        if writer:
            writer.add_scalars('loss', {
                'train': train_metrics['loss'],
                'val': val_metrics['loss'],
            }, epoch + 1)
            
            writer.add_scalars('si_snr', {
                'train': train_metrics['si_snr'],
                'val': val_metrics['si_snr'],
            }, epoch + 1)
            
            writer.add_scalar('lr/learning_rate', optimizer.param_groups[0]['lr'], epoch + 1)
            writer.add_scalar('time/epoch_seconds', epoch_time, epoch + 1)
            writer.add_scalar('time/total_minutes', total_time / 60, epoch + 1)
        
        # ---- Audio sample logging (hear the model improve!) ----
        if (epoch + 1) % audio_log_interval == 0 or epoch == 0:
            print(f"  │  🔊 Logging {num_audio_samples} audio samples to TensorBoard...")
            log_audio_samples(
                writer, model, val_loader, device, epoch + 1,
                sample_rate=config.get('data', {}).get('sample_rate', 16000),
                num_samples=num_audio_samples, use_amp=use_amp
            )
        
        # ---- Checkpointing ----
        ckpt_args = (model, optimizer, scheduler, scaler, epoch + 1, global_step,
                     best_loss, best_si_snr, config, metrics_history)
        
        # Always save last
        save_checkpoint(*ckpt_args, ckpt_dir / 'last.pt')
        
        # Save best (by loss)
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            patience_counter = 0
            save_checkpoint(*ckpt_args, ckpt_dir / 'best_loss.pt')
            print(f"  │  ⭐ New best loss: {best_loss:.4f}")
        else:
            patience_counter += 1
        
        # Save best (by SI-SNR)
        if val_metrics['si_snr'] > best_si_snr:
            best_si_snr = val_metrics['si_snr']
            save_checkpoint(*ckpt_args, ckpt_dir / 'best_sisnr.pt')
            print(f"  │  ⭐ New best SI-SNR: {best_si_snr:.2f} dB")
        
        # Periodic checkpoint
        save_every = train_cfg.get('save_every', 10)
        if (epoch + 1) % save_every == 0:
            save_checkpoint(*ckpt_args, ckpt_dir / f'epoch_{epoch+1:04d}.pt')
            print(f"  │  💾 Periodic checkpoint saved (epoch {epoch+1})")
        
        # Log patience
        if patience_counter > 0:
            print(f"  │  ⏳ No improvement ({patience_counter}/{patience})")
        
        print(f"  └─")
        
        # Save metrics history as JSON (for easy plotting)
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_history, f, indent=2)
        
        # ---- Early stopping ----
        if patience_counter >= patience:
            print(f"\n  🛑 Early stopping at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs)")
            break
        
        print()
    
    # ===== Training Complete =====
    total_time = time.time() - training_start
    
    print(f"\n{'='*60}")
    print(f"  ✅ Training complete!")
    print(f"  Total time      : {total_time/60:.1f} minutes")
    print(f"  Best val loss   : {best_loss:.4f}")
    print(f"  Best val SI-SNR : {best_si_snr:.2f} dB")
    print(f"  Checkpoints     : {ckpt_dir}")
    print(f"  TensorBoard     : tensorboard --logdir {log_dir}")
    print(f"  Metrics JSON    : {output_dir / 'metrics.json'}")
    print(f"{'='*60}\n")
    
    if writer:
        # Log final metrics
        writer.add_hparams(
            hparam_dict={
                'model_size': model_size,
                'lr': lr,
                'batch_size': config.get('data', {}).get('batch_size', 16),
            },
            metric_dict={
                'best_val_loss': best_loss,
                'best_val_si_snr': best_si_snr,
            }
        )
        writer.close()
    
    return model, output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Target Speaker Extraction model')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    train(args.config, args.resume)
