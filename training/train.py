"""
Training script for Target Speaker Extraction.

Supports:
    - CUDA GPU training (primary)
    - MPS (Apple Silicon) fallback
    - Mixed precision (FP16) for speed
    - TensorBoard logging
    - Checkpoint save/resume
    - Config-driven via YAML
    
Usage:
    python -m training.train --config configs/finetune.yaml
    python -m training.train --config configs/finetune.yaml --resume checkpoints/last.pt
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.separator import ConvTasNetTSE, build_model
from training.losses import SISnrLoss, CombinedLoss
from training.dataset import create_tse_dataloader


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Train] Using CUDA: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Train] Using Apple MPS (Metal)")
    else:
        device = torch.device("cpu")
        print("[Train] Using CPU")
    return device


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, 
                     best_loss, config, path):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'best_loss': best_loss,
        'config': config,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('step', 0), checkpoint.get('best_loss', float('inf'))


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler,
                     scaler, device, epoch, use_amp=True):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_si_snr = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
    
    for batch in pbar:
        mixture = batch['mixture'].to(device)        # (B, T)
        target = batch['target'].to(device)           # (B, T)
        enrollment = batch['enrollment'].to(device)   # (B, 192)
        
        optimizer.zero_grad()
        
        # Forward pass with optional mixed precision
        if use_amp and device.type == 'cuda':
            with autocast():
                estimated = model(mixture, enrollment)  # (B, 1, T)
                
                if isinstance(criterion, CombinedLoss):
                    losses = criterion(estimated, target)
                    loss = losses['loss']
                    si_snr_val = -losses['si_snr'].item()
                else:
                    loss = criterion(estimated, target)
                    si_snr_val = -loss.item()
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            estimated = model(mixture, enrollment)
            
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
        
        total_loss += loss.item()
        total_si_snr += si_snr_val
        num_batches += 1
        
        # Update progress bar
        avg_loss = total_loss / num_batches
        avg_si_snr = total_si_snr / num_batches
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'SI-SNR': f'{avg_si_snr:.2f} dB',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
        })
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'si_snr': total_si_snr / max(num_batches, 1),
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device, use_amp=True):
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    total_si_snr = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Validating", leave=False):
        mixture = batch['mixture'].to(device)
        target = batch['target'].to(device)
        enrollment = batch['enrollment'].to(device)
        
        if use_amp and device.type == 'cuda':
            with autocast():
                estimated = model(mixture, enrollment)
                if isinstance(criterion, CombinedLoss):
                    losses = criterion(estimated, target)
                    loss = losses['loss']
                    si_snr_val = -losses['si_snr'].item()
                else:
                    loss = criterion(estimated, target)
                    si_snr_val = -loss.item()
        else:
            estimated = model(mixture, enrollment)
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


def train(config_path: str, resume_path: str = None):
    """Main training loop."""
    
    # Load config
    config = load_config(config_path)
    train_cfg = config.get('training', {})
    model_cfg = config.get('model', {})
    
    # Setup
    device = get_device()
    use_amp = train_cfg.get('mixed_precision', True) and device.type == 'cuda'
    
    # Create output directory
    exp_name = config.get('experiment_name', 'tse_experiment')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(train_cfg.get('output_dir', 'checkpoints')) / f"{exp_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n{'='*60}")
    print(f"  Your Denoizer — Training")
    print(f"  Experiment: {exp_name}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {device}")
    print(f"  Mixed Precision: {use_amp}")
    print(f"{'='*60}\n")
    
    # ===== Build Model =====
    model_size = model_cfg.get('size', 'tiny')
    speaker_dim = model_cfg.get('speaker_dim', 192)
    
    model = build_model(config_name=model_size, speaker_dim=speaker_dim)
    model = model.to(device)
    
    trainable = count_parameters(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] Trainable: {trainable:,} / {total:,} parameters\n")
    
    # ===== Build DataLoaders =====
    print("[Data] Building training dataloader...")
    train_loader = create_tse_dataloader(config, speaker_encoder=None)
    
    val_config = config.copy()
    val_data = val_config.get('data', {}).copy()
    val_data['samples_per_epoch'] = val_data.get('val_samples', 1000)
    val_data['batch_size'] = val_data.get('batch_size', 16)
    val_config['data'] = val_data
    
    print("[Data] Building validation dataloader...")
    val_loader = create_tse_dataloader(val_config, speaker_encoder=None)
    
    # ===== Loss, Optimizer, Scheduler =====
    loss_type = train_cfg.get('loss', 'si_snr')
    if loss_type == 'combined':
        criterion = CombinedLoss(
            si_snr_weight=train_cfg.get('si_snr_weight', 1.0),
            stft_weight=train_cfg.get('stft_weight', 0.1),
        )
    else:
        criterion = SISnrLoss()
    
    lr = float(train_cfg.get('learning_rate', 3e-4))
    weight_decay = float(train_cfg.get('weight_decay', 0.01))
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    # Cosine annealing with warmup
    num_epochs = train_cfg.get('epochs', 50)
    warmup_epochs = train_cfg.get('warmup_epochs', 3)
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    import math
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scaler = GradScaler() if use_amp else None
    
    # ===== Resume from checkpoint =====
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')
    
    if resume_path and Path(resume_path).exists():
        print(f"[Resume] Loading checkpoint: {resume_path}")
        start_epoch, global_step, best_loss = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler
        )
        print(f"[Resume] Epoch {start_epoch}, Step {global_step}, Best Loss: {best_loss:.4f}")
    
    # ===== TensorBoard =====
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(output_dir / 'tb_logs')
    except ImportError:
        writer = None
        print("[Train] TensorBoard not available, logging to console only")
    
    # ===== Training Loop =====
    patience = train_cfg.get('patience', 10)
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"  Starting training: {num_epochs} epochs")
    print(f"  Steps/epoch: {steps_per_epoch}")
    print(f"  Warmup: {warmup_epochs} epochs ({warmup_steps} steps)")
    print(f"  Early stopping patience: {patience}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch + 1, use_amp
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, use_amp)
        
        epoch_time = time.time() - epoch_start
        
        # Logging
        print(f"\n  Epoch {epoch+1}/{num_epochs} ({epoch_time:.0f}s)")
        print(f"  Train — Loss: {train_metrics['loss']:.4f}, SI-SNR: {train_metrics['si_snr']:.2f} dB")
        print(f"  Val   — Loss: {val_metrics['loss']:.4f}, SI-SNR: {val_metrics['si_snr']:.2f} dB")
        
        if writer:
            writer.add_scalar('train/loss', train_metrics['loss'], epoch)
            writer.add_scalar('train/si_snr', train_metrics['si_snr'], epoch)
            writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            writer.add_scalar('val/si_snr', val_metrics['si_snr'], epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoints
        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch + 1, global_step,
            best_loss, config, output_dir / 'last.pt'
        )
        
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch + 1, global_step,
                best_loss, config, output_dir / 'best.pt'
            )
            print(f"  ★ New best model saved! (loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
        
        # Save periodic checkpoints
        if (epoch + 1) % train_cfg.get('save_every', 10) == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch + 1, global_step,
                best_loss, config, output_dir / f'epoch_{epoch+1}.pt'
            )
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
        
        print()
    
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best validation loss: {best_loss:.4f}")
    print(f"  Checkpoints saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    if writer:
        writer.close()
    
    return model, output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Target Speaker Extraction model')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    train(args.config, args.resume)
