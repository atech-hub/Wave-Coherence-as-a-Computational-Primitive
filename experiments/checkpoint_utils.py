"""
Checkpoint utilities for Wave Coherence training experiments.

Supports:
- Saving model + optimizer + training state at intervals
- Resuming from checkpoint with correct curriculum stage and magnitude freeze state
- Automatic detection of latest checkpoint in a directory
- RNG state preservation for bit-identical reproducibility
"""

import os
import glob
import torch


def save_checkpoint(path, model, optimizer, step, val_loss, config):
    """
    Save a training checkpoint including RNG state for reproducibility.

    Args:
        path: filepath to save (e.g., 'checkpoints/step_1000.pt')
        model: the nn.Module
        optimizer: the optimizer
        step: current training step (int)
        val_loss: most recent validation loss (float)
        config: dict of training configuration (band count, curriculum stages,
                maestro settings, two-stage settings, etc.)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'step': step,
        'val_loss': val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'rng_state': torch.random.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }, path)
    print(f"  Checkpoint saved: {path} (step {step}, val {val_loss:.4f})")


def load_checkpoint(path, model, optimizer, device='cuda'):
    """
    Load a training checkpoint and restore all state including RNG.

    Args:
        path: filepath to load
        model: the nn.Module (must have same architecture as saved)
        optimizer: the optimizer (must have same param groups as saved)
        device: device to map tensors to

    Returns:
        dict with 'step', 'val_loss', 'config'

    Note: weights_only=False is required because optimizer state dicts
    contain non-tensor objects (param group metadata).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore RNG state for bit-identical reproducibility
    # RNG states must be on CPU as ByteTensors regardless of map_location
    if 'rng_state' in checkpoint:
        torch.random.set_rng_state(checkpoint['rng_state'].cpu())
    if 'cuda_rng_state' in checkpoint and checkpoint['cuda_rng_state'] is not None:
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())

    print(f"  Checkpoint loaded: {path} (step {checkpoint['step']}, val {checkpoint['val_loss']:.4f})")
    return {
        'step': checkpoint['step'],
        'val_loss': checkpoint['val_loss'],
        'config': checkpoint.get('config', {}),
    }


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the most recent checkpoint in a directory.

    Expects filenames like 'step_000000.pt', 'step_000200.pt', etc.
    Returns None if no checkpoints found.
    """
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'step_*.pt')))
    if not checkpoints:
        return None
    return checkpoints[-1]


def get_curriculum_stage(step, prog_stages):
    """
    Determine which curriculum stage is active at a given step.

    Args:
        step: current training step
        prog_stages: list of (start_step, n_bands) tuples

    Returns:
        n_bands: number of active bands at this step
    """
    active_bands = prog_stages[0][1]
    for start_step, n_bands in prog_stages:
        if step >= start_step:
            active_bands = n_bands
    return active_bands


def should_magnitude_be_frozen(step, mag_free_step, two_stage_enabled):
    """
    Determine whether magnitude parameters should be frozen at a given step.

    Args:
        step: current training step
        mag_free_step: step at which magnitude unfreezes
        two_stage_enabled: whether two-stage training is active

    Returns:
        bool: True if magnitude should be frozen
    """
    if not two_stage_enabled:
        return False
    return step < mag_free_step
