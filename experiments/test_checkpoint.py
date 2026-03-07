"""
Checkpoint verification test.

Tests:
1. Curriculum and magnitude state resolution at various steps
2. Round-trip: train 1000 steps → checkpoint → resume → train to 2000
3. Fresh 2000-step run for comparison
4. Val losses should match within noise (or bit-identical with RNG state)
"""

import math, os, sys, shutil, time

# Add experiments dir to path for checkpoint_utils import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkpoint_utils import get_curriculum_stage, should_magnitude_be_frozen

# --- Test 1: State resolution sanity checks ---

PROG_STAGES = [(0, 8), (667, 24), (1334, 64)]
MAG_FREE_STEP = 1000

def test_state_resolution():
    print("=" * 60)
    print("  Test 1: Curriculum & magnitude state resolution")
    print("=" * 60)

    cases = [
        (500,  8,  True,  "stage 1, mag frozen"),
        (800,  24, True,  "stage 2, mag frozen"),
        (1400, 64, False, "stage 3, mag unfrozen"),
        (0,    8,  True,  "step 0, stage 1, mag frozen"),
        (667,  24, True,  "stage 2 boundary, mag frozen"),
        (999,  24, True,  "just before mag unfreeze"),
        (1000, 24, False, "mag unfreeze boundary"),
        (1334, 64, False, "stage 3 boundary, mag free"),
        (1999, 64, False, "final step"),
    ]

    all_pass = True
    for step, expected_bands, expected_frozen, desc in cases:
        bands = get_curriculum_stage(step, PROG_STAGES)
        frozen = should_magnitude_be_frozen(step, MAG_FREE_STEP, True)
        bands_ok = bands == expected_bands
        frozen_ok = frozen == expected_frozen
        status = "PASS" if (bands_ok and frozen_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  step {step:>5} | bands={bands:>2} (exp {expected_bands:>2}) "
              f"| frozen={str(frozen):>5} (exp {str(expected_frozen):>5}) "
              f"| {status} — {desc}")

    print(f"\n  State resolution: {'ALL PASS' if all_pass else 'FAILED'}\n")
    return all_pass


# --- Test 2 & 3: Round-trip checkpoint test ---

def test_checkpoint_roundtrip():
    """Train 1000 steps with checkpoints, resume to 2000, compare with fresh 2000."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Import the actual training infrastructure
    from phaseC_integrated import (
        GPT, Dataset, download_shakespeare, estimate_loss,
        N_BANDS, N_EMBD, BLOCK_SIZE, BATCH_SIZE, LEARNING_RATE,
        MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS, DEVICE, MAESTRO_DIM,
        PROG_STAGES, MAG_FREE_STEP,
    )
    from checkpoint_utils import save_checkpoint, load_checkpoint, find_latest_checkpoint

    print("=" * 60)
    print("  Test 2: Checkpoint round-trip (maestro + curriculum)")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    text = download_shakespeare()
    dataset = Dataset(text)

    HALF = 1000  # checkpoint at this step
    FULL = MAX_ITERS  # 2000

    # --- Helper: run training for a range of steps ---
    def run_training(start_step, end_step, model, optimizer, dataset,
                     curriculum=True, two_stage=False, label=""):
        val_loss = float('inf')
        for i in range(start_step, end_step):
            # Curriculum
            if curriculum:
                for step_thresh, nb in PROG_STAGES:
                    if i >= step_thresh:
                        model.n_bands_active = nb
            else:
                model.n_bands_active = N_BANDS

            # Two-stage: zero magnitude gradients during frozen phase
            if i % EVAL_INTERVAL == 0 or i == end_step - 1:
                losses = estimate_loss(model, dataset)
                val_loss = losses['val']
                print(f"    [{label}] step {i:>5} | val {val_loss:.4f}")

            x, y = dataset.get_batch("train")
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()

            if two_stage and i < MAG_FREE_STEP:
                if model.mag.grad is not None:
                    model.mag.grad.zero_()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        return val_loss

    # --- Run A: Fresh 2000 steps (reference) ---
    print(f"\n  --- Run A: Fresh {FULL} steps ---")
    torch.manual_seed(42)
    model_a = GPT(dataset.vocab_size, mode="kerr", use_maestro=True, use_mag=False).to(DEVICE)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=LEARNING_RATE)
    val_a = run_training(0, FULL, model_a, opt_a, dataset,
                         curriculum=True, two_stage=False, label="fresh")

    # Final eval
    model_a.n_bands_active = N_BANDS
    final_a = estimate_loss(model_a, dataset)
    print(f"  Run A final val: {final_a['val']:.6f}")

    # --- Run B: 1000 steps → checkpoint → resume → 1000 more steps ---
    print(f"\n  --- Run B: {HALF} steps + checkpoint + resume + {HALF} more ---")
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '_test_checkpoints')
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    # Phase 1: train 0 to HALF
    torch.manual_seed(42)
    model_b = GPT(dataset.vocab_size, mode="kerr", use_maestro=True, use_mag=False).to(DEVICE)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=LEARNING_RATE)
    val_b1 = run_training(0, HALF, model_b, opt_b, dataset,
                          curriculum=True, two_stage=False, label="part1")

    # Save checkpoint at step HALF-1 (last completed step)
    ckpt_path = os.path.join(ckpt_dir, f'step_{HALF-1:06d}.pt')
    config = {'mode': 'kerr', 'curriculum': True, 'two_stage': False, 'use_maestro': True}
    save_checkpoint(ckpt_path, model_b, opt_b, HALF - 1, val_b1, config)

    # Phase 2: create fresh model+optimizer, load checkpoint, resume
    print(f"\n  --- Resuming from checkpoint at step {HALF-1} ---")
    model_b2 = GPT(dataset.vocab_size, mode="kerr", use_maestro=True, use_mag=False).to(DEVICE)
    opt_b2 = torch.optim.AdamW(model_b2.parameters(), lr=LEARNING_RATE)
    ckpt = load_checkpoint(ckpt_path, model_b2, opt_b2, device=DEVICE)
    resume_step = ckpt['step'] + 1
    print(f"  Resuming training from step {resume_step}")

    # Verify curriculum stage on resume
    active_bands = get_curriculum_stage(resume_step, PROG_STAGES)
    print(f"  Curriculum bands at resume: {active_bands}")

    val_b2 = run_training(resume_step, FULL, model_b2, opt_b2, dataset,
                          curriculum=True, two_stage=False, label="part2")

    # Final eval
    model_b2.n_bands_active = N_BANDS
    final_b = estimate_loss(model_b2, dataset)
    print(f"  Run B final val: {final_b['val']:.6f}")

    # --- Compare ---
    print(f"\n{'=' * 60}")
    print(f"  COMPARISON")
    print(f"{'=' * 60}")
    print(f"  Run A (fresh {FULL} steps):     val = {final_a['val']:.6f}")
    print(f"  Run B (checkpoint + resume):  val = {final_b['val']:.6f}")
    diff = abs(final_a['val'] - final_b['val'])
    pct_diff = diff / final_a['val'] * 100
    print(f"  Difference: {diff:.6f} ({pct_diff:.4f}%)")

    if diff < 0.001:
        print(f"  PASS — bit-identical or near-identical (diff < 0.001)")
        match = True
    elif pct_diff < 0.5:
        print(f"  PASS — within noise (< 0.5%)")
        match = True
    else:
        print(f"  FAIL — results diverged beyond noise threshold")
        match = False

    # Cleanup
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
        print(f"  Cleaned up {ckpt_dir}")

    return match


if __name__ == "__main__":
    t0 = time.time()

    pass1 = test_state_resolution()

    pass2 = test_checkpoint_roundtrip()

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  State resolution:      {'PASS' if pass1 else 'FAIL'}")
    print(f"  Checkpoint round-trip: {'PASS' if pass2 else 'FAIL'}")
    print(f"  Total time: {time.time() - t0:.1f}s")
    print(f"{'=' * 60}")

    sys.exit(0 if (pass1 and pass2) else 1)
