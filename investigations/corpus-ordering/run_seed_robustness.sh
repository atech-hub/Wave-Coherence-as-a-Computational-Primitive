#!/bin/bash
# Seed robustness test for corpus ordering hypothesis
# 4 configurations × 5 seeds = 30 training sessions
# Parallelized: up to 10 concurrent runs (4 threads each)
# Expected time: ~25 minutes (vs ~2 hours sequential)

set -e

SEEDS=(42 100 200 300 400)
SHAK="data/input.txt"
CHILD="data/children.txt"
ITERS=3000
RESULTS_DIR="seed_robustness"
EXE="target/release/kerr-engine.exe"
BASE_DIR="$(pwd)"

mkdir -p "$RESULTS_DIR"

echo "=== Seed Robustness Test: Corpus Ordering ==="
echo "Seeds: ${SEEDS[*]}"
echo "Parallel: up to 10 concurrent (4 threads each)"
echo "Start: $(date)"
echo ""

# Helper: run training in isolated temp dir, copy results out
run_train() {
    local label="$1"
    shift
    local tmpdir
    tmpdir=$(mktemp -d)
    (
        cd "$tmpdir"
        "$BASE_DIR/$EXE" "$@" > log.txt 2>&1
        cp training_summary.json "$BASE_DIR/$RESULTS_DIR/${label}.json"
        # Copy checkpoint if it exists (needed for phase 2)
        if [ -f checkpoint_final.bin ]; then
            cp checkpoint_final.bin "$BASE_DIR/$RESULTS_DIR/${label}_ckpt.bin"
        fi
    )
    rm -rf "$tmpdir"
}

# Phase 1: All standalone runs in parallel (10 runs)
echo "--- Phase 1: Standalone runs (10 jobs) ---"
PIDS=()
for SEED in "${SEEDS[@]}"; do
    run_train "seed${SEED}_shak_only" train "$BASE_DIR/$SHAK" $ITERS 4 64 3e-4 --seed "$SEED" &
    PIDS+=($!)
    run_train "seed${SEED}_child_only" train "$BASE_DIR/$CHILD" $ITERS 4 64 3e-4 --seed "$SEED" &
    PIDS+=($!)
done

echo "  Launched ${#PIDS[@]} parallel jobs, waiting..."
FAIL=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAIL=$((FAIL+1))
done
echo "  Phase 1 done ($(date)). Failures: $FAIL"
echo ""

# Phase 2: Sequential runs — need phase 1 checkpoints
echo "--- Phase 2: Sequential runs (10 jobs) ---"
PIDS=()
for SEED in "${SEEDS[@]}"; do
    # Children → Shakespeare (resume child checkpoint on shakespeare)
    run_train "seed${SEED}_child_then_shak" train "$BASE_DIR/$SHAK" $((ITERS * 2)) 4 64 3e-4 --seed "$SEED" \
        --resume "$BASE_DIR/$RESULTS_DIR/seed${SEED}_child_only_ckpt.bin" &
    PIDS+=($!)

    # Shakespeare → Children (resume shak checkpoint on children)
    run_train "seed${SEED}_shak_then_child" train "$BASE_DIR/$CHILD" $((ITERS * 2)) 4 64 3e-4 --seed "$SEED" \
        --resume "$BASE_DIR/$RESULTS_DIR/seed${SEED}_shak_only_ckpt.bin" &
    PIDS+=($!)
done

echo "  Launched ${#PIDS[@]} parallel jobs, waiting..."
FAIL=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAIL=$((FAIL+1))
done
echo "  Phase 2 done ($(date)). Failures: $FAIL"
echo ""

# Summary table
echo "=== RESULTS SUMMARY ==="
echo ""
printf "%-6s  %-12s  %-12s  %-16s  %-16s\n" "Seed" "Shak Only" "Child Only" "Child->Shak" "Shak->Child"
printf "%-6s  %-12s  %-12s  %-16s  %-16s\n" "----" "---------" "----------" "-----------" "-----------"

for SEED in "${SEEDS[@]}"; do
    SHAK_LOSS=$(grep -o '"final_val_loss": [0-9.]*' "$RESULTS_DIR/seed${SEED}_shak_only.json" | grep -o '[0-9.]*$')
    CHILD_LOSS=$(grep -o '"final_val_loss": [0-9.]*' "$RESULTS_DIR/seed${SEED}_child_only.json" | grep -o '[0-9.]*$')
    C2S_LOSS=$(grep -o '"final_val_loss": [0-9.]*' "$RESULTS_DIR/seed${SEED}_child_then_shak.json" | grep -o '[0-9.]*$')
    S2C_LOSS=$(grep -o '"final_val_loss": [0-9.]*' "$RESULTS_DIR/seed${SEED}_shak_then_child.json" | grep -o '[0-9.]*$')
    printf "%-6s  %-12s  %-12s  %-16s  %-16s\n" "$SEED" "$SHAK_LOSS" "$CHILD_LOSS" "$C2S_LOSS" "$S2C_LOSS"
done

# Winner per seed
echo ""
echo "--- Winner per seed (lowest val_loss) ---"
for SEED in "${SEEDS[@]}"; do
    SHAK_LOSS=$(grep -o '"final_val_loss": [0-9.]*' "$RESULTS_DIR/seed${SEED}_shak_only.json" | grep -o '[0-9.]*$')
    CHILD_LOSS=$(grep -o '"final_val_loss": [0-9.]*' "$RESULTS_DIR/seed${SEED}_child_only.json" | grep -o '[0-9.]*$')
    C2S_LOSS=$(grep -o '"final_val_loss": [0-9.]*' "$RESULTS_DIR/seed${SEED}_child_then_shak.json" | grep -o '[0-9.]*$')
    S2C_LOSS=$(grep -o '"final_val_loss": [0-9.]*' "$RESULTS_DIR/seed${SEED}_shak_then_child.json" | grep -o '[0-9.]*$')

    BEST="shak_only"
    BEST_VAL="$SHAK_LOSS"
    for PAIR in "child_only:$CHILD_LOSS" "child->shak:$C2S_LOSS" "shak->child:$S2C_LOSS"; do
        NAME="${PAIR%%:*}"
        VAL="${PAIR#*:}"
        if (( $(echo "$VAL < $BEST_VAL" | bc -l) )); then
            BEST="$NAME"
            BEST_VAL="$VAL"
        fi
    done
    echo "  Seed $SEED: $BEST ($BEST_VAL)"
done

echo ""
echo "Finish: $(date)"
echo ""
echo "Verdict:"
echo "  5/5 shak->child wins = ROBUST (publish)"
echo "  4/5 = STRONG (publish with caveat)"
echo "  3/5 = WEAK (partial, note seed sensitivity)"
echo "  2/5 or less = NOISE (do not publish ordering claim)"
