#!/bin/bash
# Test 2: Legal as extreme case
# Does Legal→Children perform worse than Shakespeare→Children?
# If yes: "hard packaging builds capacity" not "any constraint works"
#
# 3 seeds × 3 configs = 9 baseline + 6 sequential = 15 sessions
# Parallelized, ~15 minutes

set -e

SEEDS=(42 100 200)
SHAK="data/input.txt"
CHILD="data/children.txt"
LEGAL="data/legal.txt"
ITERS=3000
RESULTS_DIR="test2_legal"
EXE="target/release/kerr-engine.exe"
BASE_DIR="$(pwd)"

mkdir -p "$RESULTS_DIR"

echo "=== Test 2: Legal as Extreme Case ==="
echo "Seeds: ${SEEDS[*]}"
echo "Hypothesis: Shak->Child > Legal->Child (complexity builds capacity, repetition doesn't)"
echo "Start: $(date)"
echo ""

run_train() {
    local label="$1"
    shift
    local tmpdir
    tmpdir=$(mktemp -d)
    (
        cd "$tmpdir"
        "$BASE_DIR/$EXE" "$@" > log.txt 2>&1
        cp training_summary.json "$BASE_DIR/$RESULTS_DIR/${label}.json"
        if [ -f checkpoint_final.bin ]; then
            cp checkpoint_final.bin "$BASE_DIR/$RESULTS_DIR/${label}_ckpt.bin"
        fi
    )
    rm -rf "$tmpdir"
}

# Phase 1: Standalone runs (legal_only + shak_only + child_only per seed)
echo "--- Phase 1: Standalone runs (9 jobs) ---"
PIDS=()
for SEED in "${SEEDS[@]}"; do
    run_train "seed${SEED}_legal_only" train "$BASE_DIR/$LEGAL" $ITERS 4 64 3e-4 --seed "$SEED" &
    PIDS+=($!)
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

# Phase 2: Sequential runs (legal->child and shak->child per seed)
echo "--- Phase 2: Sequential runs (6 jobs) ---"
PIDS=()
for SEED in "${SEEDS[@]}"; do
    # Legal → Children
    run_train "seed${SEED}_legal_then_child" train "$BASE_DIR/$CHILD" $((ITERS * 2)) 4 64 3e-4 --seed "$SEED" \
        --resume "$BASE_DIR/$RESULTS_DIR/seed${SEED}_legal_only_ckpt.bin" &
    PIDS+=($!)

    # Shakespeare → Children (reference from Test 1, re-run for these seeds)
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

# Summary
echo "=== RESULTS SUMMARY ==="
echo ""
echo "Final val_loss on CHILDREN corpus (lower = better):"
echo ""
printf "%-6s  %-12s  %-12s  %-12s  %-16s  %-16s\n" "Seed" "Child Only" "Shak Only" "Legal Only" "Shak->Child" "Legal->Child"
printf "%-6s  %-12s  %-12s  %-12s  %-16s  %-16s\n" "----" "----------" "---------" "----------" "-----------" "------------"

for SEED in "${SEEDS[@]}"; do
    CHILD_LOSS=$(grep -o '"final_val_loss": [0-9.]*' "$RESULTS_DIR/seed${SEED}_child_only.json" | grep -o '[0-9.]*$')
    SHAK_LOSS=$(grep -o '"final_val_loss": [0-9.]*' "$RESULTS_DIR/seed${SEED}_shak_only.json" | grep -o '[0-9.]*$')
    LEGAL_LOSS=$(grep -o '"final_val_loss": [0-9.]*' "$RESULTS_DIR/seed${SEED}_legal_only.json" | grep -o '[0-9.]*$')
    S2C_LOSS=$(grep -o '"final_val_loss": [0-9.]*' "$RESULTS_DIR/seed${SEED}_shak_then_child.json" | grep -o '[0-9.]*$')
    L2C_LOSS=$(grep -o '"final_val_loss": [0-9.]*' "$RESULTS_DIR/seed${SEED}_legal_then_child.json" | grep -o '[0-9.]*$')
    printf "%-6s  %-12s  %-12s  %-12s  %-16s  %-16s\n" "$SEED" "$CHILD_LOSS" "$SHAK_LOSS" "$LEGAL_LOSS" "$S2C_LOSS" "$L2C_LOSS"
done

echo ""
echo "Key question: Is Shak->Child consistently better than Legal->Child?"
echo "If yes: complexity (Shakespeare) builds capacity, repetition (legal) doesn't."
echo "If Legal->Child ≈ Shak->Child: any constraint works, not just complexity."
echo "If Legal->Child > Shak->Child: legal actually builds better foundations (?!)"
