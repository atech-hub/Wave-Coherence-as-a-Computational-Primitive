#!/bin/bash
# Test 3: Three-stage corpus curriculum + iteration confound control
# 3 configs × 3 seeds = 9 training sessions (some multi-stage)
#
# Config 1: Legal → Shakespeare → Children (3000 + 3000 + 3000 = 9000 iters)
# Config 2: Children → Shakespeare → Legal (reversed, 9000 iters)
# Config 3: Children only × 9000 iters (iteration-count control)
#
# Final val_loss measured on the LAST corpus in each sequence.
# Config 1 & 3 measure on children. Config 2 measures on legal.
# For fair comparison, also measure Config 2 on children via a 4th config:
# Config 4: Children only × 3000 (baseline reference from Test 1b)

set -e

SEEDS=(42 100 200)
SHAK="data/input.txt"
CHILD="data/children.txt"
LEGAL="data/legal.txt"
ITERS=3000
RESULTS_DIR="test3_threestage"
EXE="target/release/kerr-engine.exe"
BASE_DIR="$(pwd)"

mkdir -p "$RESULTS_DIR"

echo "=== Test 3: Three-Stage Corpus Curriculum ==="
echo "Seeds: ${SEEDS[*]}"
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
        cp log.txt "$BASE_DIR/$RESULTS_DIR/${label}.log"
    )
    rm -rf "$tmpdir"
}

# Phase 1: First-stage training (all seeds in parallel)
# Legal (for config 1), Children (for configs 2, 3, 4), Shakespeare standalone not needed
echo "--- Phase 1: First-stage training ---"
PIDS=()
for SEED in "${SEEDS[@]}"; do
    # Config 1 stage 1: Legal
    run_train "seed${SEED}_c1_legal" train "$BASE_DIR/$LEGAL" $ITERS 4 64 3e-4 --seed "$SEED" &
    PIDS+=($!)
    # Config 2 stage 1: Children
    run_train "seed${SEED}_c2_child" train "$BASE_DIR/$CHILD" $ITERS 4 64 3e-4 --seed "$SEED" &
    PIDS+=($!)
    # Config 3: Children 9000 iters (can run all at once, no stages)
    run_train "seed${SEED}_c3_child9k" train "$BASE_DIR/$CHILD" $((ITERS * 3)) 4 64 3e-4 --seed "$SEED" &
    PIDS+=($!)
done
echo "  Launched ${#PIDS[@]} jobs, waiting..."
for PID in "${PIDS[@]}"; do wait "$PID" || true; done
echo "  Phase 1 done ($(date))"
echo ""

# Phase 2: Second-stage training
echo "--- Phase 2: Second-stage training ---"
PIDS=()
for SEED in "${SEEDS[@]}"; do
    # Config 1 stage 2: Legal → Shakespeare
    run_train "seed${SEED}_c1_legal_shak" train "$BASE_DIR/$SHAK" $((ITERS * 2)) 4 64 3e-4 --seed "$SEED" \
        --resume "$BASE_DIR/$RESULTS_DIR/seed${SEED}_c1_legal_ckpt.bin" &
    PIDS+=($!)
    # Config 2 stage 2: Children → Shakespeare
    run_train "seed${SEED}_c2_child_shak" train "$BASE_DIR/$SHAK" $((ITERS * 2)) 4 64 3e-4 --seed "$SEED" \
        --resume "$BASE_DIR/$RESULTS_DIR/seed${SEED}_c2_child_ckpt.bin" &
    PIDS+=($!)
done
echo "  Launched ${#PIDS[@]} jobs, waiting..."
for PID in "${PIDS[@]}"; do wait "$PID" || true; done
echo "  Phase 2 done ($(date))"
echo ""

# Phase 3: Third-stage training
echo "--- Phase 3: Third-stage training ---"
PIDS=()
for SEED in "${SEEDS[@]}"; do
    # Config 1 stage 3: Legal → Shakespeare → Children
    run_train "seed${SEED}_c1_final" train "$BASE_DIR/$CHILD" $((ITERS * 3)) 4 64 3e-4 --seed "$SEED" \
        --resume "$BASE_DIR/$RESULTS_DIR/seed${SEED}_c1_legal_shak_ckpt.bin" &
    PIDS+=($!)
    # Config 2 stage 3: Children → Shakespeare → Legal
    run_train "seed${SEED}_c2_final" train "$BASE_DIR/$LEGAL" $((ITERS * 3)) 4 64 3e-4 --seed "$SEED" \
        --resume "$BASE_DIR/$RESULTS_DIR/seed${SEED}_c2_child_shak_ckpt.bin" &
    PIDS+=($!)
done
echo "  Launched ${#PIDS[@]} jobs, waiting..."
for PID in "${PIDS[@]}"; do wait "$PID" || true; done
echo "  Phase 3 done ($(date))"
echo ""

# Summary
echo "=== RESULTS SUMMARY ==="
echo ""
echo "All configs: 9000 total iterations. Val loss on final corpus."
echo ""
printf "%-6s  %-22s  %-22s  %-18s\n" "Seed" "Legal->Shak->Child" "Child->Shak->Legal" "Child 9K (control)"
printf "%-6s  %-22s  %-22s  %-18s\n" "----" "------------------" "------------------" "------------------"

for SEED in "${SEEDS[@]}"; do
    C1_LOSS=$(grep -o '"final_val_loss": [0-9.]*' "$RESULTS_DIR/seed${SEED}_c1_final.json" | grep -o '[0-9.]*$')
    C2_LOSS=$(grep -o '"final_val_loss": [0-9.]*' "$RESULTS_DIR/seed${SEED}_c2_final.json" | grep -o '[0-9.]*$')
    C3_LOSS=$(grep -o '"final_val_loss": [0-9.]*' "$RESULTS_DIR/seed${SEED}_c3_child9k.json" | grep -o '[0-9.]*$')
    printf "%-6s  %-22s  %-22s  %-18s\n" "$SEED" "$C1_LOSS" "$C2_LOSS" "$C3_LOSS"
done

echo ""
echo "Key questions:"
echo "  1. Does Legal->Shak->Child beat Child-9K? (diversity > iteration count)"
echo "  2. Does three-stage beat two-stage Shak->Child (~1.95 from Test 1b)?"
echo "  3. Is there an ordering effect between the two three-stage configs?"
echo ""
echo "Note: Config 1 and 3 val_loss on children. Config 2 val_loss on legal."
echo "Cross-corpus comparison (Config 1 vs Config 2) requires same eval corpus."
echo "Config 1 vs Config 3 is the clean comparison (both eval on children, same total iters)."
