#!/bin/bash
# Poll GPU 3 every 60s. When free, run the full pipeline (music domain).
# Produces 4 merged checkpoints: lambda=0.5 x delta_fmt in {0.8, 0.9, 1.0, 1.1}
# Then evals each with unified_benchmark.py on GPU 3.
#
# Resume support:
#   START_STEP=3 bash wait_and_run_music.sh   # reuse masks + Fisher, start from attention

GPU_ID=3
# Step 2 Fisher needs nearly the whole 44GB card for backward, so wait until GPU 3
# is effectively idle instead of starting as soon as ~30GB is free.
MIN_FREE_MB=20000
POLL_SEC=60
OUTBASE="/data/uqlinh/rain_merging/span_aware_outputs_music"
LOG="$OUTBASE/wait_and_run.log"
LOCKFILE="$OUTBASE/wait_and_run.lock"

PYTHON=/data/uqlinh/exp3rt/venv/bin/python
BENCHMARK=/home/uqlinh/ReasonMerge/EXP3RT/theory_test/unified_benchmark.py
DATA_PATH=/data/uqlinh/ReasonMerge/data/music/test/test_reczero.parquet
START_STEP="${START_STEP:-1}"
RAW_DELTA_MODE="${RAW_DELTA_MODE:-0}"
BASE_MODEL="${BASE_MODEL:-/data/uqlinh/merged_models/10-3-reczero4800-tallrec480/to-test/Qwen2.5-3B-Instruct}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-/data/uqlinh/tallrec/output-music-fixed/final}"
RAW_DELTA_DIR="${RAW_DELTA_DIR:-$OUTBASE/raw_tallrec_delta}"
RAW_DELTA_PATH="${RAW_DELTA_PATH:-$RAW_DELTA_DIR/projected_task_vectors.pkl}"
export PYTHONUNBUFFERED=1

# Base config — delta_fmt and checkpoint_dir are overridden per run below
BASE_CFG='{
  "model_r_path":       "/data/uqlinh/reczero-music-ckpt",
  "model_b_path":       "/data/uqlinh/tallrec/output-music-fixed/final",
  "stage1_delta_path":  "/data/uqlinh/rain_merging/stage1_output_music/projected_task_vectors.pkl",
  "dr_path":  "data/dr_music_calibration.jsonl",
  "di_path":  "data/di_music_calibration.jsonl",
  "test_path": "data/dr_music_calibration.jsonl",
  "masks_dir":          "OUTBASE/masks",
  "fisher_dir":         "OUTBASE/fisher",
  "attention_dir":      "OUTBASE/attention",
  "responsibility_dir": "OUTBASE/responsibility",
  "floors_dir":         "OUTBASE/floors",
  "coefficients_dir":   "OUTBASE/coefficients",
  "checkpoint_dir":     "OUTBASE/checkpoints",
  "eval_dir":           "OUTBASE/eval",
  "device": "cuda:3",
  "dtype_str": "bfloat16",
  "attention_chunk_size": 256
}'

# Substitute OUTBASE placeholder
CFG="${BASE_CFG//OUTBASE/$OUTBASE}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$OUTBASE"

exec 9>"$LOCKFILE"
if ! flock -n 9; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Another instance is already running (lock: $LOCKFILE). Exiting." | tee -a "$LOG"
    exit 1
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "Watcher started (music). Waiting for GPU $GPU_ID to have > ${MIN_FREE_MB}MB free..."

while true; do
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU_ID 2>/dev/null | tr -d ' ')
    if [ -z "$FREE_MB" ]; then
        log "ERROR: Could not query GPU $GPU_ID. Is nvidia-smi available?"
        sleep $POLL_SEC
        continue
    fi
    log "GPU $GPU_ID free memory: ${FREE_MB}MB / ${MIN_FREE_MB}MB threshold"
    if [ "$FREE_MB" -ge "$MIN_FREE_MB" ]; then
        log "Threshold met (${FREE_MB}MB >= ${MIN_FREE_MB}MB). Launching pipeline on cuda:$GPU_ID"
        break
    fi
    sleep $POLL_SEC
done

run() {
    local quoted=""
    printf -v quoted '%q ' "$@"
    log ">>> ${quoted% }"
    "$@" >> "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        log "ERROR: command exited with code $rc: ${quoted% }"
        exit $rc
    fi
}

require_file() {
    if [ ! -f "$1" ]; then
        log "ERROR: required file missing: $1"
        exit 1
    fi
}

prepare_stage1_delta() {
    if [ "$RAW_DELTA_MODE" = "1" ]; then
        log "Preparing raw-checkpoint delta from TallRec (nullspace bypass mode)"
        require_file "$BASE_MODEL/config.json"
        require_file "$INSTRUCT_MODEL/config.json"
        mkdir -p "$RAW_DELTA_DIR"
        run "$PYTHON" "$SCRIPT_DIR/../raw_ckpt_to_projected.py" \
            --base "$BASE_MODEL" \
            --instruct "$INSTRUCT_MODEL" \
            --output_file "$RAW_DELTA_PATH" \
            --merge_types qkvof
        CFG=$(printf '%s' "$CFG" | "$PYTHON" -c "
import json, sys
cfg = json.load(sys.stdin)
cfg['stage1_delta_path'] = '$RAW_DELTA_PATH'
print(json.dumps(cfg))
")
    fi

    local delta_path
    delta_path=$(printf '%s' "$CFG" | "$PYTHON" -c "import json, sys; print(json.load(sys.stdin)['stage1_delta_path'])")
    require_file "$delta_path"
}

if ! [[ "$START_STEP" =~ ^[1-8]$ ]]; then
    log "ERROR: START_STEP must be an integer from 1 to 8 (got '$START_STEP')"
    exit 1
fi

prepare_stage1_delta

# ── Steps 1-6: shared across all merge variants ───────────────────────────────

log "Resume mode: START_STEP=$START_STEP"

if [ "$START_STEP" -le 1 ]; then
    log "=== Step 1: span masks (CPU) ==="
    run $PYTHON step1_span_masks.py --override_json "$CFG"
else
    log "=== Step 1: span masks (CPU) — skipped (START_STEP=$START_STEP) ==="
    require_file "$OUTBASE/masks/dr_masks.pkl"
    require_file "$OUTBASE/masks/di_masks.pkl"
fi

if [ "$START_STEP" -le 2 ]; then
    log "=== Step 2: Fisher (GPU $GPU_ID) ==="
    run $PYTHON step2_fisher.py --override_json "$CFG"
else
    log "=== Step 2: Fisher (GPU $GPU_ID) — skipped (START_STEP=$START_STEP) ==="
    for span in tag match user item; do
        for proj in q k v o; do
            require_file "$OUTBASE/fisher/fisher_raw_${span}_${proj}.pt"
            require_file "$OUTBASE/fisher/fisher_norm_${span}_${proj}.pt"
        done
    done
fi

if [ "$START_STEP" -le 3 ]; then
    log "=== Step 3: attention statistics (GPU $GPU_ID) ==="
    run $PYTHON step3_attention.py --override_json "$CFG"
else
    log "=== Step 3: attention statistics (GPU $GPU_ID) — skipped (START_STEP=$START_STEP) ==="
    require_file "$OUTBASE/attention/U.pt"
    require_file "$OUTBASE/attention/a_align.pt"
    require_file "$OUTBASE/attention/u_leak.pt"
fi

if [ "$START_STEP" -le 4 ]; then
    log "=== Step 4: responsibility ==="
    run $PYTHON step4_responsibility.py --override_json "$CFG"
else
    log "=== Step 4: responsibility — skipped (START_STEP=$START_STEP) ==="
fi

if [ "$START_STEP" -le 5 ]; then
    log "=== Step 5: floors ==="
    run $PYTHON step5_floors.py --override_json "$CFG"
else
    log "=== Step 5: floors — skipped (START_STEP=$START_STEP) ==="
fi

if [ "$START_STEP" -le 6 ]; then
    log "=== Step 6: coefficients ==="
    run $PYTHON step6_coefficients.py --override_json "$CFG"
else
    log "=== Step 6: coefficients — skipped (START_STEP=$START_STEP) ==="
fi

# ── Steps 7-8 + eval: one pass per delta_fmt ─────────────────────────────────

LAMBDA=0.5
LAMBDA_TAG="0_50"

for DELTA_FMT in 0.8 0.9 1.0 1.1; do
    # Build tag: 0.8 -> 0_80, 0.9 -> 0_90, 1.0 -> 1_00, 1.1 -> 1_10
    DELTA_TAG=$(python3 -c "v='$DELTA_FMT'; a,b=v.split('.'); print(f'{a}_{int(b)*10:02d}')")
    RUN_NAME="merged_lambda_${LAMBDA_TAG}_delta_fmt_${DELTA_TAG}"
    CKPT_DIR="$OUTBASE/checkpoints/$RUN_NAME"
    EVAL_DIR="$OUTBASE/eval/${RUN_NAME}_full_eval_bs1_long"

    # Inject delta_fmt into cfg for this run
    RUN_CFG=$(echo "$CFG" | python3 -c "
import sys, json
cfg = json.load(sys.stdin)
cfg['delta_fmt'] = $DELTA_FMT
print(json.dumps(cfg))
")

    log "=== Step 7: merge (lambda=$LAMBDA, delta_fmt=$DELTA_FMT) ==="
    run $PYTHON step7_merge.py --override_json "$RUN_CFG" --lambda_global $LAMBDA

    log "=== Step 8: smoke test ($RUN_NAME) ==="
    $PYTHON step8_smoketest.py \
        --checkpoint_dir "$CKPT_DIR" \
        --override_json "$RUN_CFG" >> "$LOG" 2>&1
    SMOKE_RC=$?
    if [ $SMOKE_RC -eq 0 ]; then
        log "Smoke test PASSED ($RUN_NAME)."
    else
        log "Smoke test FAILED (exit $SMOKE_RC) for $RUN_NAME — continuing to eval anyway."
    fi

    log "=== Eval: $RUN_NAME ==="
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON "$BENCHMARK" eval \
        --model_path "$CKPT_DIR" \
        --model_name "$RUN_NAME" \
        --data_path "$DATA_PATH" \
        --regime native_reczero \
        --output_dir "$EVAL_DIR" \
        --device cuda \
        --dtype bfloat16 \
        --batch_size 1 \
        --max_input_tokens 6000 \
        --max_new_tokens 2000 \
        --seed 1234 \
        --log_every 1 \
        --strict_parse_mode rate_tag_only \
        --length_bucket_batches >> "$LOG" 2>&1
    EVAL_RC=$?
    if [ $EVAL_RC -eq 0 ]; then
        log "Eval PASSED ($RUN_NAME). Results in $EVAL_DIR"
    else
        log "Eval FAILED (exit $EVAL_RC) for $RUN_NAME."
    fi

done

log "All runs complete."
