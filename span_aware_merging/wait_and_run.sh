#!/bin/bash
# Poll GPU 1 every 60s. When free memory > 12GB, run the full pipeline there.

GPU_ID=3
MIN_FREE_MB=0   # 12 GB
POLL_SEC=60
START_STEP="${START_STEP:-1}"
RAW_DELTA_MODE="${RAW_DELTA_MODE:-0}"
BASE_MODEL="${BASE_MODEL:-/data/uqlinh/merged_models/10-3-reczero4800-tallrec480/to-test/Qwen2.5-3B-Instruct}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-/data/uqlinh/tallrec/output-yelp-clean-20260418/final}"
LOG="/data/uqlinh/rain_merging/span_aware_outputs/wait_and_run.log"
LOCKFILE="/data/uqlinh/rain_merging/span_aware_outputs/wait_and_run.lock"
RAW_DELTA_DIR="${RAW_DELTA_DIR:-/data/uqlinh/rain_merging/span_aware_outputs/raw_tallrec_delta}"
RAW_DELTA_PATH="${RAW_DELTA_PATH:-$RAW_DELTA_DIR/projected_task_vectors.pkl}"
OUTBASE="/data/uqlinh/rain_merging/span_aware_outputs"
LAMBDA="${LAMBDA:-0.5}"
DELTA_FMT="${DELTA_FMT:-0.9}"

CFG='{
  "model_r_path":       "/data/uqlinh/reczero-yelp-ckpt/global_step_320",
  "model_b_path":       "/data/uqlinh/tallrec/output-yelp-clean-20260418/final",
  "stage1_delta_path":  "/data/uqlinh/rain_merging/stage1_output_20260423_195226/projected_task_vectors.pkl",
  "dr_path":  "data/dr_calibration.jsonl",
  "di_path":  "data/di_calibration.jsonl",
  "test_path": "data/dr_calibration.jsonl",
  "masks_dir":          "/data/uqlinh/rain_merging/span_aware_outputs/masks",
  "fisher_dir":         "/data/uqlinh/rain_merging/span_aware_outputs/fisher",
  "attention_dir":      "/data/uqlinh/rain_merging/span_aware_outputs/attention",
  "responsibility_dir": "/data/uqlinh/rain_merging/span_aware_outputs/responsibility",
  "floors_dir":         "/data/uqlinh/rain_merging/span_aware_outputs/floors",
  "coefficients_dir":   "/data/uqlinh/rain_merging/span_aware_outputs/coefficients",
  "checkpoint_dir":     "/data/uqlinh/rain_merging/span_aware_outputs/checkpoints",
  "eval_dir":           "/data/uqlinh/rain_merging/span_aware_outputs/eval",
  "device": "cuda:3",
  "dtype_str": "bfloat16",
  "delta_fmt": 0.9
}'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$(dirname "$LOG")"

# Prevent multiple concurrent instances
exec 9>"$LOCKFILE"
if ! flock -n 9; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Another instance is already running (lock: $LOCKFILE). Exiting." | tee -a "$LOG"
    exit 1
fi

PYTHON=/data/uqlinh/exp3rt/venv/bin/python

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

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

log "Watcher started. Waiting for GPU $GPU_ID to have > ${MIN_FREE_MB}MB free..."

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
    log ">>> $*"
    "$@" >> "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        log "ERROR: command exited with code $rc: $*"
        exit $rc
    fi
}

prepare_stage1_delta

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

LAMBDA_TAG=$("$PYTHON" -c "v=float('$LAMBDA'); print(f'{v:.2f}'.replace('.', '_'))")
DELTA_TAG=$("$PYTHON" -c "v=float('$DELTA_FMT'); print(f'{v:.2f}'.replace('.', '_'))")
RUN_NAME="merged_lambda_${LAMBDA_TAG}_delta_fmt_${DELTA_TAG}"
CKPT_DIR="$OUTBASE/checkpoints/$RUN_NAME"

RUN_CFG=$(printf '%s' "$CFG" | "$PYTHON" -c "
import json, sys
cfg = json.load(sys.stdin)
cfg['delta_fmt'] = float('$DELTA_FMT')
print(json.dumps(cfg))
")

log "=== Step 7: merge (lambda=$LAMBDA, delta_fmt=$DELTA_FMT) ==="
run $PYTHON step7_merge.py --override_json "$RUN_CFG" --lambda_global "$LAMBDA"

log "=== Step 8: smoke test ==="
$PYTHON step8_smoketest.py \
    --checkpoint_dir "$CKPT_DIR" \
    --override_json "$RUN_CFG" >> "$LOG" 2>&1
SMOKE_RC=$?
if [ $SMOKE_RC -eq 0 ]; then
    log "Smoke test PASSED."
else
    log "Smoke test FAILED (exit $SMOKE_RC). Check $LOG for details."
fi

log "Pipeline complete."
