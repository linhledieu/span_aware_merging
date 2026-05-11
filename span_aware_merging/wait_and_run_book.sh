#!/bin/bash
# Poll a GPU until enough memory is free, then run the span-aware merge pipeline
# for the book domain. Raw stage1/stage2 book files are converted on demand into
# the calibration schema expected by span_aware_merging.
#
# Example:
#   GPU_ID=3 OUTBASE=/data/uqlinh/rain_merging/span_aware_outputs_book \
#   bash wait_and_run_book.sh
#
# Resume support:
#   START_STEP=3 bash wait_and_run_book.sh

set -euo pipefail

GPU_ID="${GPU_ID:-3}"
MIN_FREE_MB="${MIN_FREE_MB:-30000}"
POLL_SEC="${POLL_SEC:-60}"
PYTHON="${PYTHON:-/data/uqlinh/exp3rt/venv/bin/python}"
BENCHMARK="${BENCHMARK:-/home/uqlinh/ReasonMerge/EXP3RT/theory_test/unified_benchmark.py}"
START_STEP="${START_STEP:-1}"
export PYTHONUNBUFFERED=1

DEFAULT_BASE="/data/uqlinh/merged_models/10-3-reczero4800-tallrec480/to-test/Qwen2.5-3B-Instruct"
DEFAULT_INSTRUCT="/data/uqlinh/tallrec/output-book-20260418/final"
DEFAULT_TARGET="/data/uqlinh/reczero/checkpoints/08-3/actor/global_step_2000"
DEFAULT_STAGE1_DATA="/home/uqlinh/RAIN-Merging/data/book/book_reason_correct_direct_fail_full_subset.jsonl"
DEFAULT_STAGE2_DATA="/home/uqlinh/RAIN-Merging/data/book/instruction_calibration_set_from_reason_correct_direct_fail.jsonl"
DEFAULT_OUTPUT="/data/uqlinh/rain_merging/book_span_aware_outputs"

BASE_MODEL="${BASE_MODEL:-$DEFAULT_BASE}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-$DEFAULT_INSTRUCT}"
TARGET_MODEL="${TARGET_MODEL:-$DEFAULT_TARGET}"
STAGE1_RAW="${STAGE1_RAW:-$DEFAULT_STAGE1_DATA}"
STAGE2_RAW="${STAGE2_RAW:-$DEFAULT_STAGE2_DATA}"
RAW_DELTA_MODE="${RAW_DELTA_MODE:-0}"
STAGE1_DELTA_PATH="${STAGE1_DELTA_PATH:-/data/uqlinh/rain_merging/stage1_output_book/stage1_output/projected_task_vectors.pkl}"
OUTBASE="${OUTBASE:-/data/uqlinh/rain_merging/span_aware_outputs_book}"
CALIB_DIR="${CALIB_DIR:-$PWD/data/book}"
DR_PATH="${DR_PATH:-$CALIB_DIR/dr_book_calibration.jsonl}"
DI_PATH="${DI_PATH:-$CALIB_DIR/di_book_calibration.jsonl}"
TEST_PATH="${TEST_PATH:-$DR_PATH}"
DATA_PATH="${DATA_PATH:-/data/uqlinh/reczero/data/book/test.parquet}"
LAMBDA="${LAMBDA:-0.5}"
DELTA_FMT_LIST="${DELTA_FMT_LIST:-0.8 0.9 1.0 1.1}"
ATTENTION_CHUNK_SIZE="${ATTENTION_CHUNK_SIZE:-256}"

LOG="$OUTBASE/wait_and_run.log"
LOCKFILE="$OUTBASE/wait_and_run.lock"
RAW_DELTA_DIR="${RAW_DELTA_DIR:-$OUTBASE/raw_tallrec_delta}"
RAW_DELTA_PATH="${RAW_DELTA_PATH:-$RAW_DELTA_DIR/projected_task_vectors.pkl}"

BASE_CFG_TEMPLATE='{
  "model_r_path":       "TARGET_MODEL_PLACEHOLDER",
  "model_b_path":       "INSTRUCT_MODEL_PLACEHOLDER",
  "stage1_delta_path":  "STAGE1_DELTA_PATH_PLACEHOLDER",
  "dr_path":  "DR_PATH_PLACEHOLDER",
  "di_path":  "DI_PATH_PLACEHOLDER",
  "test_path": "TEST_PATH_PLACEHOLDER",
  "masks_dir":          "OUTBASE_PLACEHOLDER/masks",
  "fisher_dir":         "OUTBASE_PLACEHOLDER/fisher",
  "attention_dir":      "OUTBASE_PLACEHOLDER/attention",
  "responsibility_dir": "OUTBASE_PLACEHOLDER/responsibility",
  "floors_dir":         "OUTBASE_PLACEHOLDER/floors",
  "coefficients_dir":   "OUTBASE_PLACEHOLDER/coefficients",
  "checkpoint_dir":     "OUTBASE_PLACEHOLDER/checkpoints",
  "eval_dir":           "OUTBASE_PLACEHOLDER/eval",
  "device": "cuda:GPU_ID_PLACEHOLDER",
  "dtype_str": "bfloat16",
  "attention_chunk_size": ATTENTION_CHUNK_SIZE_PLACEHOLDER
}'

CFG="${BASE_CFG_TEMPLATE//TARGET_MODEL_PLACEHOLDER/$TARGET_MODEL}"
CFG="${CFG//INSTRUCT_MODEL_PLACEHOLDER/$INSTRUCT_MODEL}"
CFG="${CFG//STAGE1_DELTA_PATH_PLACEHOLDER/$STAGE1_DELTA_PATH}"
CFG="${CFG//DR_PATH_PLACEHOLDER/$DR_PATH}"
CFG="${CFG//DI_PATH_PLACEHOLDER/$DI_PATH}"
CFG="${CFG//TEST_PATH_PLACEHOLDER/$TEST_PATH}"
CFG="${CFG//OUTBASE_PLACEHOLDER/$OUTBASE}"
CFG="${CFG//GPU_ID_PLACEHOLDER/$GPU_ID}"
CFG="${CFG//ATTENTION_CHUNK_SIZE_PLACEHOLDER/$ATTENTION_CHUNK_SIZE}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$OUTBASE" "$CALIB_DIR"

exec 9>"$LOCKFILE"
if ! flock -n 9; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Another instance is already running (lock: $LOCKFILE). Exiting." | tee -a "$LOG"
    exit 1
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run() {
    local quoted=""
    printf -v quoted '%q ' "$@"
    log ">>> ${quoted% }"
    "$@" >> "$LOG" 2>&1
    local rc=$?
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

prepare_calibration_files() {
    log "Preparing span-aware calibration files for book domain"
    STAGE1_RAW="$STAGE1_RAW" \
    STAGE2_RAW="$STAGE2_RAW" \
    DR_PATH="$DR_PATH" \
    DI_PATH="$DI_PATH" \
    run "$PYTHON" -c '
import json
import os
from pathlib import Path

stage1_raw = Path(os.environ["STAGE1_RAW"])
stage2_raw = Path(os.environ["STAGE2_RAW"])
dr_path = Path(os.environ["DR_PATH"])
di_path = Path(os.environ["DI_PATH"])

def write_dr(inp: Path, out: Path) -> int:
    count = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with inp.open() as fin, out.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt = rec["prompt_long_native"]
            response = rec["raw_output_long_native"]
            full_text = prompt + response if prompt.endswith("\n") else prompt + "\n" + response
            out_rec = {
                "id": str(rec.get("example_id", count)),
                "full_text": full_text,
                "gt": float(rec["gt"]),
            }
            fout.write(json.dumps(out_rec) + "\n")
            count += 1
    return count

def write_di(inp: Path, out: Path) -> int:
    count = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with inp.open() as fin, out.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt = rec["prompt"]
            response = rec["response"]
            full_text = prompt + response if prompt.endswith("\n") else prompt + "\n" + response
            meta = rec.get("meta", {})
            gt = meta.get("gt", 0.0) if isinstance(meta, dict) else 0.0
            out_rec = {
                "id": str(rec["id"]),
                "full_text": full_text,
                "gt": float(gt),
            }
            fout.write(json.dumps(out_rec) + "\n")
            count += 1
    return count

n_dr = write_dr(stage1_raw, dr_path)
n_di = write_di(stage2_raw, di_path)
print(f"Prepared D_R: {n_dr} -> {dr_path}")
print(f"Prepared D_I: {n_di} -> {di_path}")
'
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
        STAGE1_DELTA_PATH="$RAW_DELTA_PATH"
        CFG=$(printf '%s' "$CFG" | "$PYTHON" -c "
import json, sys
cfg = json.load(sys.stdin)
cfg['stage1_delta_path'] = '$RAW_DELTA_PATH'
print(json.dumps(cfg))
")
    fi

    require_file "$STAGE1_DELTA_PATH"
}

if ! [[ "$START_STEP" =~ ^[1-8]$ ]]; then
    log "ERROR: START_STEP must be an integer from 1 to 8 (got '$START_STEP')"
    exit 1
fi

require_file "$PYTHON"
require_file "$BENCHMARK"
require_file "$STAGE1_RAW"
require_file "$STAGE2_RAW"
require_file "$TARGET_MODEL/config.json"
require_file "$INSTRUCT_MODEL/config.json"

prepare_calibration_files
prepare_stage1_delta
require_file "$DR_PATH"
require_file "$DI_PATH"

log "Watcher started (book). Waiting for GPU $GPU_ID to have > ${MIN_FREE_MB}MB free..."

while true; do
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" 2>/dev/null | tr -d ' ')
    if [ -z "$FREE_MB" ]; then
        log "ERROR: Could not query GPU $GPU_ID. Is nvidia-smi available?"
        sleep "$POLL_SEC"
        continue
    fi
    log "GPU $GPU_ID free memory: ${FREE_MB}MB / ${MIN_FREE_MB}MB threshold"
    if [ "$FREE_MB" -ge "$MIN_FREE_MB" ]; then
        log "Threshold met (${FREE_MB}MB >= ${MIN_FREE_MB}MB). Launching pipeline on cuda:$GPU_ID"
        break
    fi
    sleep "$POLL_SEC"
done

log "Resume mode: START_STEP=$START_STEP"

if [ "$START_STEP" -le 1 ]; then
    log "=== Step 1: span masks (CPU) ==="
    run "$PYTHON" step1_span_masks.py --override_json "$CFG"
else
    log "=== Step 1: span masks (CPU) — skipped (START_STEP=$START_STEP) ==="
    require_file "$OUTBASE/masks/dr_masks.pkl"
    require_file "$OUTBASE/masks/di_masks.pkl"
fi

if [ "$START_STEP" -le 2 ]; then
    log "=== Step 2: Fisher (GPU $GPU_ID) ==="
    run "$PYTHON" step2_fisher.py --override_json "$CFG"
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
    run "$PYTHON" step3_attention.py --override_json "$CFG"
else
    log "=== Step 3: attention statistics (GPU $GPU_ID) — skipped (START_STEP=$START_STEP) ==="
    require_file "$OUTBASE/attention/U.pt"
    require_file "$OUTBASE/attention/a_align.pt"
    require_file "$OUTBASE/attention/u_leak.pt"
fi

if [ "$START_STEP" -le 4 ]; then
    log "=== Step 4: responsibility ==="
    run "$PYTHON" step4_responsibility.py --override_json "$CFG"
else
    log "=== Step 4: responsibility — skipped (START_STEP=$START_STEP) ==="
fi

if [ "$START_STEP" -le 5 ]; then
    log "=== Step 5: floors ==="
    run "$PYTHON" step5_floors.py --override_json "$CFG"
else
    log "=== Step 5: floors — skipped (START_STEP=$START_STEP) ==="
fi

if [ "$START_STEP" -le 6 ]; then
    log "=== Step 6: coefficients ==="
    run "$PYTHON" step6_coefficients.py --override_json "$CFG"
else
    log "=== Step 6: coefficients — skipped (START_STEP=$START_STEP) ==="
fi

LAMBDA_TAG=$("$PYTHON" -c "v=float('$LAMBDA'); print(f'{v:.2f}'.replace('.', '_'))")

for DELTA_FMT in $DELTA_FMT_LIST; do
    DELTA_TAG=$("$PYTHON" -c "v=float('$DELTA_FMT'); print(f'{v:.2f}'.replace('.', '_'))")
    RUN_NAME="merged_lambda_${LAMBDA_TAG}_delta_fmt_${DELTA_TAG}"
    CKPT_DIR="$OUTBASE/checkpoints/$RUN_NAME"
    EVAL_DIR="$OUTBASE/eval/${RUN_NAME}_full_eval_bs1_long"

    RUN_CFG=$(printf '%s' "$CFG" | "$PYTHON" -c "
import json, sys
cfg = json.load(sys.stdin)
cfg['delta_fmt'] = float('$DELTA_FMT')
print(json.dumps(cfg))
")

    log "=== Step 7: merge (lambda=$LAMBDA, delta_fmt=$DELTA_FMT) ==="
    run "$PYTHON" step7_merge.py --override_json "$RUN_CFG" --lambda_global "$LAMBDA"

    log "=== Step 8: smoke test ($RUN_NAME) ==="
    "$PYTHON" step8_smoketest.py \
        --checkpoint_dir "$CKPT_DIR" \
        --override_json "$RUN_CFG" >> "$LOG" 2>&1
    SMOKE_RC=$?
    if [ $SMOKE_RC -eq 0 ]; then
        log "Smoke test PASSED ($RUN_NAME)."
    else
        log "Smoke test FAILED (exit $SMOKE_RC) for $RUN_NAME — continuing to eval anyway."
    fi

    log "=== Eval: $RUN_NAME ==="
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$BENCHMARK" eval \
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
