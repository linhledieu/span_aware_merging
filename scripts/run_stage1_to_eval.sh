#!/bin/bash

set -euo pipefail

# End-to-end pipeline:
#   Stage 1: nullspace projection + per-layer lambda search
#   Stage 2: QP alpha optimization
#   Stage 3: unified merge
#   Eval: unified_benchmark.py eval
#
# Edit the variables in the Configuration section if you want different paths.

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

PYTHON_BIN="${PYTHON_BIN:-/data/uqlinh/exp3rt/venv/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export CUDA_VISIBLE_DEVICES

REPO_DIR="/home/uqlinh/RAIN-Merging"
EVAL_SCRIPT="/home/uqlinh/ReasonMerge/EXP3RT/theory_test/unified_benchmark.py"

# =========================
# Configuration
# =========================

DEFAULT_BASE="/data/uqlinh/merged_models/10-3-reczero4800-tallrec480/to-test/Qwen2.5-3B-Instruct"
DEFAULT_INSTRUCT="/data/uqlinh/tallrec/output-yelp-clean-20260418/final"
DEFAULT_TARGET="/data/uqlinh/reczero-yelp-ckpt/global_step_320"
DEFAULT_DATA="/home/uqlinh/ReasonMerge/Long-to-Short-via-Model-Merging/RAIN-Merging/data/layer2_reason_correct_direct_fail_low_all_high_exact.jsonl"
DEFAULT_OUTPUT="/data/uqlinh/rain_merging/stage1_output_$(date +%Y%m%d_%H%M%S)"

RUN_NAME="${RUN_NAME:-$(basename "${DEFAULT_OUTPUT}")}"
RUN_ROOT="${RUN_ROOT:-${DEFAULT_OUTPUT}}"

BASE_MODEL="${BASE_MODEL:-$DEFAULT_BASE}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-$DEFAULT_INSTRUCT}"
TARGET_MODEL="${TARGET_MODEL:-$DEFAULT_TARGET}"

STAGE1_TEXTS="${STAGE1_TEXTS:-$DEFAULT_DATA}"
STAGE2_TEXTS="${STAGE2_TEXTS:-/home/uqlinh/RAIN-Merging/data/instruction_calibration_set_from_reason_correct_direct_fail_100.jsonl}"
EVAL_DATA="${EVAL_DATA:-/data/uqlinh/ReasonMerge/data/yelp/test/test_reczero.parquet}"
EVAL_REGIME="${EVAL_REGIME:-native_reczero}"

MAX_SAMPLES="${MAX_SAMPLES:-100}"
LAYERS_TAIL="${LAYERS_TAIL:-27}"
HEADS="${HEADS:-all}"
MERGE_TYPES="${MERGE_TYPES:-qkvof}"
COMPUTE_PRECISION="${COMPUTE_PRECISION:-fp32}"
LAMBDA_RIDGE="${LAMBDA_RIDGE:-1e-4}"
CG_MAXIT="${CG_MAXIT:-100}"
CG_TOL="${CG_TOL:-1e-5}"

Q_ROWS="${Q_ROWS:-8}"
K_ROWS="${K_ROWS:-8}"
V_ROWS="${V_ROWS:-4}"
O_ROWS="${O_ROWS:-4}"
FFN_ROWS="${FFN_ROWS:-4}"
W_Q="${W_Q:-1.0}"
W_K="${W_K:-1.0}"
W_V="${W_V:-1.0}"
W_O="${W_O:-1.0}"
W_FFN="${W_FFN:-1.0}"
READOUT_DIRS="${READOUT_DIRS:-2}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-7168}"
PROJECTION_MODE="${PROJECTION_MODE:-dual_ab}"
LAMBDA_CANDIDATES="${LAMBDA_CANDIDATES:-1.5,1.2,1.0,0.8,0.5,0.3}"
MARGIN_RETENTION_THRESHOLD="${MARGIN_RETENTION_THRESHOLD:-0.80}"
LAMBDA_SEARCH_MAX_NEW_TOKENS="${LAMBDA_SEARCH_MAX_NEW_TOKENS:-512}"
LAMBDA_GENERATION_BATCH_SIZE="${LAMBDA_GENERATION_BATCH_SIZE:-16}"
LAMBDA_MARGIN_BATCH_SIZE="${LAMBDA_MARGIN_BATCH_SIZE:-16}"

QP_VARIANT="${QP_VARIANT:-two_pass}"
PRIOR_SCALAR="${PRIOR_SCALAR:-1.0}"
L2_PRIOR="${L2_PRIOR:-0.1}"
L1_REG="${L1_REG:-0.0}"
BOX_LO="${BOX_LO:-0.0}"
BOX_HI="${BOX_HI:-1.5}"
DEVICE="${DEVICE:-cuda:0}"
H_LAMBDA="${H_LAMBDA:-1.0}"
H_MU="${H_MU:-1.0}"
RHO_DU="${RHO_DU:-0.5}"
KAPPA_A="${KAPPA_A:-1.0}"
KAPPA_U="${KAPPA_U:-1.0}"

SCALING_FACTOR="${SCALING_FACTOR:-1.0}"
MODEL_NAME="${MODEL_NAME:-merged_model}"

EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
EVAL_DTYPE="${EVAL_DTYPE:-bfloat16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_INPUT_TOKENS="${EVAL_MAX_INPUT_TOKENS:-6000}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-1000}"
EVAL_SEED="${EVAL_SEED:-1234}"
EVAL_LOG_EVERY="${EVAL_LOG_EVERY:-1}"
STRICT_PARSE_MODE="${STRICT_PARSE_MODE:-rate_tag_only}"

# Derived paths
STAGE1_DIR="${RUN_ROOT}/stage1_output"
STAGE2_DIR="${RUN_ROOT}/stage2_output"
STAGE3_DIR="${RUN_ROOT}/stage3_output"
PROJECTED_FILE="${STAGE1_DIR}/projected_task_vectors.pkl"
ALPHA_FILE="${STAGE2_DIR}/alpha_true_forward_align_leak.pt"
MERGED_MODEL_PATH="${STAGE3_DIR}/${MODEL_NAME}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-/data/uqlinh/ReasonMerge/eval_results/yelp/merged_model/${RUN_NAME}_test_reczero_eval}"
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-${RUN_NAME}_${MODEL_NAME}}"

show_help() {
    cat <<EOF
Usage: $0

Environment overrides:
  RUN_NAME, RUN_ROOT
  BASE_MODEL, INSTRUCT_MODEL, TARGET_MODEL
  STAGE1_TEXTS, STAGE2_TEXTS, EVAL_DATA, EVAL_REGIME
  MAX_SAMPLES, LAYERS_TAIL, HEADS, MERGE_TYPES
  LAMBDA_CANDIDATES, MARGIN_RETENTION_THRESHOLD
  QP_VARIANT, BOX_LO, BOX_HI, PRIOR_SCALAR, L2_PRIOR
  MODEL_NAME, SCALING_FACTOR
  EVAL_OUTPUT_DIR, EVAL_BATCH_SIZE, EVAL_MAX_NEW_TOKENS

Example:
  CUDA_VISIBLE_DEVICES=2 RUN_NAME=myrun $0
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi

print_header() {
    echo -e "${BLUE}=======================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=======================================================================${NC}"
}

run_cmd() {
    echo -e "${YELLOW}$ $*${NC}"
    "$@"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo -e "${RED}❌ Missing required file: $1${NC}"
        exit 1
    fi
}

mkdir -p "$STAGE1_DIR" "$STAGE2_DIR" "$STAGE3_DIR" "$(dirname "$EVAL_OUTPUT_DIR")"

require_file "$PYTHON_BIN"
require_file "$REPO_DIR/nullspace_projection_compute.py"
require_file "$REPO_DIR/qp_true_forward_fast.py"
require_file "$REPO_DIR/unified_model_merge.py"
require_file "$EVAL_SCRIPT"
require_file "$STAGE1_TEXTS"
require_file "$STAGE2_TEXTS"

print_header "End-to-End Merge Pipeline"
echo -e "${GREEN}Run name:${NC} $RUN_NAME"
echo -e "${GREEN}CUDA_VISIBLE_DEVICES:${NC} $CUDA_VISIBLE_DEVICES"
echo -e "${GREEN}Run root:${NC} $RUN_ROOT"
echo -e "${GREEN}Base model:${NC} $BASE_MODEL"
echo -e "${GREEN}Instruct model:${NC} $INSTRUCT_MODEL"
echo -e "${GREEN}Target model:${NC} $TARGET_MODEL"
echo -e "${GREEN}Stage 1 texts:${NC} $STAGE1_TEXTS"
echo -e "${GREEN}Stage 2 texts:${NC} $STAGE2_TEXTS"
echo -e "${GREEN}Eval data:${NC} $EVAL_DATA"
echo -e "${GREEN}Final merged model:${NC} $MERGED_MODEL_PATH"
echo -e "${GREEN}Eval output:${NC} $EVAL_OUTPUT_DIR"

print_header "Stage 1: Nullspace Projection + Lambda Search"
run_cmd "$PYTHON_BIN" "$REPO_DIR/nullspace_projection_compute.py" \
  --base "$BASE_MODEL" \
  --instruct "$INSTRUCT_MODEL" \
  --target "$TARGET_MODEL" \
  --texts_r "$STAGE1_TEXTS" \
  --max_samples_r "$MAX_SAMPLES" \
  --merge_types "$MERGE_TYPES" \
  --layers_tail "$LAYERS_TAIL" \
  --heads "$HEADS" \
  --compute_precision "$COMPUTE_PRECISION" \
  --lambda_ridge "$LAMBDA_RIDGE" \
  --cg_maxit "$CG_MAXIT" \
  --cg_tol "$CG_TOL" \
  --q_rows_per_text "$Q_ROWS" \
  --k_rows_per_text "$K_ROWS" \
  --v_rows_per_text "$V_ROWS" \
  --o_rows_per_text "$O_ROWS" \
  --ffn_rows_per_text "$FFN_ROWS" \
  --w_q "$W_Q" \
  --w_k "$W_K" \
  --w_v "$W_V" \
  --w_o "$W_O" \
  --w_ffn "$W_FFN" \
  --readout_dirs "$READOUT_DIRS" \
  --lambda_candidates "$LAMBDA_CANDIDATES" \
  --margin_retention_threshold "$MARGIN_RETENTION_THRESHOLD" \
  --lambda_search_max_new_tokens "$LAMBDA_SEARCH_MAX_NEW_TOKENS" \
  --lambda_generation_batch_size "$LAMBDA_GENERATION_BATCH_SIZE" \
  --lambda_margin_batch_size "$LAMBDA_MARGIN_BATCH_SIZE" \
  --qk_device "cuda:0" \
  --vo_device "cuda:0" \
  --ffn_device "cuda:0" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --projection_mode "$PROJECTION_MODE" \
  --use_hooks \
  --output_file "$PROJECTED_FILE"

require_file "$PROJECTED_FILE"

print_header "Stage 2: QP Alpha Optimization"
STAGE2_CMD=(
  "$PYTHON_BIN" "$REPO_DIR/qp_true_forward_fast.py"
  --projected_file "$PROJECTED_FILE"
  --base_model "$TARGET_MODEL"
  --json_data "$STAGE2_TEXTS"
  --layers all
  --heads all
  --prior_scalar "$PRIOR_SCALAR"
  --l2_prior "$L2_PRIOR"
  --l1 "$L1_REG"
  --box_lo "$BOX_LO"
  --box_hi "$BOX_HI"
  --device "$DEVICE"
  --out "$STAGE2_DIR"
  --qp_variant "$QP_VARIANT"
  --H_lambda "$H_LAMBDA"
  --H_mu "$H_MU"
  --rho_du "$RHO_DU"
  --verbose
)

if [[ "$QP_VARIANT" == "anchor_only" || "$QP_VARIANT" == "post_only" ]]; then
  STAGE2_CMD+=(--kappa_a "$KAPPA_A" --kappa_u "$KAPPA_U")
fi

run_cmd "${STAGE2_CMD[@]}"

if [[ ! -f "$ALPHA_FILE" ]]; then
    echo -e "${RED}❌ Stage 2 finished without producing $ALPHA_FILE${NC}"
    echo -e "${YELLOW}This usually means all layers were excluded by lambda filtering, so Stage 3/Eval are skipped.${NC}"
    exit 1
fi

print_header "Stage 3: Unified Merge"
run_cmd "$PYTHON_BIN" "$REPO_DIR/unified_model_merge.py" \
  --projected_file "$PROJECTED_FILE" \
  --base_model "$TARGET_MODEL" \
  --alpha_file "$ALPHA_FILE" \
  --scaling_factor "$SCALING_FACTOR" \
  --output_dir "$STAGE3_DIR" \
  --model_name "$MODEL_NAME" \
  --verbose

if [[ ! -d "$MERGED_MODEL_PATH" ]]; then
    echo -e "${RED}❌ Merged model directory was not created: $MERGED_MODEL_PATH${NC}"
    exit 1
fi

print_header "Eval"
run_cmd "$PYTHON_BIN" "$EVAL_SCRIPT" eval \
  --model_path "$MERGED_MODEL_PATH" \
  --model_name "$EVAL_MODEL_NAME" \
  --data_path "$EVAL_DATA" \
  --regime "$EVAL_REGIME" \
  --output_dir "$EVAL_OUTPUT_DIR" \
  --device "$EVAL_DEVICE" \
  --dtype "$EVAL_DTYPE" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --max_input_tokens "$EVAL_MAX_INPUT_TOKENS" \
  --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
  --seed "$EVAL_SEED" \
  --log_every "$EVAL_LOG_EVERY" \
  --strict_parse_mode "$STRICT_PARSE_MODE" \
  --early_stop_valid_rating \
  --length_bucket_batches

print_header "Done"
echo -e "${GREEN}Projected file:${NC} $PROJECTED_FILE"
echo -e "${GREEN}Alpha file:${NC} $ALPHA_FILE"
echo -e "${GREEN}Merged model:${NC} $MERGED_MODEL_PATH"
echo -e "${GREEN}Eval output:${NC} $EVAL_OUTPUT_DIR"
