#!/usr/bin/env python3
"""
End-to-end pipeline: nullspace projection → Fisher → attention → merge → eval
Target checkpoint: /data/uqlinh/merged_models/music/music_ta_align_rho07_rcp_soft_tallrec
Target eval:       /data/uqlinh/merged_models/music/eval_results/music_ta_align_rho07_rcp_soft_tallrec

All steps are idempotent — skipped if output already exists.
Run on GPU 1: CUDA_VISIBLE_DEVICES=1 python run_music_rho07_rcp_soft_pipeline.py
"""
import json
import os
import subprocess
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
ABLATION_ROOT = "/data/uqlinh/rain_merging/span_aware_outputs_music/ablation"
REPO_DIR      = "/home/uqlinh/RAIN-Merging"
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))

PYTHON    = sys.executable
GPU_ID    = "1"       # physical GPU; CUDA_VISIBLE_DEVICES remaps it to cuda:0

# Models
BASE_MODEL    = "/data/uqlinh/merged_models/Qwen2.5-3B-Instruct"
RECZERO_MODEL = "/data/uqlinh/reczero-music-ckpt"
TALLREC_MODEL = "/data/uqlinh/tallrec/output-music-fixed/final"

# Calibration data
STAGE1_DATA = "/home/uqlinh/RAIN-Merging/data/music/music_reason_correct_direct_fail_full_subset.jsonl"
DR_PATH     = "/home/uqlinh/RAIN-Merging/data/music/music_reason_correct_direct_fail_full_subset.jsonl"
MASKS_DIR   = "/data/uqlinh/rain_merging/span_aware_outputs_music/masks"

# Stage 1 output
STAGE1_DIR   = f"{ABLATION_ROOT}/stage1_output"
PROJECTED_PKL = f"{STAGE1_DIR}/projected_task_vectors.pkl"

# Step 2 / 3 outputs
FISHER_DIR   = f"{ABLATION_ROOT}/fisher"
ATTENTION_DIR = f"{ABLATION_ROOT}/attention"
A_ALIGN_PATH  = f"{ATTENTION_DIR}/a_align.pt"

# Final checkpoint + eval
OUTPUT_CKPT  = "/data/uqlinh/merged_models/music/music_ta_align_rho07_rcp_soft_tallrec"
MERGE_SCRIPT = "/home/uqlinh/ReasonMerge/Long-to-Short-via-Model-Merging/run_merge_rho07_rcp_soft_music.py"

EVAL_SCRIPT  = "/home/uqlinh/ReasonMerge/EXP3RT/theory_test/unified_benchmark.py"
EVAL_DATA    = "/home/uqlinh/ReasonMerge/RecZero-main/data/test_reczero/music/test.parquet"
EVAL_OUT_DIR = "/data/uqlinh/merged_models/music/eval_results/music_ta_align_rho07_rcp_soft_tallrec"
MODEL_NAME   = "music_ta_align_rho07_rcp_soft_tallrec"

# step2_fisher config
STEP2_CFG = {
    "model_r_path":    RECZERO_MODEL,
    "model_b_path":    TALLREC_MODEL,
    "fisher_dir":      FISHER_DIR,
    "masks_dir":       MASKS_DIR,
    "dr_path":         DR_PATH,
    "stage1_delta_path": PROJECTED_PKL,
    "device":          "cuda:0",
    "dtype_str":       "bfloat16",
    "delta_fmt":       0.0,
    "delta_coh":       0.0,
    "USE_FISHER_RESPONSIBILITY": False,
    "fisher_log_every": 10,
}

# step3_attention config
STEP3_CFG = {
    **STEP2_CFG,
    "attention_dir": ATTENTION_DIR,
}


def log(msg: str) -> None:
    print(f"\n{'='*60}\n{msg}\n{'='*60}", flush=True)


def run(cmd: list, label: str = "") -> None:
    if label:
        log(label)
    print("$", " ".join(str(x) for x in cmd), flush=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU_ID}
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def skip(path: str, label: str) -> bool:
    if os.path.exists(path):
        print(f"[SKIP] {label} — already exists: {path}", flush=True)
        return True
    return False


# ── Step 1: Nullspace projection ───────────────────────────────────────────────
os.makedirs(STAGE1_DIR, exist_ok=True)
if not skip(PROJECTED_PKL, "Stage 1 nullspace projection"):
    run([
        PYTHON, f"{REPO_DIR}/nullspace_projection_compute.py",
        "--base",      BASE_MODEL,
        "--instruct",  TALLREC_MODEL,
        "--target",    RECZERO_MODEL,
        "--texts_r",   STAGE1_DATA,
        "--max_samples_r", "100",
        "--merge_types", "qkvof",
        "--layers_tail", "27",
        "--heads",     "all",
        "--compute_precision", "fp32",
        "--lambda_ridge", "1e-4",
        "--cg_maxit",  "100",
        "--cg_tol",    "1e-5",
        "--q_rows_per_text", "8",
        "--k_rows_per_text", "8",
        "--v_rows_per_text", "4",
        "--o_rows_per_text", "4",
        "--ffn_rows_per_text", "4",
        "--readout_dirs", "2",
        "--max_seq_len", "7168",
        "--projection_mode", "dual_ab",
        "--use_hooks",
        "--output_file", PROJECTED_PKL,
    ], label="Stage 1: Nullspace Projection")

# ── Step 2: Fisher sensitivity ─────────────────────────────────────────────────
os.makedirs(FISHER_DIR, exist_ok=True)
fisher_sentinel = f"{FISHER_DIR}/fisher_norm_user_q.pt"
if not skip(fisher_sentinel, "Step 2 Fisher"):
    run([
        PYTHON, os.path.join(SCRIPT_DIR, "step2_fisher.py"),
        "--override_json", json.dumps(STEP2_CFG),
    ], label="Step 2: Fisher Sensitivity")

# ── Step 3: Attention alignment ────────────────────────────────────────────────
os.makedirs(ATTENTION_DIR, exist_ok=True)
if not skip(A_ALIGN_PATH, "Step 3 Attention"):
    run([
        PYTHON, os.path.join(SCRIPT_DIR, "step3_attention.py"),
        "--override_json", json.dumps(STEP3_CFG),
    ], label="Step 3: Attention Alignment")

# ── Merge: rho=0.70 + RCP soft suppression ────────────────────────────────────
ckpt_sentinel = os.path.join(OUTPUT_CKPT, "config.json")
if not skip(ckpt_sentinel, "Merge"):
    run([
        PYTHON, MERGE_SCRIPT,
    ], label="Merge: rho07 + RCP soft suppression → checkpoint")

# ── Eval ───────────────────────────────────────────────────────────────────────
os.makedirs(EVAL_OUT_DIR, exist_ok=True)
eval_sentinel = os.path.join(EVAL_OUT_DIR, "metrics.json")
if not skip(eval_sentinel, "Eval"):
    run([
        PYTHON, EVAL_SCRIPT, "eval",
        "--model_path",         OUTPUT_CKPT,
        "--model_name",         MODEL_NAME,
        "--data_path",          EVAL_DATA,
        "--regime",             "native_reczero",
        "--output_dir",         EVAL_OUT_DIR,
        "--device",             "cuda",
        "--dtype",              "bfloat16",
        "--batch_size",         "8",
        "--max_input_tokens",   "6000",
        "--max_new_tokens",     "2000",
        "--seed",               "1234",
        "--log_every",          "25",
        "--strict_parse_mode",  "rate_tag_only",
        "--resume_append",
    ], label="Eval")

log("Pipeline complete")
print(f"  Checkpoint : {OUTPUT_CKPT}")
print(f"  Eval output: {EVAL_OUT_DIR}")
