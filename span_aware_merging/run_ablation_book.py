"""SPEAR Ablation Study — Book.

Phase 0: Wait for reczero val predictions, then run step0b to produce:
  - bias.pkl          (bias-projection only)
  - nullspace_bias.pkl (nullspace + bias)
  Combined with existing nullspace.pkl → 3 stage1 variants.

Then runs the same 4-round ablation as the Yelp study.

Round 1: projection variant  (nullspace / bias / nullspace_bias) — 3-way parallel
Round 2: fisher responsibility (True / False)
Round 3: delta_fmt grid        (0.0 / 0.5 / 0.7 / 0.9) — 3 new in parallel
Round 3b: weight formula       (paper / inverse)
Round 4: lambda grid           (0.35 / 0.4 / 0.45 / 0.5 / 0.55 / 0.6) — 3 at a time
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Fixed paths ────────────────────────────────────────────────────────────────
ABLATION_ROOT = "/data/uqlinh/rain_merging/span_aware_outputs_book/ablation"

STAGE1_DIR  = f"{ABLATION_ROOT}/stage1_output"
FISHER_DIR  = f"{ABLATION_ROOT}/fisher"
MASKS_DIR   = f"{ABLATION_ROOT}/masks"

STAGE1_PKLS = {
    "nullspace":      f"{STAGE1_DIR}/nullspace.pkl",
    "bias":           f"{STAGE1_DIR}/bias.pkl",
    "nullspace_bias": f"{STAGE1_DIR}/nullspace_bias.pkl",
}

MODEL_R_PATH    = "/data/uqlinh/reczero/checkpoints/book/actor/global_step_2000"
MODEL_B_PATH    = "/data/uqlinh/tallrec/output-book-20260418/final"

RECZERO_PREDS   = "/data/uqlinh/ReasonMerge/eval_results/book/reczero_ckpt2000/val/predictions.jsonl"
TALLREC_PREDS   = "/data/uqlinh/ReasonMerge/eval_results/book/val_set/tallrec_fixed/predictions.jsonl"
RECZERO_METRICS = "/data/uqlinh/ReasonMerge/eval_results/book/reczero_ckpt2000/val/metrics.json"

STEP0B   = "/home/uqlinh/RAIN-Merging/step0b_bias_projection.py"
TEST_DATA = "/data/uqlinh/reczero/data/book/test.parquet"
BENCH     = "/home/uqlinh/ReasonMerge/EXP3RT/theory_test/unified_benchmark.py"

GPU_ID  = "1"        # fixed to GPU 1
DEVICE  = "cuda:0"   # CUDA_VISIBLE_DEVICES remaps physical GPU to index 0

PYTHON     = sys.executable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_JSONL = os.path.join(ABLATION_ROOT, "ablation_results.jsonl")

# Book-specific weights (from existing coefficients meta)
W_USER  = 0.392145
W_ITEM  = 0.381171
W_MATCH = 0.226684

# Book JSD importance values (from jsd_importance/human_readable_report.txt)
_SPAN_IMPORTANCE_BOOK = {"user": 0.008158, "item": 0.008988, "match": 0.020674}

POLL_SEC     = 180    # 3 min between reczero-metrics checks
MIN_FREE_MB  = 16384  # 16 GB (bfloat16 needs ~14 GB)
GPU_POLL_SEC = 180    # 3 min between GPU checks
GPU_CANDIDATES = ["1"]

# ── Helpers ────────────────────────────────────────────────────────────────────


def _gpu_free_mb(gpu_id: str) -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits", "-i", gpu_id],
            stderr=subprocess.DEVNULL,
        )
        return int(out.decode().strip())
    except Exception:
        return 0


def pick_gpu() -> str:
    """Poll GPU_CANDIDATES every GPU_POLL_SEC until one has >= MIN_FREE_MB free.
    Returns the winning GPU id and sets the global GPU_ID."""
    global GPU_ID
    while True:
        for gid in GPU_CANDIDATES:
            free = _gpu_free_mb(gid)
            print(f"[{time.strftime('%F %T')}] GPU {gid} free: {free} MiB", flush=True)
            if free >= MIN_FREE_MB:
                GPU_ID = gid
                print(f"[{time.strftime('%F %T')}] Using GPU {GPU_ID} ({free} MiB free)", flush=True)
                return GPU_ID
        print(f"[{time.strftime('%F %T')}] No GPU has {MIN_FREE_MB} MiB free, "
              f"retrying in {GPU_POLL_SEC}s ...", flush=True)
        time.sleep(GPU_POLL_SEC)

def script(name):
    return os.path.join(SCRIPT_DIR, name)


def run(cmd, label=""):
    tag = f"[{label}] " if label else ""
    print(f"\n{tag}>>> {' '.join(str(x) for x in cmd)}", flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    result = subprocess.run(cmd, check=False, env=env)
    elapsed = time.time() - t0
    print(f"{tag}finished in {elapsed/60:.1f}m (rc={result.returncode})", flush=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def run_parallel(fns, max_workers=3):
    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn): i for i, fn in enumerate(fns)}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                errors.append(e)
    if errors:
        raise errors[0]


def cfg_json(extra: dict) -> str:
    base = {
        "model_r_path":  MODEL_R_PATH,
        "model_b_path":  MODEL_B_PATH,
        "fisher_dir":    FISHER_DIR,
        "masks_dir":     MASKS_DIR,
        "device":        DEVICE,
        "dtype_str":     "bfloat16",
        "w_user":        W_USER,
        "w_item":        W_ITEM,
        "w_match":       W_MATCH,
        "USE_JSD_WEIGHTS":          True,
        "USE_ATTENTION_MODULATION": True,
        "USE_FORMAT_SUPPRESSION":   True,
    }
    base.update(extra)
    return json.dumps(base)


def dirs(proj, fisher_resp, fmt_tag=None):
    fr = "frTrue" if fisher_resp else "frFalse"
    attn_dir  = f"{ABLATION_ROOT}/attention/attn_{proj}"
    resp_dir  = f"{ABLATION_ROOT}/responsibility/resp_{proj}_{fr}"
    fmt_dir   = f"{ABLATION_ROOT}/format_suppression/fmt_{proj}_{fr}_fmt{fmt_tag}" if fmt_tag else None
    eval_dir  = f"{ABLATION_ROOT}/eval_results"
    return dict(
        attention_dir=attn_dir,
        responsibility_dir=resp_dir,
        floors_dir=fmt_dir,
        eval_dir=eval_dir,
    )


def ensure_dirs(*paths):
    for p in paths:
        if p:
            os.makedirs(p, exist_ok=True)


def read_rmse(eval_out_dir):
    metrics_path = os.path.join(eval_out_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"WARNING: metrics.json not found at {metrics_path}", flush=True)
        return float("inf")
    with open(metrics_path) as f:
        m = json.load(f)
    return float(m.get("rmse", float("inf")))


def append_result(record: dict):
    with open(RESULTS_JSONL, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Result: {record}", flush=True)


# ── Phase 0: wait for reczero metrics, then build pkl files ───────────────────

def phase0_build_pkls():
    # Wait for reczero val to finish
    if not os.path.exists(RECZERO_METRICS):
        print(f"Waiting for reczero val metrics at {RECZERO_METRICS} ...", flush=True)
        while not os.path.exists(RECZERO_METRICS):
            time.sleep(POLL_SEC)
            print(f"[{time.strftime('%F %T')}] still waiting...", flush=True)
    print(f"reczero val metrics found — proceeding.", flush=True)

    os.makedirs(STAGE1_DIR, exist_ok=True)

    # Build bias.pkl (bias-projection of nullspace)
    if not os.path.exists(STAGE1_PKLS["bias"]):
        print("\n--- Phase 0a: building bias.pkl ---", flush=True)
        run([
            PYTHON, STEP0B,
            "--input_pkl",            STAGE1_PKLS["nullspace"],
            "--model_r_path",         MODEL_R_PATH,
            "--reczero_predictions",  RECZERO_PREDS,
            "--tallrec_predictions",  TALLREC_PREDS,
            "--output_pkl",           STAGE1_PKLS["bias"],
            "--output_dir",           STAGE1_DIR,
            "--skip_tallrec_validation",
            "--device",               "cuda:0",
        ], label="step0b/bias")
    else:
        print(f"bias.pkl already exists, skipping.", flush=True)

    # Build nullspace_bias.pkl (nullspace then bias projection)
    if not os.path.exists(STAGE1_PKLS["nullspace_bias"]):
        print("\n--- Phase 0b: building nullspace_bias.pkl ---", flush=True)
        run([
            PYTHON, STEP0B,
            "--input_pkl",            STAGE1_PKLS["nullspace"],
            "--model_r_path",         MODEL_R_PATH,
            "--reczero_predictions",  RECZERO_PREDS,
            "--tallrec_predictions",  TALLREC_PREDS,
            "--output_pkl",           STAGE1_PKLS["nullspace_bias"],
            "--output_dir",           STAGE1_DIR,
            "--skip_tallrec_validation",
            "--device",               "cuda:0",
        ], label="step0b/nullspace_bias")
    else:
        print(f"nullspace_bias.pkl already exists, skipping.", flush=True)


# ── Step runners ───────────────────────────────────────────────────────────────

def run_step3(proj):
    attn_dir = dirs(proj, fisher_resp=True)["attention_dir"]
    marker = os.path.join(attn_dir, "attention_meta.json")
    if os.path.exists(marker):
        print(f"[step3/{proj}] already done, skipping.", flush=True)
        return
    ensure_dirs(attn_dir)
    extra = {
        "stage1_delta_path": STAGE1_PKLS[proj],
        "attention_dir":     attn_dir,
    }
    run([PYTHON, script("step3_attention.py"), "--override_json", cfg_json(extra)],
        label=f"step3/{proj}")


def run_step4(proj, fisher_resp):
    fr = "frTrue" if fisher_resp else "frFalse"
    d = dirs(proj, fisher_resp)
    resp_dir = d["responsibility_dir"]
    marker = os.path.join(resp_dir, "resp_q.pt")
    if os.path.exists(marker):
        print(f"[step4/{proj}/{fr}] already done, skipping.", flush=True)
        return
    ensure_dirs(resp_dir)
    extra = {
        "stage1_delta_path":         STAGE1_PKLS[proj],
        "attention_dir":             d["attention_dir"],
        "responsibility_dir":        resp_dir,
        "USE_FISHER_RESPONSIBILITY": fisher_resp,
    }
    run([PYTHON, script("step4_responsibility.py"), "--override_json", cfg_json(extra)],
        label=f"step4/{proj}/{fr}")


def run_step5(proj, fisher_resp, delta_fmt):
    fr = "frTrue" if fisher_resp else "frFalse"
    fmt_tag = f"{delta_fmt:.2f}".replace(".", "")
    d = dirs(proj, fisher_resp, fmt_tag)
    fmt_dir = d["floors_dir"]
    marker = os.path.join(fmt_dir, "fisher_supp_tag_q.pt")
    if os.path.exists(marker):
        print(f"[step5/{proj}/{fr}/fmt{fmt_tag}] already done, skipping.", flush=True)
        return
    ensure_dirs(fmt_dir)
    extra = {
        "stage1_delta_path": STAGE1_PKLS[proj],
        "attention_dir":     d["attention_dir"],
        "floors_dir":        fmt_dir,
        "delta_fmt":         delta_fmt,
    }
    run([PYTHON, script("step5_format_suppression.py"), "--override_json", cfg_json(extra)],
        label=f"step5/{proj}/{fr}/fmt{fmt_tag}")


def run_step6(proj, fisher_resp, delta_fmt, weight_formula=None):
    fr = "frTrue" if fisher_resp else "frFalse"
    fmt_tag = f"{delta_fmt:.2f}".replace(".", "")
    w_suffix = f"_w{weight_formula}" if weight_formula else ""
    coeff_dir = f"{ABLATION_ROOT}/coefficients/coeff_{proj}_{fr}_fmt{fmt_tag}{w_suffix}"
    d = dirs(proj, fisher_resp, fmt_tag)
    marker = os.path.join(coeff_dir, "alpha_q.pt")
    if os.path.exists(marker):
        print(f"[step6/{proj}/{fr}/fmt{fmt_tag}{w_suffix}] already done, skipping.", flush=True)
        return coeff_dir
    ensure_dirs(coeff_dir)
    extra = {
        "stage1_delta_path":  STAGE1_PKLS[proj],
        "attention_dir":      d["attention_dir"],
        "responsibility_dir": d["responsibility_dir"],
        "floors_dir":         d["floors_dir"],
        "coefficients_dir":   coeff_dir,
    }
    cmd = [PYTHON, script("step6_coefficients_simple.py"), "--override_json", cfg_json(extra)]
    if weight_formula:
        cmd += ["--weight_formula", weight_formula,
                "--span_importance_json", json.dumps(_SPAN_IMPORTANCE_BOOK)]
    run(cmd, label=f"step6/{proj}/{fr}/fmt{fmt_tag}{w_suffix}")
    return coeff_dir


def run_step7_and_eval(proj, fisher_resp, delta_fmt, lam, weight_formula=None, coeff_dir=None):
    fr = "frTrue" if fisher_resp else "frFalse"
    fmt_tag = f"{delta_fmt:.2f}".replace(".", "")
    lam_tag = f"{lam:.2f}".replace(".", "")
    w_suffix = f"_w{weight_formula}" if weight_formula else ""

    if coeff_dir is None:
        coeff_dir = f"{ABLATION_ROOT}/coefficients/coeff_{proj}_{fr}_fmt{fmt_tag}{w_suffix}"
    ckpt_dir = f"{ABLATION_ROOT}/merged_models/ckpt_{proj}_{fr}_fmt{fmt_tag}{w_suffix}"

    lam_str = f"{lam:.2f}".replace(".", "_")
    fmt_str = f"{delta_fmt:.2f}".replace(".", "_")
    merged_path = os.path.join(ckpt_dir, f"merged_lambda_{lam_str}_delta_fmt_{fmt_str}")

    run_name     = f"ablation_book_{proj}_{fr}_fmt{fmt_tag}{w_suffix}_l{lam_tag}"
    eval_out_dir = os.path.join(f"{ABLATION_ROOT}/eval_results", run_name)
    metrics_path = os.path.join(eval_out_dir, "metrics.json")

    if os.path.exists(metrics_path):
        print(f"[eval/{run_name}] already done, skipping.", flush=True)
        return read_rmse(eval_out_dir)

    ensure_dirs(ckpt_dir, eval_out_dir)

    if not os.path.exists(os.path.join(merged_path, "config.json")):
        pick_gpu()
        extra = {
            "stage1_delta_path": STAGE1_PKLS[proj],
            "coefficients_dir":  coeff_dir,
            "checkpoint_dir":    ckpt_dir,
            "delta_fmt":         delta_fmt,
            "lambda_global":     lam,
        }
        run([PYTHON, script("step7_merge.py"),
             "--override_json", cfg_json(extra),
             "--lambda_global", str(lam)],
            label=f"step7/{proj}/{fr}/fmt{fmt_tag}{w_suffix}/l{lam_tag}")
    else:
        print(f"[step7/{proj}/{fr}/fmt{fmt_tag}{w_suffix}/l{lam_tag}] merged model exists, skipping.", flush=True)

    run([PYTHON, "-u", BENCH, "eval",
         "--model_path",        merged_path,
         "--model_name",        run_name,
         "--data_path",         TEST_DATA,
         "--regime",            "native_reczero",
         "--output_dir",        eval_out_dir,
         "--device",            "cuda",
         "--dtype",             "bfloat16",
         "--batch_size",        "8",
         "--max_input_tokens",  "6000",
         "--max_new_tokens",    "2000",
         "--seed",              "1234",
         "--log_every",         "1",
         "--strict_parse_mode", "rate_tag_only",
         ],
        label=f"eval/{proj}/{fr}/fmt{fmt_tag}{w_suffix}/l{lam_tag}")

    rmse = read_rmse(eval_out_dir)
    append_result({
        "proj": proj, "fisher_resp": fisher_resp,
        "delta_fmt": delta_fmt, "lambda": lam,
        "weight_formula": weight_formula, "run_name": run_name, "rmse": rmse,
    })
    return rmse


def pipeline_456_eval(proj, fisher_resp, delta_fmt, lam, round_name, weight_formula=None):
    run_step4(proj, fisher_resp)
    run_step5(proj, fisher_resp, delta_fmt)
    coeff_dir = run_step6(proj, fisher_resp, delta_fmt, weight_formula=weight_formula)
    rmse = run_step7_and_eval(proj, fisher_resp, delta_fmt, lam,
                              weight_formula=weight_formula, coeff_dir=coeff_dir)
    append_result({
        "round": round_name, "proj": proj, "fisher_resp": fisher_resp,
        "delta_fmt": delta_fmt, "lambda": lam,
        "weight_formula": weight_formula, "rmse": rmse,
    })
    return rmse


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_round", type=str, default="0",
                        choices=["0", "1", "2", "3", "3b", "4"],
                        help="Resume from: 0=phase0+pkls, 1-4=ablation rounds")
    parser.add_argument("--best_proj",   type=str, default=None)
    parser.add_argument("--best_fisher", type=str, default=None)
    parser.add_argument("--best_fmt",    type=float, default=None)
    parser.add_argument("--best_weight", type=str, default=None,
                        choices=["paper", "inverse"])
    args = parser.parse_args()

    os.makedirs(ABLATION_ROOT, exist_ok=True)

    round_order = ["0", "1", "2", "3", "3b", "4"]
    def _from(r):
        return round_order.index(args.start_round) <= round_order.index(r)

    # ── Phase 0: wait + build pkls ────────────────────────────────────────────
    if _from("0"):
        phase0_build_pkls()

    # ── Pick GPU before starting GPU-heavy work ───────────────────────────────
    print("\n--- Waiting for a free GPU (>=24 GB) on GPU 1 ---")
    pick_gpu()

    # ── Round 1: projection variant ───────────────────────────────────────────
    projs = ("nullspace", "bias", "nullspace_bias")

    if _from("1") and args.best_proj is None:
        print("\n" + "="*60)
        print("ROUND 1 — Stage 1 projection variant  (3-way parallel)")
        print("="*60)

        print("\n--- Step 3: attention (parallel) ---")
        run_parallel([lambda p=p: run_step3(p) for p in projs])

        print("\n--- Steps 4-7 + eval (2 at a time) ---")
        r1_results = {}

        def _r1_job(proj):
            rmse = pipeline_456_eval(proj, fisher_resp=True, delta_fmt=0.9,
                                     lam=0.5, round_name="round1")
            r1_results[proj] = rmse

        run_parallel([lambda p=p: _r1_job(p) for p in projs], max_workers=2)

        best_proj = min(r1_results, key=r1_results.get)
        for p in projs:
            print(f"  {p}: RMSE={r1_results[p]:.4f}")
        print(f"\nRound 1 winner: {best_proj}  RMSE={r1_results[best_proj]:.4f}")
        append_result({"round": "round1_summary", "best_proj": best_proj,
                       "results": r1_results})
    else:
        best_proj = args.best_proj
        print(f"\nSkipping Round 1 — using BEST_PROJ={best_proj}")

    # ── Round 2: fisher responsibility ────────────────────────────────────────
    if _from("2") and args.best_fisher is None:
        print("\n" + "="*60)
        print(f"ROUND 2 — Fisher responsibility  (proj={best_proj})")
        print("="*60)

        fr_true_run  = f"ablation_book_{best_proj}_frTrue_fmt09_l05"
        rmse_true  = read_rmse(os.path.join(ABLATION_ROOT, "eval_results", fr_true_run))
        rmse_false = pipeline_456_eval(best_proj, fisher_resp=False, delta_fmt=0.9,
                                       lam=0.5, round_name="round2")

        best_fisher = rmse_true <= rmse_false
        print(f"  frTrue:  RMSE={rmse_true:.4f}")
        print(f"  frFalse: RMSE={rmse_false:.4f}")
        print(f"\nRound 2 winner: fisher_resp={best_fisher}")
        append_result({"round": "round2_summary", "best_proj": best_proj,
                       "best_fisher_resp": best_fisher,
                       "results": {"frTrue": rmse_true, "frFalse": rmse_false}})
    else:
        best_fisher = (args.best_fisher.lower() != "false") if args.best_fisher else True
        print(f"\nSkipping Round 2 — using BEST_FISHER={best_fisher}")

    # ── Round 3: delta_fmt grid ───────────────────────────────────────────────
    if _from("3") and args.best_fmt is None:
        print("\n" + "="*60)
        print(f"ROUND 3 — Format suppression  (proj={best_proj}, fr={best_fisher})")
        print("="*60)

        fr_tag   = "frTrue" if best_fisher else "frFalse"
        done_run = f"ablation_book_{best_proj}_{fr_tag}_fmt09_l05"
        r3_results = {0.9: read_rmse(os.path.join(ABLATION_ROOT, "eval_results", done_run))}

        new_fmts = (0.0, 0.5, 0.7)
        print(f"\n--- delta_fmt in {new_fmts} (2 at a time) ---")

        def _r3_job(dfmt):
            rmse = pipeline_456_eval(best_proj, fisher_resp=best_fisher, delta_fmt=dfmt,
                                     lam=0.5, round_name="round3")
            r3_results[dfmt] = rmse

        run_parallel([lambda d=d: _r3_job(d) for d in new_fmts], max_workers=2)

        for dfmt in new_fmts:
            print(f"  delta_fmt={dfmt}: RMSE={r3_results[dfmt]:.4f}")
        print(f"  delta_fmt=0.9:  RMSE={r3_results[0.9]:.4f}  (reused)")

        best_fmt = min(r3_results, key=r3_results.get)
        print(f"\nRound 3 winner: delta_fmt={best_fmt}  RMSE={r3_results[best_fmt]:.4f}")
        append_result({"round": "round3_summary", "best_proj": best_proj,
                       "best_fisher_resp": best_fisher, "best_fmt": best_fmt,
                       "results": {str(k): v for k, v in r3_results.items()}})
    else:
        best_fmt = args.best_fmt if args.best_fmt is not None else 0.9
        print(f"\nSkipping Round 3 — using BEST_FMT={best_fmt}")

    # ── Round 3b: weight formula ──────────────────────────────────────────────
    if _from("3b") and args.best_weight is None:
        print("\n" + "="*60)
        print(f"ROUND 3b — Weight formula  (proj={best_proj}, fr={best_fisher}, fmt={best_fmt})")
        print("="*60)

        fr_tag  = "frTrue" if best_fisher else "frFalse"
        fmt_tag = f"{best_fmt:.2f}".replace(".", "")
        paper_run = f"ablation_book_{best_proj}_{fr_tag}_fmt{fmt_tag}_l05"
        paper_eval_dir = os.path.join(ABLATION_ROOT, "eval_results", paper_run)
        rmse_paper = read_rmse(paper_eval_dir)

        coeff_dir_inv = run_step6(best_proj, best_fisher, best_fmt, weight_formula="inverse")
        rmse_inverse  = run_step7_and_eval(best_proj, best_fisher, best_fmt, lam=0.5,
                                           weight_formula="inverse", coeff_dir=coeff_dir_inv)

        r3b = {"paper": rmse_paper, "inverse": rmse_inverse}
        best_weight = min(r3b, key=r3b.get)
        print(f"  paper (reused): RMSE={rmse_paper:.4f}")
        print(f"  inverse:        RMSE={rmse_inverse:.4f}")
        print(f"\nRound 3b winner: weight_formula={best_weight}  RMSE={r3b[best_weight]:.4f}")
        append_result({"round": "round3b_summary", "best_proj": best_proj,
                       "best_fisher_resp": best_fisher, "best_fmt": best_fmt,
                       "best_weight": best_weight, "results": r3b})
    else:
        best_weight = args.best_weight if args.best_weight else "paper"
        print(f"\nSkipping Round 3b — using BEST_WEIGHT={best_weight}")

    # ── Round 4: lambda grid ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"ROUND 4 — Lambda  (proj={best_proj}, fr={best_fisher}, fmt={best_fmt}, w={best_weight})")
    print("="*60)

    wf_r4 = best_weight if best_weight != "paper" else None
    coeff_dir_r4 = run_step6(best_proj, best_fisher, best_fmt,
                             weight_formula=wf_r4)

    r4_results = {}
    print("\n--- lambda in (0.35, 0.4, 0.45, 0.5, 0.55, 0.6) step7+eval (3 at a time) ---")

    def _r4_job(lam):
        rmse = run_step7_and_eval(best_proj, best_fisher, best_fmt, lam,
                                  weight_formula=wf_r4, coeff_dir=coeff_dir_r4)
        r4_results[lam] = rmse

    run_parallel([lambda l=l: _r4_job(l) for l in (0.35, 0.4, 0.45, 0.5, 0.55, 0.6)], max_workers=2)

    for lam in (0.35, 0.4, 0.45, 0.5, 0.55, 0.6):
        print(f"  lambda={lam}: RMSE={r4_results[lam]:.4f}")

    best_lam = min(r4_results, key=r4_results.get)
    print(f"\nRound 4 winner: lambda={best_lam}  RMSE={r4_results[best_lam]:.4f}")
    append_result({"round": "round4_summary",
                   "best_proj": best_proj, "best_fisher_resp": best_fisher,
                   "best_fmt": best_fmt, "best_weight": best_weight,
                   "best_lambda": best_lam,
                   "results": {str(k): v for k, v in sorted(r4_results.items())}})

    print("\n" + "="*60)
    print("FINAL BEST CONFIGURATION")
    print(f"  projection:     {best_proj}")
    print(f"  fisher_resp:    {best_fisher}")
    print(f"  delta_fmt:      {best_fmt}")
    print(f"  weight_formula: {best_weight}")
    print(f"  lambda:         {best_lam}")
    print(f"  RMSE:           {r4_results[best_lam]:.4f}")
    print(f"\nAll results: {RESULTS_JSONL}")
    print("="*60)


if __name__ == "__main__":
    main()
