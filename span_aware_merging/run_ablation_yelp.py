"""SPEAR Ablation Study — Yelp.

Round 1:  projection variant  (nullspace / bias / nullspace_bias) — 3-way parallel
Round 2:  fisher responsibility (True / False)  — reuses Round 1 attention
Round 2b: fisher beta dampening (0.25 / 0.5 / 0.75) — only when best_fisher=True,
          reuses Round 1 attention, runs step4_responsibility_beta.py in parallel
Round 3:  delta_fmt grid        (0.0 / 0.5 / 0.7 / 0.9) — reuses Round 1-2 resp
Round 4:  lambda grid           (0.4 / 0.5 / 0.6)        — reuses Round 3 coeff

Round 1 parallelism:
  - all 3 step3_attention jobs run concurrently
  - all 3 step4+5+6 jobs run concurrently (CPU-only, no GPU conflict)
  - all 3 step7+eval jobs run concurrently

All outputs land under ABLATION_ROOT.
Results are appended to ablation_results.jsonl after each eval.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Fixed paths ────────────────────────────────────────────────────────────────
ABLATION_ROOT = "/data/uqlinh/rain_merging/span_aware_outputs_yelp/ablation"

STAGE1_PKLS = {
    "nullspace":      f"{ABLATION_ROOT}/stage1_output_yelp/nullspace_task_vectors.pkl",
    "bias":           f"{ABLATION_ROOT}/stage1_output_yelp/bias_projected_yelp.pkl",
    "nullspace_bias": f"{ABLATION_ROOT}/stage1_output_yelp/nullspace_bias.pkl",
}

FISHER_DIR  = f"{ABLATION_ROOT}/fisher"
MASKS_DIR   = "/data/uqlinh/rain_merging/span_aware_outputs_yelp/masks"

MODEL_R_PATH = "/data/uqlinh/reczero-yelp-ckpt/global_step_320"
MODEL_B_PATH = "/data/uqlinh/tallrec/output-yelp-clean-20260418/final"

TEST_DATA = "/data/uqlinh/ReasonMerge/data/yelp/test/test_reczero.parquet"
BENCH     = "/home/uqlinh/ReasonMerge/EXP3RT/theory_test/unified_benchmark.py"

GPU_ID    = "0"
DEVICE    = "cuda:0"   # CUDA_VISIBLE_DEVICES remaps physical GPU to index 0

MIN_FREE_MB  = 16384  # 16 GB (bfloat16 needs ~14 GB)
GPU_POLL_SEC = 180

PYTHON     = sys.executable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_JSONL = os.path.join(ABLATION_ROOT, "ablation_results.jsonl")

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


def pick_gpu():
    global GPU_ID
    while True:
        free = _gpu_free_mb(GPU_ID)
        print(f"[{time.strftime('%F %T')}] GPU {GPU_ID} free: {free} MiB", flush=True)
        if free >= MIN_FREE_MB:
            print(f"[{time.strftime('%F %T')}] GPU {GPU_ID} ready ({free} MiB free)", flush=True)
            return
        print(f"[{time.strftime('%F %T')}] GPU {GPU_ID} not ready, retrying in {GPU_POLL_SEC}s ...", flush=True)
        time.sleep(GPU_POLL_SEC)


def script(name):
    return os.path.join(SCRIPT_DIR, name)


def run(cmd, label="", env_extra=None):
    tag = f"[{label}] " if label else ""
    print(f"\n{tag}>>> {' '.join(str(x) for x in cmd)}", flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(cmd, check=False, env=env)
    elapsed = time.time() - t0
    print(f"{tag}finished in {elapsed/60:.1f}m (rc={result.returncode})", flush=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def run_parallel(fns, max_workers=3):
    """Run a list of zero-arg callables concurrently, raise on first failure."""
    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn): fn.__name__ if hasattr(fn, "__name__") else str(i)
                   for i, fn in enumerate(fns)}
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
        "w_user":        0.4078,
        "w_item":        0.4156,
        "w_match":       0.1766,
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
    coeff_dir = f"{ABLATION_ROOT}/coefficients/coeff_{proj}_{fr}_fmt{fmt_tag}" if fmt_tag else None
    ckpt_dir  = f"{ABLATION_ROOT}/merged_models/ckpt_{proj}_{fr}_fmt{fmt_tag}" if fmt_tag else None
    eval_dir  = f"{ABLATION_ROOT}/eval_results"
    return dict(
        attention_dir=attn_dir,
        responsibility_dir=resp_dir,
        floors_dir=fmt_dir,
        coefficients_dir=coeff_dir,
        checkpoint_dir=ckpt_dir,
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


def dirs_beta(proj, beta):
    """Directory names for a beta-dampened responsibility variant."""
    beta_tag = f"{beta:.2f}".replace(".", "")
    attn_dir  = f"{ABLATION_ROOT}/attention/attn_{proj}"   # reuses Round 1 attention
    resp_dir  = f"{ABLATION_ROOT}/responsibility/resp_{proj}_frBeta{beta_tag}"
    return dict(
        attention_dir=attn_dir,
        responsibility_dir=resp_dir,
    )


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


def run_step4_beta(proj, beta):
    """Run step4_responsibility_beta.py for a given fisher_beta value."""
    d = dirs_beta(proj, beta)
    resp_dir = d["responsibility_dir"]
    marker = os.path.join(resp_dir, "resp_q.pt")
    beta_tag = f"{beta:.2f}".replace(".", "")
    if os.path.exists(marker):
        print(f"[step4_beta/{proj}/b{beta_tag}] already done, skipping.", flush=True)
        return
    ensure_dirs(resp_dir)
    extra = {
        "stage1_delta_path":         STAGE1_PKLS[proj],
        "attention_dir":             d["attention_dir"],
        "responsibility_dir":        resp_dir,
        "USE_FISHER_RESPONSIBILITY": True,
        "FISHER_BETA":               beta,
    }
    run([PYTHON, script("step4_responsibility_beta.py"), "--override_json", cfg_json(extra)],
        label=f"step4_beta/{proj}/b{beta_tag}")


def pipeline_beta_456_eval(proj, beta, delta_fmt, lam):
    """Steps 4(beta)-5-6-7-eval for a single beta variant."""
    d = dirs_beta(proj, beta)
    beta_tag = f"{beta:.2f}".replace(".", "")
    fmt_tag  = f"{delta_fmt:.2f}".replace(".", "")
    lam_tag  = f"{lam:.2f}".replace(".", "")

    resp_dir  = d["responsibility_dir"]
    fmt_dir   = f"{ABLATION_ROOT}/format_suppression/fmt_{proj}_frBeta{beta_tag}_fmt{fmt_tag}"
    coeff_dir = f"{ABLATION_ROOT}/coefficients/coeff_{proj}_frBeta{beta_tag}_fmt{fmt_tag}"
    ckpt_dir  = f"{ABLATION_ROOT}/merged_models/ckpt_{proj}_frBeta{beta_tag}_fmt{fmt_tag}"
    eval_dir  = f"{ABLATION_ROOT}/eval_results"

    run_name     = f"ablation_{proj}_frBeta{beta_tag}_fmt{fmt_tag}_l{lam_tag}"
    eval_out_dir = os.path.join(eval_dir, run_name)
    metrics_path = os.path.join(eval_out_dir, "metrics.json")

    if os.path.exists(metrics_path):
        print(f"[eval/{run_name}] already done, skipping.", flush=True)
        rmse = read_rmse(eval_out_dir)
        append_result({"round": "round2b", "proj": proj, "fisher_beta": beta,
                       "delta_fmt": delta_fmt, "lambda": lam,
                       "run_name": run_name, "rmse": rmse})
        return rmse

    ensure_dirs(resp_dir, fmt_dir, coeff_dir, ckpt_dir, eval_out_dir)

    # step4 beta
    run_step4_beta(proj, beta)

    # step5 format suppression (reuses fmt_dir distinct from frTrue)
    marker5 = os.path.join(fmt_dir, "fisher_supp_tag_q.pt")
    if not os.path.exists(marker5):
        extra5 = {
            "stage1_delta_path": STAGE1_PKLS[proj],
            "attention_dir":     d["attention_dir"],
            "floors_dir":        fmt_dir,
            "delta_fmt":         delta_fmt,
        }
        run([PYTHON, script("step5_format_suppression.py"), "--override_json", cfg_json(extra5)],
            label=f"step5_beta/{proj}/b{beta_tag}/fmt{fmt_tag}")
    else:
        print(f"[step5_beta/{proj}/b{beta_tag}/fmt{fmt_tag}] already done, skipping.", flush=True)

    # step6 coefficients
    marker6 = os.path.join(coeff_dir, "alpha_q.pt")
    if not os.path.exists(marker6):
        extra6 = {
            "stage1_delta_path":  STAGE1_PKLS[proj],
            "attention_dir":      d["attention_dir"],
            "responsibility_dir": resp_dir,
            "floors_dir":         fmt_dir,
            "coefficients_dir":   coeff_dir,
        }
        run([PYTHON, script("step6_coefficients_simple.py"), "--override_json", cfg_json(extra6)],
            label=f"step6_beta/{proj}/b{beta_tag}/fmt{fmt_tag}")
    else:
        print(f"[step6_beta/{proj}/b{beta_tag}/fmt{fmt_tag}] already done, skipping.", flush=True)

    # step7 merge
    lam_str    = f"{lam:.2f}".replace(".", "_")
    fmt_str    = f"{delta_fmt:.2f}".replace(".", "_")
    merged_path = os.path.join(ckpt_dir, f"merged_lambda_{lam_str}_delta_fmt_{fmt_str}")
    if not os.path.exists(os.path.join(merged_path, "config.json")):
        extra7 = {
            "stage1_delta_path": STAGE1_PKLS[proj],
            "coefficients_dir":  coeff_dir,
            "checkpoint_dir":    ckpt_dir,
            "delta_fmt":         delta_fmt,
            "lambda_global":     lam,
        }
        run([PYTHON, script("step7_merge.py"),
             "--override_json", cfg_json(extra7),
             "--lambda_global", str(lam)],
            label=f"step7_beta/{proj}/b{beta_tag}/fmt{fmt_tag}/l{lam_tag}")
    else:
        print(f"[step7_beta/{proj}/b{beta_tag}] merged model exists, skipping.", flush=True)

    pick_gpu()
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
        label=f"eval_beta/{proj}/b{beta_tag}")

    rmse = read_rmse(eval_out_dir)
    append_result({"round": "round2b", "proj": proj, "fisher_beta": beta,
                   "delta_fmt": delta_fmt, "lambda": lam,
                   "run_name": run_name, "rmse": rmse})
    return rmse


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
        cmd += ["--weight_formula", weight_formula]
    run(cmd, label=f"step6/{proj}/{fr}/fmt{fmt_tag}{w_suffix}")
    return coeff_dir


def run_step7_and_eval(proj, fisher_resp, delta_fmt, lam, weight_formula=None, coeff_dir=None):
    fr = "frTrue" if fisher_resp else "frFalse"
    fmt_tag = f"{delta_fmt:.2f}".replace(".", "")
    lam_tag = f"{lam:.2f}".replace(".", "")
    w_suffix = f"_w{weight_formula}" if weight_formula else ""
    d = dirs(proj, fisher_resp, fmt_tag)

    if coeff_dir is None:
        coeff_dir = f"{ABLATION_ROOT}/coefficients/coeff_{proj}_{fr}_fmt{fmt_tag}{w_suffix}"
    ckpt_dir = f"{ABLATION_ROOT}/merged_models/ckpt_{proj}_{fr}_fmt{fmt_tag}{w_suffix}"

    lam_str = f"{lam:.2f}".replace(".", "_")
    fmt_str = f"{delta_fmt:.2f}".replace(".", "_")
    merged_path = os.path.join(ckpt_dir, f"merged_lambda_{lam_str}_delta_fmt_{fmt_str}")

    run_name     = f"ablation_{proj}_{fr}_fmt{fmt_tag}{w_suffix}_l{lam_tag}"
    eval_out_dir = os.path.join(d["eval_dir"], run_name)
    metrics_path = os.path.join(eval_out_dir, "metrics.json")

    if os.path.exists(metrics_path):
        print(f"[eval/{run_name}] already done, skipping.", flush=True)
        return read_rmse(eval_out_dir)

    ensure_dirs(ckpt_dir, eval_out_dir)

    if not os.path.exists(os.path.join(merged_path, "config.json")):
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

    pick_gpu()
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
    """Steps 4-6 then eval for a single variant. Called from thread pool."""
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
    parser.add_argument("--start_round", type=str, default="1",
                        choices=["1", "2", "2b", "3", "3b", "4"],
                        help="Resume from this round (1, 2, 2b, 3, 3b, 4)")
    parser.add_argument("--best_proj",   type=str, default=None,
                        help="Override BEST_PROJ (skip Round 1 selection)")
    parser.add_argument("--best_fisher", type=str, default=None,
                        help="'True'/'False' — override BEST_FISHER")
    parser.add_argument("--best_fmt",    type=float, default=None,
                        help="Override BEST_FMT delta_fmt value")
    parser.add_argument("--best_beta",   type=float, default=None,
                        help="Override BEST_BETA fisher_beta value (skip Round 2b)")
    parser.add_argument("--best_weight", type=str, default=None,
                        choices=["paper", "inverse"],
                        help="Override BEST_WEIGHT formula (skip Round 3b)")
    args = parser.parse_args()

    os.makedirs(ABLATION_ROOT, exist_ok=True)

    # ── Round 1: projection variant — fully parallel ───────────────────────────
    round_order = ["1", "2", "2b", "3", "3b", "4"]

    def _from(r):
        return round_order.index(args.start_round) <= round_order.index(r)

    if _from("1") and args.best_proj is None:
        print("\n" + "="*60)
        print("ROUND 1 — Stage 1 projection variant  (3-way parallel)")
        print("="*60)

        projs = ("nullspace", "bias", "nullspace_bias")

        # 3 attention jobs in parallel
        print("\n--- Step 3: attention (parallel) ---")
        run_parallel([lambda p=p: run_step3(p) for p in projs])

        # 3 × (steps 4-6 + step7 + eval) 2 at a time
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

        fr_true_run  = f"ablation_{best_proj}_frTrue_fmt09_l05"
        fr_true_eval = os.path.join(ABLATION_ROOT, "eval_results", fr_true_run)
        rmse_true  = read_rmse(fr_true_eval)
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

    # ── Round 2b: Fisher beta dampening — 3-way parallel (only when best_fisher=True) ──
    if _from("2b") and best_fisher and args.best_beta is None:
        print("\n" + "="*60)
        print(f"ROUND 2b — Fisher beta dampening  (proj={best_proj}, nullspace pkl lambda=0.5, delta_fmt=0.9)")
        print("="*60)

        BETA_GRID = (0.25, 0.5, 0.75)
        r2b_results = {}

        def _r2b_job(beta):
            rmse = pipeline_beta_456_eval(best_proj, beta, delta_fmt=0.9, lam=0.5)
            r2b_results[beta] = rmse

        print(f"\n--- FISHER_BETA in {BETA_GRID} (3 in parallel) ---")
        run_parallel([lambda b=b: _r2b_job(b) for b in BETA_GRID], max_workers=3)

        for beta in BETA_GRID:
            print(f"  beta={beta}: RMSE={r2b_results[beta]:.4f}")

        best_beta = min(r2b_results, key=r2b_results.get)
        print(f"\nRound 2b winner: fisher_beta={best_beta}  RMSE={r2b_results[best_beta]:.4f}")
        append_result({"round": "round2b_summary", "best_proj": best_proj,
                       "best_fisher_resp": best_fisher, "best_beta": best_beta,
                       "results": {str(k): v for k, v in r2b_results.items()}})
    elif _from("2b") and not best_fisher:
        best_beta = None
        print(f"\nSkipping Round 2b — best_fisher=False, beta dampening not applicable.")
    else:
        best_beta = args.best_beta
        print(f"\nSkipping Round 2b — using BEST_BETA={best_beta}")

    # ── Round 3: delta_fmt grid — 3 new jobs in parallel ─────────────────────
    if _from("3") and args.best_fmt is None:
        print("\n" + "="*60)
        print(f"ROUND 3 — Format suppression  (proj={best_proj}, fr={best_fisher})")
        print("="*60)

        fr_tag   = "frTrue" if best_fisher else "frFalse"
        done_run = f"ablation_{best_proj}_{fr_tag}_fmt09_l05"
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

    # ── Round 3b: weight formula — 2 jobs in parallel ─────────────────────────
    if _from("3b") and args.best_weight is None:
        print("\n" + "="*60)
        print(f"ROUND 3b — Weight formula  (proj={best_proj}, fr={best_fisher}, fmt={best_fmt})")
        print("="*60)

        # paper formula with best_fmt/lam=0.5 is already computed in Round 3 — reuse
        fr_tag   = "frTrue" if best_fisher else "frFalse"
        fmt_tag  = f"{best_fmt:.2f}".replace(".", "")
        paper_run = f"ablation_{best_proj}_{fr_tag}_fmt{fmt_tag}_wpaper_l05"
        paper_eval_dir = os.path.join(ABLATION_ROOT, "eval_results", paper_run)

        # Check if paper was already run under the _wpaper suffix; if not, check
        # the Round 3 result (no suffix, which also used paper weights via config)
        if not os.path.exists(os.path.join(paper_eval_dir, "metrics.json")):
            no_suffix_run = f"ablation_{best_proj}_{fr_tag}_fmt{fmt_tag}_l05"
            no_suffix_dir = os.path.join(ABLATION_ROOT, "eval_results", no_suffix_run)
            rmse_paper = read_rmse(no_suffix_dir)
        else:
            rmse_paper = read_rmse(paper_eval_dir)

        r3b_results = {"paper": rmse_paper}

        print(f"  paper (reused): RMSE={rmse_paper:.4f}")
        print("\n--- inverse formula (step6+7+eval) ---")

        coeff_dir_inv = run_step6(best_proj, best_fisher, best_fmt, weight_formula="inverse")
        rmse_inverse  = run_step7_and_eval(best_proj, best_fisher, best_fmt, lam=0.5,
                                           weight_formula="inverse", coeff_dir=coeff_dir_inv)
        r3b_results["inverse"] = rmse_inverse
        print(f"  inverse: RMSE={rmse_inverse:.4f}")

        best_weight = min(r3b_results, key=r3b_results.get)
        print(f"\nRound 3b winner: weight_formula={best_weight}  RMSE={r3b_results[best_weight]:.4f}")
        append_result({"round": "round3b_summary", "best_proj": best_proj,
                       "best_fisher_resp": best_fisher, "best_fmt": best_fmt,
                       "best_weight": best_weight, "results": r3b_results})
    else:
        best_weight = args.best_weight if args.best_weight else "paper"
        print(f"\nSkipping Round 3b — using BEST_WEIGHT={best_weight}")

    # ── Round 4: lambda grid — all 6 step7+eval, 3 at a time ─────────────────
    print("\n" + "="*60)
    print(f"ROUND 4 — Lambda  (proj={best_proj}, fr={best_fisher}, fmt={best_fmt}, w={best_weight})")
    print("="*60)

    # step6 coefficients are shared across all lambdas — build once first
    coeff_dir_r4 = run_step6(best_proj, best_fisher, best_fmt,
                             weight_formula=best_weight if best_weight != "paper" else None)

    r4_results = {}

    print("\n--- lambda in (0.35, 0.4, 0.45, 0.5, 0.55, 0.6) step7+eval (3 at a time) ---")

    wf_r4 = best_weight if best_weight != "paper" else None

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
