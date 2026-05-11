"""Run steps 4-7 for music frTrue and frFalse variants."""
import json
import os
import subprocess
import sys

ABLATION_ROOT = "/data/uqlinh/rain_merging/span_aware_outputs_music/ablation"
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PYTHON        = sys.executable

BASE_CFG = {
    "model_r_path": "/data/uqlinh/reczero-music-ckpt",
    "model_b_path": "/data/uqlinh/tallrec/output-music-fixed/final",
    "fisher_dir":   f"{ABLATION_ROOT}/fisher",
    "masks_dir":    "/data/uqlinh/rain_merging/span_aware_outputs_music/masks",
    "device":       "cuda:0",
    "dtype_str":    "bfloat16",
    # weights = 1/jsd normalised: user=0.010354, item=0.008060, match=0.050882
    "w_user":  0.401912,
    "w_item":  0.516303,
    "w_match": 0.081785,
    "USE_JSD_WEIGHTS":          True,
    "USE_ATTENTION_MODULATION": True,
    "USE_FORMAT_SUPPRESSION":   True,
    "stage1_delta_path": f"{ABLATION_ROOT}/stage1_output/projected_task_vectors.pkl",
    "attention_dir":     f"{ABLATION_ROOT}/attention",
    "delta_fmt":      0.9,
    "lambda_global":  0.5,
}


def run(cmd, label=""):
    print(f"\n[{label}] >>> {' '.join(str(x) for x in cmd)}", flush=True)
    result = subprocess.run(cmd, check=False, env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"})
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def run_variant(fisher_resp=None, fisher_beta=None):
    if fisher_beta is not None:
        betatag = f"{fisher_beta:.2f}".replace(".", "")
        tag = f"frBeta{betatag}"
        step4_script = "step4_responsibility_beta.py"
        extra = {"FISHER_BETA": fisher_beta}
    else:
        tag = "frTrue" if fisher_resp else "frFalse"
        step4_script = "step4_responsibility.py"
        extra = {"USE_FISHER_RESPONSIBILITY": fisher_resp}

    resp_dir  = f"{ABLATION_ROOT}/responsibility/resp_nullspace_{tag}"
    fmt_dir   = f"{ABLATION_ROOT}/format_suppression/fmt_nullspace_{tag}_fmt090"
    coeff_dir = f"{ABLATION_ROOT}/coefficients/coeff_nullspace_{tag}_fmt090"
    ckpt_dir  = f"{ABLATION_ROOT}/merged_models/ckpt_nullspace_{tag}_fmt090"
    for d in [resp_dir, fmt_dir, coeff_dir, ckpt_dir]:
        os.makedirs(d, exist_ok=True)

    cfg = {
        **BASE_CFG,
        **extra,
        "responsibility_dir": resp_dir,
        "floors_dir":         fmt_dir,
        "coefficients_dir":   coeff_dir,
        "checkpoint_dir":     ckpt_dir,
    }
    cfg_str = json.dumps(cfg)

    print(f"\n{'='*50}\n{tag}\n{'='*50}", flush=True)

    if not os.path.exists(os.path.join(resp_dir, "resp_q.pt")):
        run([PYTHON, os.path.join(SCRIPT_DIR, step4_script),
             "--override_json", cfg_str], label=f"step4/{tag}")
    else:
        print(f"[step4/{tag}] already done, skipping.")

    if not os.path.exists(os.path.join(fmt_dir, "fisher_supp_tag_q.pt")):
        run([PYTHON, os.path.join(SCRIPT_DIR, "step5_format_suppression.py"),
             "--override_json", cfg_str], label=f"step5/{tag}")
    else:
        print(f"[step5/{tag}] already done, skipping.")

    if not os.path.exists(os.path.join(coeff_dir, "alpha_q.pt")):
        run([PYTHON, os.path.join(SCRIPT_DIR, "step6_coefficients_simple.py"),
             "--override_json", cfg_str], label=f"step6/{tag}")
    else:
        print(f"[step6/{tag}] already done, skipping.")

    merged_path = os.path.join(ckpt_dir, "merged_lambda_0_50_delta_fmt_0_90")
    if not os.path.exists(os.path.join(merged_path, "config.json")):
        run([PYTHON, os.path.join(SCRIPT_DIR, "step7_merge.py"),
             "--override_json", cfg_str, "--lambda_global", "0.5"],
            label=f"step7/{tag}")
    else:
        print(f"[step7/{tag}] already done, skipping.")

    print(f"\n[{tag}] Merged model: {merged_path}", flush=True)


if __name__ == "__main__":
    run_variant(fisher_resp=True)
    run_variant(fisher_resp=False)
    run_variant(fisher_beta=0.5)
    print("\nAll done.")
