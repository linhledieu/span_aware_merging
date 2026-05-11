"""Run all four ablation conditions and the lambda grid search."""
import argparse
import csv
import json
import os
import subprocess
import sys

from config import load_config

ABLATION_CONDITIONS = [
    {
        "name": "uniform_baseline",
        "USE_JSD_WEIGHTS": False,
        "USE_ATTENTION_MODULATION": False,
        "USE_FISHER_FLOORS": False,
        "USE_FISHER_RESPONSIBILITY": False,
    },
    {
        "name": "jsd_weights_only",
        "USE_JSD_WEIGHTS": True,
        "USE_ATTENTION_MODULATION": False,
        "USE_FISHER_FLOORS": False,
        "USE_FISHER_RESPONSIBILITY": False,
    },
    {
        "name": "jsd_plus_attention",
        "USE_JSD_WEIGHTS": True,
        "USE_ATTENTION_MODULATION": True,
        "USE_FISHER_FLOORS": False,
        "USE_FISHER_RESPONSIBILITY": False,
    },
    {
        "name": "full_method",
        "USE_JSD_WEIGHTS": True,
        "USE_ATTENTION_MODULATION": True,
        "USE_FISHER_FLOORS": True,
        "USE_FISHER_RESPONSIBILITY": True,
    },
]


def run(cmd, check=True):
    print(f"\n>>> {' '.join(str(x) for x in cmd)}", flush=True)
    result = subprocess.run(cmd, check=False)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.returncode


def make_condition_cfg(base_cfg, condition):
    """Merge condition flags and redirect output dirs to condition-specific subdirs."""
    cfg = dict(base_cfg)
    name = condition["name"]
    for k, v in condition.items():
        if k != "name":
            cfg[k] = v
    # Per-condition output directories
    for dir_key in ("fisher_dir", "attention_dir", "responsibility_dir",
                    "floors_dir", "coefficients_dir"):
        cfg[dir_key] = os.path.join(base_cfg[dir_key], name)
    cfg["checkpoint_dir"] = os.path.join(base_cfg["checkpoint_dir"], name)
    # Rerun load_config flag resolution after applying condition flags
    from config import load_config as _lc
    cfg = _lc(json.dumps(cfg))
    # Restore dirs (load_config uses get_defaults which would reset them)
    # Actually we need to re-apply dirs after load_config resets them
    # Workaround: pass as part of override so load_config sees them
    return cfg


def build_override_for_condition(base_cfg, condition):
    """Build override dict that load_config will apply on top of defaults."""
    override = {}
    name = condition["name"]
    # Condition flags
    for k, v in condition.items():
        if k != "name":
            override[k] = v
    # Copy all base_cfg values that are non-default (paths, etc.)
    for k in ("model_r_path", "model_b_path", "stage1_delta_path",
              "dr_path", "di_path", "test_path", "device", "dtype_str",
              "lambda_grid", "rho", "nu", "delta_fmt", "delta_coh",
              "alpha_upper", "lambda_global",
              "w_user", "w_item", "w_match"):
        override[k] = base_cfg[k]
    # Condition-specific output dirs
    for dir_key in ("fisher_dir", "attention_dir", "responsibility_dir",
                    "floors_dir", "coefficients_dir"):
        override[dir_key] = os.path.join(base_cfg[dir_key], name)
    override["checkpoint_dir"] = os.path.join(base_cfg["checkpoint_dir"], name)
    override["masks_dir"] = base_cfg["masks_dir"]
    override["eval_dir"] = base_cfg["eval_dir"]
    return override


def load_results(eval_dir, output_name):
    path = os.path.join(eval_dir, f"{output_name}_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--override_json", type=str, default=None)
    args = parser.parse_args()

    base_cfg = load_config(args.override_json)
    python = sys.executable
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def script(name):
        return os.path.join(script_dir, name)

    # Step 1 runs once (does not depend on any ablation flags)
    print("\n" + "=" * 60)
    print("Running step1_span_masks.py (once)")
    print("=" * 60)
    run([python, script("step1_span_masks.py"),
         "--override_json", json.dumps({"model_r_path": base_cfg["model_r_path"],
                                        "dr_path": base_cfg["dr_path"],
                                        "di_path": base_cfg["di_path"],
                                        "masks_dir": base_cfg["masks_dir"]})])

    all_results = []

    for condition in ABLATION_CONDITIONS:
        name = condition["name"]
        print(f"\n{'='*60}")
        print(f"Condition: {name}")
        print(f"{'='*60}")

        override = build_override_for_condition(base_cfg, condition)
        override_str = json.dumps(override)

        # Create output dirs
        for dir_key in ("fisher_dir", "attention_dir", "responsibility_dir",
                        "floors_dir", "coefficients_dir", "checkpoint_dir"):
            os.makedirs(override[dir_key], exist_ok=True)

        run([python, script("step2b_renormalize_fisher.py"), "--override_json", override_str])
        run([python, script("step3_attention.py"), "--override_json", override_str])
        run([python, script("step4_responsibility.py"), "--override_json", override_str])
        run([python, script("step5_format_suppression.py"), "--override_json", override_str])
        run([python, script("step6_coefficients_simple.py"), "--override_json", override_str])
        run([python, script("step7_merge.py"), "--override_json", override_str,
             "--lambda_global", "1.0"])

        lam_str = "1_00"
        ckpt_dir = os.path.join(override["checkpoint_dir"], f"merged_lambda_{lam_str}")
        smoke_rc = run(
            [python, script("step8_smoketest.py"), "--checkpoint_dir", ckpt_dir,
             "--override_json", override_str],
            check=False,
        )
        if smoke_rc != 0:
            print(f"WARNING: step8 smoke test failed for condition '{name}', continuing.")

        eval_name = f"{name}_lambda1"
        eval_override = dict(override)
        run([python, script("evaluate.py"),
             "--checkpoint_dir", ckpt_dir,
             "--test_data", base_cfg["test_path"],
             "--output_name", eval_name,
             "--override_json", json.dumps(eval_override)])

        res = load_results(base_cfg["eval_dir"], eval_name)
        if res:
            res["condition_name"] = name
            res["lambda"] = 1.0
            all_results.append(res)

    # Lambda grid search for full_method only
    full_condition = ABLATION_CONDITIONS[3]
    full_override = build_override_for_condition(base_cfg, full_condition)
    print(f"\n{'='*60}\nLambda grid search (full_method)\n{'='*60}")

    for lam in base_cfg["lambda_grid"]:
        lam_tag = f"{lam:.2f}".replace(".", "_")
        ckpt_dir = os.path.join(full_override["checkpoint_dir"], f"merged_lambda_{lam_tag}")
        run([python, script("step7_merge.py"),
             "--override_json", json.dumps(full_override),
             "--lambda_global", str(lam)])

        eval_name = f"full_method_lambda_{lam_tag}"
        run([python, script("evaluate.py"),
             "--checkpoint_dir", ckpt_dir,
             "--test_data", base_cfg["test_path"],
             "--output_name", eval_name,
             "--override_json", json.dumps(full_override)])

        res = load_results(base_cfg["eval_dir"], eval_name)
        if res:
            res["condition_name"] = "full_method"
            res["lambda"] = lam
            all_results.append(res)

    # Reference: uniform_baseline lambda=1.0
    ref = next((r for r in all_results if r.get("condition_name") == "uniform_baseline"), None)

    def delta(res, key):
        if ref is None:
            return float("nan")
        return res.get(key, float("nan")) - ref.get(key, float("nan"))

    summary_rows = []
    for res in all_results:
        summary_rows.append({
            "condition_name": res.get("condition_name", ""),
            "lambda": res.get("lambda", 1.0),
            "MAE": res.get("mae", float("nan")),
            "RMSE": res.get("rmse", float("nan")),
            "format_intact_rate": res.get("format_intact_rate", float("nan")),
            "mean_output_tokens": res.get("mean_output_tokens", float("nan")),
            "mean_match_tokens": res.get("mean_match_tokens", float("nan")),
            "mean_user_tokens": res.get("mean_user_tokens", float("nan")),
            "mean_item_tokens": res.get("mean_item_tokens", float("nan")),
            "delta_MAE_vs_uniform": delta(res, "mae"),
            "delta_RMSE_vs_uniform": delta(res, "rmse"),
            "delta_tokens_vs_uniform": delta(res, "mean_output_tokens"),
        })

    os.makedirs(base_cfg["eval_dir"], exist_ok=True)

    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    csv_path = os.path.join(base_cfg["eval_dir"], "ablation_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = os.path.join(base_cfg["eval_dir"], "ablation_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_rows, f, indent=2, default=str)

    print(f"\nAblation summary → {csv_path}")
    print(f"\n{'condition':<25} {'lambda':>7} {'MAE':>8} {'RMSE':>8} {'FIR':>6} {'dMAE':>8}")
    for row in summary_rows:
        print(
            f"{row['condition_name']:<25} {row['lambda']:>7.2f}"
            f" {row['MAE']:>8.4f} {row['RMSE']:>8.4f}"
            f" {row['format_intact_rate']:>6.3f} {row['delta_MAE_vs_uniform']:>8.4f}"
        )


if __name__ == "__main__":
    main()
