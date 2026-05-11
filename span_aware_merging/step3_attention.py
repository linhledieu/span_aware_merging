"""Step 3: Compute attention statistics (U, alignment) on D_I.

This step always runs fully regardless of ablation flags.
The only effect of USE_ATTENTION_MODULATION=False is that nu is resolved to 0
in config, which step6 reads. This step always computes real tensors.
u_leak is always zero (leakage dropped) and saved for interface compatibility.
"""
import argparse
import json
import os
import pickle
import time

import torch

from config import load_config
from utils.attention_utils import compute_attention_statistics
from utils.model_utils import load_merged_model_at_alpha1, get_head_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--override_json", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.override_json)

    os.makedirs(cfg["attention_dir"], exist_ok=True)

    di_masks_path = os.path.join(cfg["masks_dir"], "di_masks.pkl")
    with open(di_masks_path, "rb") as f:
        di_records = pickle.load(f)

    valid_records = [r for r in di_records if r["valid"]]
    print(f"Loaded {len(di_records)} D_I records, {len(valid_records)} valid", flush=True)
    print(
        "Attention config: "
        f"device={cfg['device']} dtype={cfg['dtype_str']} "
        f"chunk_size={cfg.get('attention_chunk_size', 8)} "
        f"log_every={cfg.get('attention_log_every', 10)}"
        ,
        flush=True,
    )

    print("Loading merged model at alpha=1...", flush=True)
    model, _ = load_merged_model_at_alpha1(cfg)
    head_config = get_head_config(model)

    t0 = time.time()
    stats = compute_attention_statistics(model, valid_records, head_config, cfg)
    print(f"Attention statistics finished in {time.time() - t0:.1f}s", flush=True)

    U = stats["U"]
    a_align = stats["a"]
    u_leak = stats["u"]  # always zeros — leakage dropped

    # Shape assertions
    num_layers = head_config["num_layers"]
    H_q = head_config["num_attention_heads"]
    assert U.shape == (num_layers, H_q, 3), f"U shape mismatch: {U.shape}"
    assert a_align.shape == U.shape, f"a_align shape mismatch: {a_align.shape}"

    torch.save(U, os.path.join(cfg["attention_dir"], "U.pt"))
    torch.save(a_align, os.path.join(cfg["attention_dir"], "a_align.pt"))
    torch.save(u_leak, os.path.join(cfg["attention_dir"], "u_leak.pt"))

    meta = {
        "num_layers": num_layers,
        "num_attention_heads": H_q,
        "num_key_value_heads": head_config["num_key_value_heads"],
        "num_valid_examples": len(valid_records),
        "span_order": ["user", "item", "match"],
    }
    with open(os.path.join(cfg["attention_dir"], "attention_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved U, a_align, u_leak: shape {list(U.shape)}", flush=True)

    # Per-layer diagnostics
    print(f"\n{'Layer':>6}  {'U_user':>8}  {'U_item':>8}  {'U_match':>8}  flag", flush=True)
    for l in range(num_layers):
        means = U[l].mean(dim=0)  # [3]
        flag = "" if means[2] == means.max() else "  *** match not dominant"
        print(
            f"{l:>6}  {means[0].item():>8.4f}  {means[1].item():>8.4f}"
            f"  {means[2].item():>8.4f}{flag}",
            flush=True,
        )
    overall = U.mean(dim=(0, 1))
    print(
        f"\nOverall: U_user={overall[0]:.4f}  U_item={overall[1]:.4f}  U_match={overall[2]:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
