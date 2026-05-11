"""Step 2: Compute span-conditioned diagonal Fisher for Q/K/V/O projections."""
import argparse
import json
import os
import pickle
import sys
import time

import torch

from config import load_config
from utils.fisher_utils import compute_all_fisher
from utils.model_utils import load_reasoning_model, get_qkvo_param_names, get_head_config
from utils.normalization import global_minmax_normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--override_json", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.override_json)

    # Both flags resolved to disabled means Fisher is not needed
    if cfg["delta_fmt"] == 0.0 and cfg["delta_coh"] == 0.0 and not cfg["USE_FISHER_RESPONSIBILITY"]:
        print("Fisher disabled by ablation flags, skipping.")
        sys.exit(0)

    os.makedirs(cfg["fisher_dir"], exist_ok=True)

    dr_masks_path = os.path.join(cfg["masks_dir"], "dr_masks.pkl")
    with open(dr_masks_path, "rb") as f:
        dr_records = pickle.load(f)

    # Verify alignment with source file
    with open(cfg["dr_path"]) as f:
        n_lines = sum(1 for line in f if line.strip())
    assert len(dr_records) == n_lines, (
        f"dr_records length {len(dr_records)} != dr_path line count {n_lines}"
    )
    valid_dr_records = [r for r in dr_records if r["valid"]]
    print(f"Loaded {len(dr_records)} D_R records ({len(valid_dr_records)} valid)")
    if valid_dr_records:
        seq_lens = [len(r["input_ids"]) for r in valid_dr_records]
        print(
            "Fisher input length stats: "
            f"mean={sum(seq_lens) / len(seq_lens):.1f}, "
            f"min={min(seq_lens)}, max={max(seq_lens)}"
        )
    print(f"Fisher progress logging cadence: every {cfg.get('fisher_log_every', 10)} records")

    print("Loading reasoning model...")
    model, _ = load_reasoning_model(cfg)
    proj_param_names = get_qkvo_param_names(model)
    model.train()  # Ensure grad hooks are active

    # Fisher only reads gradients from Q/K/V/O projection weights, so freezing the
    # rest of the model avoids allocating unnecessary grad buffers.
    model.requires_grad_(False)
    trainable_names = set()
    for proj_list in proj_param_names.values():
        trainable_names.update(proj_list)
    for name, param in model.named_parameters():
        if name in trainable_names:
            param.requires_grad_(True)

    # Reduce activation memory during backward. This is especially important for
    # long calibration sequences on 44GB cards.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    head_config = get_head_config(model)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params for Fisher: {n_trainable:,}")
    print(f"Head config: {head_config}")

    t0 = time.time()
    fisher = compute_all_fisher(model, dr_records, proj_param_names, head_config, cfg)
    print(f"\nTotal Fisher time: {time.time() - t0:.1f}s")

    # Save raw tensors
    for span_name, proj_dict in fisher.items():
        for proj_type, tensor in proj_dict.items():
            torch.save(tensor, os.path.join(cfg["fisher_dir"], f"fisher_raw_{span_name}_{proj_type}.pt"))

    # Per-span normalization: normalize within each span independently
    # This prevents tag Fisher (10x larger magnitude) from dominating match Fisher
    # and compressing it to near-zero after global normalization
    norm_dict = {}
    for span_name, proj_dict in fisher.items():
        span_flat = {proj_type: tensor for proj_type, tensor in proj_dict.items()}
        span_norm = global_minmax_normalize(span_flat)  # min/max computed within this span only
        for proj_type, normed in span_norm.items():
            norm_dict[f"{span_name}_{proj_type}"] = normed

    for key, tensor in norm_dict.items():
        torch.save(tensor, os.path.join(cfg["fisher_dir"], f"fisher_norm_{key}.pt"))

    # Diagnostics
    print("\n=== Fisher diagnostics ===")
    print(f"{'span':<8} {'proj':<6} {'mean':>10}")
    for span_name in fisher:
        for proj_type in ("q", "k", "v", "o"):
            t = fisher[span_name][proj_type]
            print(f"{span_name:<8} {proj_type:<6} {t.mean().item():>10.6f}")

    print("\n=== Top-5 (layer,head) by Fisher value for tag and match ===")
    for span_name in ("tag", "match"):
        if span_name not in fisher:
            continue
        for proj_type in ("q", "k", "v", "o"):
            t = fisher[span_name][proj_type]
            flat = t.reshape(-1)
            top5 = torch.topk(flat, min(5, flat.numel()))
            entries = []
            for val, idx in zip(top5.values, top5.indices):
                l = idx.item() // t.shape[1]
                h = idx.item() % t.shape[1]
                entries.append(f"({l},{h},{val.item():.4f})")
            print(f"  {span_name}/{proj_type}: {' '.join(entries)}")

    # Load delta to find which layers are actually covered
    print("\n=== Top-5 (layer,head) restricted to delta-covered layers ===")
    try:
        import re
        delta_path = cfg["stage1_delta_path"]
        if delta_path.endswith(".pkl"):
            with open(delta_path, "rb") as f:
                delta = pickle.load(f)
        else:
            delta = torch.load(delta_path, map_location="cpu")

        # Find which layer indices appear in the delta keys
        delta_layers = set()
        for key in delta.keys():
            m = re.search(r'\.layers\.(\d+)\.', key)
            if m:
                delta_layers.add(int(m.group(1)))

        if not delta_layers:
            print("  Could not detect layer indices from delta keys.")
        else:
            min_layer = min(delta_layers)
            max_layer = max(delta_layers)
            print(f"  Delta covers layers {min_layer}–{max_layer} ({len(delta_layers)} layers total)")

            for span_name in ("tag", "match"):
                if span_name not in fisher:
                    continue
                for proj_type in ("q", "k", "v", "o"):
                    t = fisher[span_name][proj_type]  # [num_layers, H]
                    # Zero out layers not in delta before finding top-5
                    mask = torch.zeros(t.shape[0], dtype=torch.bool)
                    for l in delta_layers:
                        if l < t.shape[0]:
                            mask[l] = True
                    t_masked = t.clone()
                    t_masked[~mask] = 0.0
                    flat = t_masked.reshape(-1)
                    top5 = torch.topk(flat, min(5, (flat > 0).sum().item()))
                    entries = []
                    for val, idx in zip(top5.values, top5.indices):
                        if val.item() == 0.0:
                            continue
                        l = idx.item() // t.shape[1]
                        h = idx.item() % t.shape[1]
                        entries.append(f"({l},{h},{val.item():.4f})")
                    if entries:
                        print(f"  {span_name}/{proj_type}: {' '.join(entries)}")
    except Exception as e:
        print(f"  Could not load delta for layer filtering: {e}")

    # Print normalized top-5 restricted to delta-covered layers
    print("\n=== Normalized top-5 (layer,head) restricted to delta-covered layers ===")
    try:
        for span_name in ("tag", "match"):
            for proj_type in ("q", "k", "v", "o"):
                key = f"{span_name}_{proj_type}"
                if key not in norm_dict:
                    continue
                t = norm_dict[key]  # [num_layers, H], values in [0,1]
                mask = torch.zeros(t.shape[0], dtype=torch.bool)
                for l in delta_layers:
                    if l < t.shape[0]:
                        mask[l] = True
                t_masked = t.clone()
                t_masked[~mask] = 0.0
                flat = t_masked.reshape(-1)
                top5 = torch.topk(flat, min(5, (flat > 0).sum().item()))
                entries = []
                for val, idx in zip(top5.values, top5.indices):
                    if val.item() < 1e-6:
                        continue
                    l = idx.item() // t.shape[1]
                    h = idx.item() % t.shape[1]
                    entries.append(f"({l},{h},{val.item():.4f})")
                if entries:
                    print(f"  norm {span_name}/{proj_type}: {' '.join(entries)}")
    except Exception as e:
        print(f"  Could not print normalized delta-restricted diagnostics: {e}")


if __name__ == "__main__":
    main()
