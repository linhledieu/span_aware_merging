"""
Step 2b: Re-normalize already-computed raw Fisher tensors using per-span normalization.

Run this instead of re-running step2_fisher.py when raw tensors already exist.
Fixes the bug where global normalization across all spans compressed match Fisher
to near-zero because tag Fisher is 10x larger in magnitude.
"""
import argparse
import os
import pickle
import re

import torch

from config import load_config
from utils.normalization import global_minmax_normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--override_json", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.override_json)

    all_spans = ["tag", "match", "user", "item"]
    proj_types = ["q", "k", "v", "o"]

    fisher_raw = {}
    for span_name in all_spans:
        proj_dict = {}
        for proj_type in proj_types:
            path = os.path.join(cfg["fisher_dir"], f"fisher_raw_{span_name}_{proj_type}.pt")
            if os.path.exists(path):
                proj_dict[proj_type] = torch.load(path, map_location="cpu").float()
        if proj_dict:
            fisher_raw[span_name] = proj_dict

    print(f"Loaded raw Fisher for spans: {list(fisher_raw.keys())}")
    for span_name, proj_dict in fisher_raw.items():
        for proj_type, t in proj_dict.items():
            print(f"  {span_name}/{proj_type}: shape={list(t.shape)}, mean={t.mean().item():.6f}, max={t.max().item():.6f}")

    # Per-projection normalization: normalize each (span, proj) pair independently to [0,1]
    # This ensures Q, K, V, O each get their own scale rather than V dominating
    # within a span due to magnitude differences across projection types
    norm_dict = {}
    for span_name, proj_dict in fisher_raw.items():
        for proj_type, tensor in proj_dict.items():
            t_min = tensor.min()
            t_max = tensor.max()
            if t_max > t_min:
                normed = (tensor - t_min) / (t_max - t_min)
            else:
                normed = torch.zeros_like(tensor)
            norm_dict[f"{span_name}_{proj_type}"] = normed

    print("\nSaving normalized tensors...")
    for key, tensor in norm_dict.items():
        out_path = os.path.join(cfg["fisher_dir"], f"fisher_norm_{key}.pt")
        torch.save(tensor, out_path)
        print(f"  Saved {key}: max={tensor.max().item():.4f}, mean={tensor.mean().item():.6f}")

    # Load delta to find covered layers
    delta_path = cfg["stage1_delta_path"]
    if delta_path.endswith(".pkl"):
        with open(delta_path, "rb") as f:
            delta = pickle.load(f)
    else:
        delta = torch.load(delta_path, map_location="cpu")

    delta_layers = set()
    # Try flat param-name keys first (e.g. "model.layers.9.self_attn.q_proj.weight")
    for key in delta.keys():
        m = re.search(r'\.layers\.(\d+)\.', str(key))
        if m:
            delta_layers.add(int(m.group(1)))
    # Fall back: nested structure with integer layer keys under qk/vo/ffn
    if not delta_layers and isinstance(delta, dict):
        ptv = delta.get("projected_task_vectors", {})
        for section in ("qk", "vo", "ffn"):
            if section in ptv and isinstance(ptv[section], dict):
                for k in ptv[section].keys():
                    try:
                        delta_layers.add(int(k))
                    except (ValueError, TypeError):
                        pass

    print(f"\nDelta covers {len(delta_layers)} layers: {min(delta_layers)}–{max(delta_layers)}")

    print("\n=== Normalized top-5 restricted to delta-covered layers ===")
    for span_name in ("tag", "match"):
        if span_name not in fisher_raw:
            continue
        for proj_type in proj_types:
            key = f"{span_name}_{proj_type}"
            if key not in norm_dict:
                continue
            t = norm_dict[key]  # [num_layers, H]
            mask = torch.zeros(t.shape[0], dtype=torch.bool)
            for l in delta_layers:
                if l < t.shape[0]:
                    mask[l] = True
            t_masked = t.clone()
            t_masked[~mask] = 0.0
            flat = t_masked.reshape(-1)
            k_top = min(5, (flat > 1e-6).sum().item())
            if k_top == 0:
                print(f"  {span_name}/{proj_type}: no values above threshold in delta layers")
                continue
            top5 = torch.topk(flat, k_top)
            entries = []
            for val, idx in zip(top5.values, top5.indices):
                l = idx.item() // t.shape[1]
                h = idx.item() % t.shape[1]
                entries.append(f"(layer={l},head={h},val={val.item():.4f})")
            print(f"  {span_name}/{proj_type}: {', '.join(entries)}")

    delta_fmt = cfg["delta_fmt"]
    delta_coh = cfg["delta_coh"]
    print(f"\n=== Floor preview (delta_fmt={delta_fmt}, delta_coh={delta_coh}) ===")
    print(f"{'proj':<6}  {'mean':>8}  {'median':>8}  {'p90':>8}  {'max':>8}  {'>0.3':>6}  {'>0.5':>6}")
    for proj_type in proj_types:
        tag_key = f"tag_{proj_type}"
        match_key = f"match_{proj_type}"
        if tag_key not in norm_dict or match_key not in norm_dict:
            continue
        F_tag = norm_dict[tag_key]
        F_match = norm_dict[match_key]
        floor = delta_fmt * F_tag + delta_coh * F_match
        flat = floor.reshape(-1)
        p90 = torch.quantile(flat, 0.9).item()
        print(
            f"{proj_type:<6}  {flat.mean().item():>8.4f}  {flat.median().item():>8.4f}"
            f"  {p90:>8.4f}  {flat.max().item():>8.4f}"
            f"  {(flat > 0.3).sum().item():>6}  {(flat > 0.5).sum().item():>6}"
        )


if __name__ == "__main__":
    main()
