"""Step 6 (simple): compute per-head merging coefficients on CPU only.

This is a simplified alternative to step6_coefficients.py.
It removes alignment and leakage from the alpha computation.
The coefficient is based on responsibility-weighted span preference plus Fisher floor:

C_{l,h}^p = sum_s w_s R_{l,h,s}^p
alpha_raw = alpha_upper * C_{l,h}^p / max_s w_s
alpha = min(alpha_upper, max(floor, alpha_raw))

The original step6_coefficients.py is left unchanged for ablation.

Example:

python step6_coefficients_simple.py \
  --override_json "$CFG"

Then Step 7 can be run as usual:

python step7_merge.py \
  --override_json "$CFG" \
  --lambda_global 0.5
"""
import argparse
import csv
import json
import os

import torch

from config import load_config


SPAN_NAMES = ["user", "item", "match"]


def _group_q_to_kv(tensor_q, h_kv):
    """Average [L, H_q, ...] -> [L, H_kv, ...] using contiguous query-head groups."""
    h_q = tensor_q.shape[1]
    if h_q % h_kv != 0:
        raise ValueError(f"Cannot group H_q={h_q} into H_kv={h_kv}")
    group_size = h_q // h_kv
    if tensor_q.dim() == 3:
        return tensor_q.view(tensor_q.shape[0], h_kv, group_size, tensor_q.shape[2]).mean(dim=2)
    return tensor_q.view(tensor_q.shape[0], h_kv, group_size).mean(dim=2)


def _clamp_alpha(value, floor_val, alpha_upper):
    return min(alpha_upper, max(floor_val, value))


def _mean_or_nan(values):
    return values.mean().item() if values.numel() > 0 else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--override_json", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.override_json)
    os.makedirs(cfg["coefficients_dir"], exist_ok=True)

    print("Running simplified Step 6 on CPU only.")

    meta_path = os.path.join(cfg["attention_dir"], "attention_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    span_order = meta.get("span_order", SPAN_NAMES)
    if span_order != SPAN_NAMES:
        raise ValueError(f"Expected span_order {SPAN_NAMES}, got {span_order}")

    num_layers = meta["num_layers"]
    h_q = meta["num_attention_heads"]
    h_kv = meta["num_key_value_heads"]

    u_q = torch.load(os.path.join(cfg["attention_dir"], "U.pt"), map_location="cpu").float()
    expected_u_shape = (num_layers, h_q, 3)
    if tuple(u_q.shape) != expected_u_shape:
        raise ValueError(f"U.pt shape {tuple(u_q.shape)} != expected {expected_u_shape}")

    floors = {
        proj: torch.load(os.path.join(cfg["floors_dir"], f"floor_{proj}.pt"), map_location="cpu").float()
        for proj in ("q", "k", "v", "o")
    }
    resp = {
        proj: torch.load(
            os.path.join(cfg["responsibility_dir"], f"resp_{proj}.pt"),
            map_location="cpu",
        ).float()
        for proj in ("q", "k", "v", "o")
    }

    u_kv = _group_q_to_kv(u_q, h_kv)
    proj_info = {
        "q": (h_q, u_q),
        "k": (h_kv, u_kv),
        "v": (h_kv, u_kv),
        "o": (h_q, u_q),
    }
    expected_shapes = {
        "q": (num_layers, h_q, 3),
        "k": (num_layers, h_kv, 3),
        "v": (num_layers, h_kv, 3),
        "o": (num_layers, h_q, 3),
    }
    expected_floor_shapes = {
        "q": (num_layers, h_q),
        "k": (num_layers, h_kv),
        "v": (num_layers, h_kv),
        "o": (num_layers, h_q),
    }

    w_user = float(cfg["w_user"])
    w_item = float(cfg["w_item"])
    w_match = float(cfg["w_match"])
    alpha_upper = float(cfg["alpha_upper"])
    user_head_scale = float(cfg["user_head_scale"])
    item_head_scale = float(cfg["item_head_scale"])
    w_max = max(w_user, w_item, w_match)
    safe_w_max = max(w_max, 1e-8)

    coefficients = {}
    csv_rows = []
    diagnostics = {
        "floor_gt_raw": 0,
        "alpha_at_upper": 0,
    }
    all_alpha = []
    all_compression = []
    all_dom = []

    for proj_type, (num_heads, u_heads) in proj_info.items():
        if tuple(u_heads.shape) != expected_shapes[proj_type]:
            raise ValueError(
                f"{proj_type} grouped U shape {tuple(u_heads.shape)} != expected {expected_shapes[proj_type]}"
            )
        if tuple(resp[proj_type].shape) != expected_shapes[proj_type]:
            raise ValueError(
                f"resp_{proj_type}.pt shape {tuple(resp[proj_type].shape)} != expected {expected_shapes[proj_type]}"
            )
        if tuple(floors[proj_type].shape) != expected_floor_shapes[proj_type]:
            raise ValueError(
                f"floor_{proj_type}.pt shape {tuple(floors[proj_type].shape)} != expected {expected_floor_shapes[proj_type]}"
            )

        alpha = torch.zeros((num_layers, num_heads), dtype=torch.float32)
        floor_t = floors[proj_type]
        resp_t = resp[proj_type]

        for layer in range(num_layers):
            for head_index in range(num_heads):
                r_user = resp_t[layer, head_index, 0].item()
                r_item = resp_t[layer, head_index, 1].item()
                r_match = resp_t[layer, head_index, 2].item()

                compression_score = (
                    w_user * r_user
                    + w_item * r_item
                    + w_match * r_match
                )
                alpha_raw = alpha_upper * compression_score / safe_w_max
                alpha_raw = min(alpha_upper, max(0.0, alpha_raw))

                floor_val = floor_t[layer, head_index].item()
                alpha_final = _clamp_alpha(alpha_raw, floor_val, alpha_upper)

                if floor_val > alpha_raw:
                    diagnostics["floor_gt_raw"] += 1

                dom_idx = int(resp_t[layer, head_index].argmax().item())
                dominant_span = SPAN_NAMES[dom_idx]
                if dom_idx == 0:
                    alpha_final *= user_head_scale
                elif dom_idx == 1:
                    alpha_final *= item_head_scale
                alpha_final = _clamp_alpha(alpha_final, floor_val, alpha_upper)

                if abs(alpha_final - alpha_upper) <= 1e-8:
                    diagnostics["alpha_at_upper"] += 1

                alpha[layer, head_index] = alpha_final

                u_user = u_heads[layer, head_index, 0].item()
                u_item = u_heads[layer, head_index, 1].item()
                u_match = u_heads[layer, head_index, 2].item()

                csv_rows.append(
                    {
                        "proj_type": proj_type,
                        "layer": layer,
                        "head_index": head_index,
                        "raw_coefficient": alpha_raw,
                        "floor_val": floor_val,
                        "final_coefficient": alpha_final,
                        "dominant_span": dominant_span,
                        "user_head_scale": user_head_scale,
                        "item_head_scale": item_head_scale,
                        "U_user": u_user,
                        "U_item": u_item,
                        "U_match": u_match,
                        "R_user": r_user,
                        "R_item": r_item,
                        "R_match": r_match,
                        "compression_score": compression_score,
                        "w_user": w_user,
                        "w_item": w_item,
                        "w_match": w_match,
                        "alpha_upper": alpha_upper,
                    }
                )

        coefficients[proj_type] = alpha
        torch.save(alpha, os.path.join(cfg["coefficients_dir"], f"alpha_{proj_type}.pt"))
        all_alpha.append(alpha.reshape(-1))
        all_compression.append(
            torch.tensor([row["compression_score"] for row in csv_rows if row["proj_type"] == proj_type])
        )
        all_dom.append(resp_t.argmax(dim=-1).reshape(-1))

    summary_path = os.path.join(cfg["coefficients_dir"], "alpha_summary_simple.csv")
    fieldnames = [
        "proj_type",
        "layer",
        "head_index",
        "raw_coefficient",
        "floor_val",
        "final_coefficient",
        "dominant_span",
        "user_head_scale",
        "item_head_scale",
        "U_user",
        "U_item",
        "U_match",
        "R_user",
        "R_item",
        "R_match",
        "compression_score",
        "w_user",
        "w_item",
        "w_match",
        "alpha_upper",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    meta_out = {
        "formula_description": (
            "Simplified Step 6: alpha is computed from responsibility-weighted span preference, "
            "then floored by Fisher floor and clipped to alpha_upper. Alignment and leakage are not used."
        ),
        "a_align_used": False,
        "u_leak_used": False,
        "span_order": SPAN_NAMES,
        "w_user": w_user,
        "w_item": w_item,
        "w_match": w_match,
        "alpha_upper": alpha_upper,
        "user_head_scale": user_head_scale,
        "item_head_scale": item_head_scale,
    }
    meta_out_path = os.path.join(cfg["coefficients_dir"], "alpha_simple_meta.json")
    with open(meta_out_path, "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"Saved alpha tensors to {cfg['coefficients_dir']}")
    print(f"Saved alpha_summary_simple.csv ({len(csv_rows)} rows) -> {summary_path}")
    print(f"Saved alpha_simple_meta.json -> {meta_out_path}")

    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0000001)]
    labels = ["[0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]
    print("\n=== Coefficient distribution histogram ===")
    for proj_type in ("q", "k", "v", "o"):
        flat = coefficients[proj_type].reshape(-1)
        counts = [((flat >= lo) & (flat < hi)).sum().item() for lo, hi in buckets]
        print(f"  {proj_type}: " + "  ".join(f"{label}:{count}" for label, count in zip(labels, counts)))

    all_alpha_flat = torch.cat(all_alpha, dim=0)
    all_dom_flat = torch.cat(all_dom, dim=0)
    all_compression_flat = torch.cat(all_compression, dim=0)

    print("\n=== Mean coefficient by dominant span ===")
    for idx, span_name in enumerate(SPAN_NAMES):
        mask = all_dom_flat == idx
        print(f"  {span_name}-dominant: {_mean_or_nan(all_alpha_flat[mask]):.4f}")

    print("\n=== Mean compression_score by dominant span ===")
    for idx, span_name in enumerate(SPAN_NAMES):
        mask = all_dom_flat == idx
        print(f"  {span_name}-dominant: {_mean_or_nan(all_compression_flat[mask]):.4f}")

    print("\n=== Clipping / floor diagnostics ===")
    print(f"  Heads where floor_val > alpha_raw: {diagnostics['floor_gt_raw']}")
    print(f"  Heads where final_coefficient == alpha_upper: {diagnostics['alpha_at_upper']}")

    print("\n=== Per-projection alpha summary ===")
    print(f"{'proj':<6}  {'mean':>8}  {'median':>8}  {'min':>8}  {'max':>8}  {'p90':>8}")
    for proj_type in ("q", "k", "v", "o"):
        flat = coefficients[proj_type].reshape(-1)
        p90 = torch.quantile(flat, 0.9).item()
        print(
            f"{proj_type:<6}  {flat.mean().item():>8.4f}  {flat.median().item():>8.4f}"
            f"  {flat.min().item():>8.4f}  {flat.max().item():>8.4f}  {p90:>8.4f}"
        )


if __name__ == "__main__":
    main()
