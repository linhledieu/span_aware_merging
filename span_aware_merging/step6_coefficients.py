"""Step 6: Compute per-head merging coefficients."""
import argparse
import csv
import json
import os

import torch

from config import load_config


def _group_q_to_kv(tensor_q, H_kv):
    """Average [L, H_q, ...] → [L, H_kv, ...]."""
    H_q = tensor_q.shape[1]
    group_size = H_q // H_kv
    if tensor_q.dim() == 3:
        return tensor_q.view(tensor_q.shape[0], H_kv, group_size, tensor_q.shape[2]).mean(dim=2)
    else:
        return tensor_q.view(tensor_q.shape[0], H_kv, group_size).mean(dim=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--override_json", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.override_json)

    os.makedirs(cfg["coefficients_dir"], exist_ok=True)

    with open(os.path.join(cfg["attention_dir"], "attention_meta.json")) as f:
        meta = json.load(f)
    num_layers = meta["num_layers"]
    H_q = meta["num_attention_heads"]
    H_kv = meta["num_key_value_heads"]

    # Load attention stats — shape assertions
    a_align = torch.load(os.path.join(cfg["attention_dir"], "a_align.pt"), map_location="cpu").float()
    u_leak = torch.load(os.path.join(cfg["attention_dir"], "u_leak.pt"), map_location="cpu").float()
    U_q = torch.load(os.path.join(cfg["attention_dir"], "U.pt"), map_location="cpu").float()

    expected_shape = (num_layers, H_q, 3)
    assert a_align.shape == expected_shape, f"a_align shape: {a_align.shape}"
    assert u_leak.shape == expected_shape, f"u_leak shape: {u_leak.shape}"
    assert U_q.shape == expected_shape, f"U shape: {U_q.shape}"

    # Load floors and responsibility tensors
    floors = {
        pt: torch.load(os.path.join(cfg["floors_dir"], f"floor_{pt}.pt"), map_location="cpu").float()
        for pt in ("q", "k", "v", "o")
    }
    resp = {
        pt: torch.load(os.path.join(cfg["responsibility_dir"], f"resp_{pt}.pt"), map_location="cpu").float()
        for pt in ("q", "k", "v", "o")
    }

    # KV-grouped versions for k and v
    U_kv = _group_q_to_kv(U_q, H_kv)
    a_kv = _group_q_to_kv(a_align, H_kv)
    u_kv = _group_q_to_kv(u_leak, H_kv)

    proj_info = {
        "q": (H_q, U_q, a_align, u_leak),
        "k": (H_kv, U_kv, a_kv, u_kv),
        "v": (H_kv, U_kv, a_kv, u_kv),
        "o": (H_q, U_q, a_align, u_leak),
    }

    w_user = cfg["w_user"]
    w_item = cfg["w_item"]
    w_match = cfg["w_match"]
    nu = cfg["nu"]
    rho = cfg["rho"]
    alpha_upper = cfg["alpha_upper"]

    coefficients = {}
    csv_rows = []

    for proj_type, (H, U_heads, a_heads, u_heads) in proj_info.items():
        alpha = torch.zeros(num_layers, H, dtype=torch.float32)
        floor_t = floors[proj_type]
        resp_t = resp[proj_type]

        for l in range(num_layers):
            for h in range(H):
                U_u = U_heads[l, h, 0].item()
                U_i = U_heads[l, h, 1].item()
                U_m = U_heads[l, h, 2].item()

                w_hat_u = w_user * (1 + nu * U_u)
                w_hat_i = w_item * (1 + nu * U_i)
                w_hat_m = w_match * (1 + nu * U_m)
                denom = w_hat_u + w_hat_i + w_hat_m
                if denom > 1e-8:
                    w_hat_u /= denom
                    w_hat_i /= denom
                    w_hat_m /= denom
                else:
                    w_hat_u = w_hat_i = w_hat_m = 1.0 / 3

                a_u = a_heads[l, h, 0].item()
                a_i = a_heads[l, h, 1].item()
                a_m = a_heads[l, h, 2].item()
                u_u = u_heads[l, h, 0].item()
                u_i = u_heads[l, h, 1].item()
                u_m = u_heads[l, h, 2].item()

                g_h = (w_hat_u * (a_u - rho * u_u)
                       + w_hat_i * (a_i - rho * u_i)
                       + w_hat_m * (a_m - rho * u_m))

                H_h = (w_hat_u * (1 + u_u)
                       + w_hat_i * (1 + u_i)
                       + w_hat_m * (1 + u_m))

                floor_val = floor_t[l, h].item()
                raw = floor_val if H_h <= 1e-8 else g_h / H_h
                final = max(floor_val, min(alpha_upper, raw))

                dom_idx = resp_t[l, h].argmax().item()
                dom_span = ["user", "item", "match"][dom_idx]
                if dom_idx == 0:
                    final = max(floor_val, final * cfg["user_head_scale"])
                elif dom_idx == 1:
                    final = max(floor_val, final * cfg["item_head_scale"])

                alpha[l, h] = final

                csv_rows.append({
                    "proj_type": proj_type,
                    "layer": l,
                    "head_index": h,
                    "raw_coefficient": raw,
                    "floor_val": floor_val,
                    "final_coefficient": final,
                    "dominant_span": dom_span,
                    "user_head_scale": cfg["user_head_scale"],
                    "item_head_scale": cfg["item_head_scale"],
                    "U_user": U_u, "U_item": U_i, "U_match": U_m,
                    "a_user": a_u, "a_item": a_i, "a_match": a_m,
                    "u_user": u_u, "u_item": u_i, "u_match": u_m,
                    "w_hat_user": w_hat_u, "w_hat_item": w_hat_i, "w_hat_match": w_hat_m,
                })

        coefficients[proj_type] = alpha
        torch.save(alpha, os.path.join(cfg["coefficients_dir"], f"alpha_{proj_type}.pt"))

    csv_path = os.path.join(cfg["coefficients_dir"], "alpha_summary.csv")
    fieldnames = [
        "proj_type", "layer", "head_index", "raw_coefficient", "floor_val",
        "final_coefficient", "dominant_span", "user_head_scale", "item_head_scale",
        "U_user", "U_item", "U_match",
        "a_user", "a_item", "a_match",
        "u_user", "u_item", "u_match",
        "w_hat_user", "w_hat_item", "w_hat_match",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved alpha_summary.csv ({len(csv_rows)} rows) → {csv_path}")

    # Histogram diagnostics
    buckets = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
    labels = ["[0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]
    print("\n=== Coefficient distribution ===")
    for proj_type in ("q", "k", "v", "o"):
        flat = coefficients[proj_type].reshape(-1)
        counts = [((flat >= lo) & (flat < hi)).sum().item() for lo, hi in buckets]
        print(f"  {proj_type}: " + "  ".join(f"{lb}:{c}" for lb, c in zip(labels, counts)))

    print("\n=== Mean coefficient by dominant span ===")
    print(f"{'proj':<6}  {'match_dom':>10}  {'user_dom':>9}  {'item_dom':>9}")
    for proj_type in ("q", "k", "v", "o"):
        alpha = coefficients[proj_type]
        dom = resp[proj_type].argmax(dim=-1)

        def mean_masked(t, mask):
            v = t[mask]
            return v.mean().item() if v.numel() > 0 else float("nan")

        print(
            f"{proj_type:<6}  {mean_masked(alpha, dom==2):>10.4f}"
            f"  {mean_masked(alpha, dom==0):>9.4f}"
            f"  {mean_masked(alpha, dom==1):>9.4f}"
        )


if __name__ == "__main__":
    main()
