"""Step 5: Compute per-head multiplicative format suppression tensors."""
import argparse
import json
import os

import torch

from config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--override_json", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.override_json)

    os.makedirs(cfg["floors_dir"], exist_ok=True)

    with open(os.path.join(cfg["attention_dir"], "attention_meta.json")) as f:
        meta = json.load(f)
    num_layers = meta["num_layers"]
    h_q = meta["num_attention_heads"]
    h_kv = meta["num_key_value_heads"]

    shapes = {
        "q": (num_layers, h_q),
        "k": (num_layers, h_kv),
        "v": (num_layers, h_kv),
        "o": (num_layers, h_q),
    }

    delta_fmt = cfg["delta_fmt"]

    if delta_fmt == 0.0:
        print("Format suppression disabled, saving zeros.")
        for proj_type, shape in shapes.items():
            torch.save(
                torch.zeros(shape, dtype=torch.float32),
                os.path.join(cfg["floors_dir"], f"fisher_supp_tag_{proj_type}.pt"),
            )
    else:
        for proj_type, shape in shapes.items():
            f_tag = torch.load(
                os.path.join(cfg["fisher_dir"], f"fisher_norm_tag_{proj_type}.pt"),
                map_location="cpu",
            ).float()
            assert f_tag.shape == shape, f"tag Fisher shape {f_tag.shape} != expected {shape}"

            format_supp = delta_fmt * f_tag
            torch.save(
                format_supp,
                os.path.join(cfg["floors_dir"], f"fisher_supp_tag_{proj_type}.pt"),
            )

    print("\n=== Format suppression statistics ===")
    print(f"{'proj':<6}  {'mean':>8}  {'median':>8}  {'p90':>8}  {'max':>8}  {'>0.3':>6}  {'>0.5':>6}")
    for proj_type in ("q", "k", "v", "o"):
        supp = torch.load(os.path.join(cfg["floors_dir"], f"fisher_supp_tag_{proj_type}.pt"))
        flat = supp.reshape(-1)
        p90 = torch.quantile(flat, 0.9).item()
        print(
            f"{proj_type:<6}  {flat.mean().item():>8.4f}  {flat.median().item():>8.4f}"
            f"  {p90:>8.4f}  {flat.max().item():>8.4f}"
            f"  {(flat > 0.3).sum().item():>6}  {(flat > 0.5).sum().item():>6}"
        )


if __name__ == "__main__":
    main()
