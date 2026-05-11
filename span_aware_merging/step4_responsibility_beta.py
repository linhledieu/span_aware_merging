"""Step 4 (beta variant): Compute per-head span responsibility tensors with Fisher beta dampening.

Identical to step4_responsibility.py except the head-span score uses:
    H_unnorm = M_hat * (F_hat ** fisher_beta)

where fisher_beta in (0, 1] compresses Fisher scores toward 1.0 before multiplying.
beta=1.0 reproduces frTrue exactly; beta=0.0 reduces to frFalse (uniform Fisher).

New config key: FISHER_BETA (float, default=0.5 here; falls back to cfg["fisher_beta"])
"""
import argparse
import json
import os

import torch
from transformers import AutoConfig

from config import load_config
from utils.normalization import normalize_responsibility_across_spans


def _group_q_to_kv(tensor_q, H_kv):
    """Average query-head values to KV-head groups.
    tensor_q: [num_layers, H_q, ...] → [num_layers, H_kv, ...]
    """
    H_q = tensor_q.shape[1]
    group_size = H_q // H_kv
    if tensor_q.dim() == 3:
        return tensor_q.view(tensor_q.shape[0], H_kv, group_size, tensor_q.shape[2]).mean(dim=2)
    else:
        return tensor_q.view(tensor_q.shape[0], H_kv, group_size).mean(dim=2)


def build_responsibility(U_heads, proj_type, cfg, fisher_dir, num_layers, H, fisher_beta):
    """Compute normalized responsibility tensor [num_layers, H, 3].

    Dampening is applied before normalization across spans so the relative
    ordering can shift before softmax-style normalization locks it in.
    """
    if cfg["USE_FISHER_RESPONSIBILITY"]:
        F_user = torch.load(
            os.path.join(fisher_dir, f"fisher_norm_user_{proj_type}.pt"), map_location="cpu"
        ).float()
        F_item = torch.load(
            os.path.join(fisher_dir, f"fisher_norm_item_{proj_type}.pt"), map_location="cpu"
        ).float()
        F_match = torch.load(
            os.path.join(fisher_dir, f"fisher_norm_match_{proj_type}.pt"), map_location="cpu"
        ).float()
        assert F_user.shape == (num_layers, H), \
            f"Fisher {proj_type} user shape mismatch: {F_user.shape} vs ({num_layers},{H})"
        # Dampening: compress Fisher scores toward 1.0 before multiplying with M_hat.
        # beta=1.0 → identical to frTrue; beta→0.0 → F_hat**beta → 1.0 → frFalse.
        R_user  = U_heads[:, :, 0] * (F_user  ** fisher_beta)
        R_item  = U_heads[:, :, 1] * (F_item  ** fisher_beta)
        R_match = U_heads[:, :, 2] * (F_match ** fisher_beta)
    else:
        R_user  = U_heads[:, :, 0].clone()
        R_item  = U_heads[:, :, 1].clone()
        R_match = U_heads[:, :, 2].clone()

    # Normalization happens AFTER dampening — this is intentional (see module docstring).
    ru, ri, rm = normalize_responsibility_across_spans(R_user, R_item, R_match)
    return torch.stack([ru, ri, rm], dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--override_json", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.override_json)

    # FISHER_BETA: new key, default 0.5; also accepts legacy cfg["fisher_beta"]
    fisher_beta = float(cfg.get("FISHER_BETA", cfg.get("fisher_beta", 0.5)))
    print(f"Fisher beta dampening: {fisher_beta}", flush=True)

    os.makedirs(cfg["responsibility_dir"], exist_ok=True)

    with open(os.path.join(cfg["attention_dir"], "attention_meta.json")) as f:
        meta = json.load(f)
    num_layers = meta["num_layers"]
    H_q = meta["num_attention_heads"]
    H_kv = meta["num_key_value_heads"]

    U = torch.load(os.path.join(cfg["attention_dir"], "U.pt"), map_location="cpu").float()
    assert U.shape == (num_layers, H_q, 3), f"U shape mismatch: {U.shape}"

    U_kv = _group_q_to_kv(U, H_kv)

    resp_q = build_responsibility(U, "q", cfg, cfg["fisher_dir"], num_layers, H_q, fisher_beta)
    resp_k = build_responsibility(U_kv, "k", cfg, cfg["fisher_dir"], num_layers, H_kv, fisher_beta)
    resp_v = build_responsibility(U_kv, "v", cfg, cfg["fisher_dir"], num_layers, H_kv, fisher_beta)
    resp_o = build_responsibility(U, "o", cfg, cfg["fisher_dir"], num_layers, H_q, fisher_beta)

    assert resp_q.shape == (num_layers, H_q, 3)
    assert resp_k.shape == (num_layers, H_kv, 3)
    assert resp_v.shape == (num_layers, H_kv, 3)
    assert resp_o.shape == (num_layers, H_q, 3)

    torch.save(resp_q, os.path.join(cfg["responsibility_dir"], "resp_q.pt"))
    torch.save(resp_k, os.path.join(cfg["responsibility_dir"], "resp_k.pt"))
    torch.save(resp_v, os.path.join(cfg["responsibility_dir"], "resp_v.pt"))
    torch.save(resp_o, os.path.join(cfg["responsibility_dir"], "resp_o.pt"))

    print("\n=== Dominant span distribution ===")
    print(f"{'proj':<6}  {'user_dom':>9}  {'item_dom':>9}  {'match_dom':>9}")
    for proj_type, resp in [("q", resp_q), ("k", resp_k), ("v", resp_v), ("o", resp_o)]:
        dom = resp.argmax(dim=-1)
        print(
            f"{proj_type:<6}  {(dom==0).sum().item():>9}"
            f"  {(dom==1).sum().item():>9}  {(dom==2).sum().item():>9}"
        )


if __name__ == "__main__":
    main()
