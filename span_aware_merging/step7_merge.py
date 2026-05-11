"""Step 7: Apply per-head coefficients and produce merged checkpoint.

Delta format (projected_task_vectors.pkl):
  ptv['qk'][layer][head]['dQ_proj']  [d_model, head_dim]
  ptv['qk'][layer][head]['dK_proj']  [d_model, head_dim]
  ptv['vo'][layer][head]['dV_proj']  [d_model, head_dim]
  ptv['vo'][layer][head]['dO_proj']  [d_model, head_dim]
  ptv['ffn'][layer]['dGate_proj']    full matrix
  ptv['ffn'][layer]['dUp_proj']      full matrix
  ptv['ffn'][layer]['dDown_T_proj']  full matrix

Application convention (same as unified_model_merge.py):
  q_proj: weight[h*hd:(h+1)*hd, :]      += alpha * lam * dQ.T
  k_proj: weight[kvh*hd:(kvh+1)*hd, :]  += alpha * lam * avg(dK.T)  (kvh = h // gqa_ratio, averaged over group)
  v_proj: same as k
  o_proj: weight[:, h*hd:(h+1)*hd]      += alpha * lam * dO      (column slice, no transpose)
  ffn:    weight += 1.0 * lam * delta
"""
import argparse
import json
import os
import pickle

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--override_json", type=str, default=None)
    parser.add_argument("--lambda_global", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.override_json)
    if args.lambda_global is not None:
        cfg["lambda_global"] = args.lambda_global

    lam = cfg["lambda_global"]
    print(f"lambda_global = {lam}")

    # Load Stage 1 delta
    print(f"Loading Δ_I⊥ from {cfg['stage1_delta_path']}...")
    with open(cfg["stage1_delta_path"], "rb") as f:
        raw = pickle.load(f)

    ptv = raw["projected_task_vectors"]
    delta_cfg = raw["config"]
    kv_heads = delta_cfg["kv_heads"]
    head_dim = delta_cfg["head_dim"]
    selected_layers = delta_cfg["selected_layers"]
    selected_heads = delta_cfg["selected_heads"]

    print(f"Delta covers layers {selected_layers[0]}–{selected_layers[-1]}, "
          f"{len(selected_heads)} heads, kv_heads={kv_heads}, head_dim={head_dim}")

    # Load coefficient tensors
    alpha = {
        pt: torch.load(os.path.join(cfg["coefficients_dir"], f"alpha_{pt}.pt"), map_location="cpu").float()
        for pt in ("q", "k", "v", "o")
    }

    # Load θ_R
    dtype = getattr(torch, cfg["dtype_str"])
    print(f"Loading θ_R from {cfg['model_r_path']}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_r_path"], torch_dtype=dtype, device_map="cpu"
    )
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_r_path"])

    num_q_heads = model.config.num_attention_heads
    gqa_ratio = num_q_heads // kv_heads  # e.g. 16 // 2 = 8

    attn_mod_count = 0
    ffn_mod_count = 0

    with torch.no_grad():
        # Apply Q/K — K deltas accumulated per KV slot then averaged
        for li, heads_data in ptv.get("qk", {}).items():
            layer = model.model.layers[li]
            kv_acc_k = {}
            kv_cnt_k = {}
            for h, hdata in heads_data.items():
                if "dQ_proj" in hdata:
                    a_q = alpha["q"][li, h].item()
                    d = hdata["dQ_proj"].float()
                    layer.self_attn.q_proj.weight.data[h*head_dim:(h+1)*head_dim, :] += \
                        (lam * a_q * d.T).to(dtype)
                    attn_mod_count += 1
                if "dK_proj" in hdata:
                    kv_h = h // gqa_ratio
                    d = hdata["dK_proj"].float().T
                    kv_acc_k.setdefault(kv_h, torch.zeros_like(d))
                    kv_acc_k[kv_h] += d
                    kv_cnt_k[kv_h] = kv_cnt_k.get(kv_h, 0) + 1
            for kv_h, acc in kv_acc_k.items():
                a_k = alpha["k"][li, kv_h].item()
                avg = acc / kv_cnt_k[kv_h]
                layer.self_attn.k_proj.weight.data[kv_h*head_dim:(kv_h+1)*head_dim, :] += \
                    (lam * a_k * avg).to(dtype)
                attn_mod_count += 1

        # Apply V/O — V deltas accumulated per KV slot then averaged
        for li, heads_data in ptv.get("vo", {}).items():
            layer = model.model.layers[li]
            kv_acc_v = {}
            kv_cnt_v = {}
            for h, hdata in heads_data.items():
                kv_h = h // gqa_ratio
                if "dV_proj" in hdata:
                    d = hdata["dV_proj"].float().T
                    kv_acc_v.setdefault(kv_h, torch.zeros_like(d))
                    kv_acc_v[kv_h] += d
                    kv_cnt_v[kv_h] = kv_cnt_v.get(kv_h, 0) + 1
                if "dO_proj" in hdata:
                    a_o = alpha["o"][li, h].item()
                    d = hdata["dO_proj"].float()
                    layer.self_attn.o_proj.weight.data[:, h*head_dim:(h+1)*head_dim] += \
                        (lam * a_o * d).to(dtype)
                    attn_mod_count += 1
            for kv_h, acc in kv_acc_v.items():
                a_v = alpha["v"][li, kv_h].item()
                avg = acc / kv_cnt_v[kv_h]
                layer.self_attn.v_proj.weight.data[kv_h*head_dim:(kv_h+1)*head_dim, :] += \
                    (lam * a_v * avg).to(dtype)
                attn_mod_count += 1

        # Apply FFN at coefficient 1.0
        for li, ffn_data in ptv.get("ffn", {}).items():
            layer = model.model.layers[li]
            if "dGate_proj" in ffn_data:
                d = ffn_data["dGate_proj"].to(dtype)
                layer.mlp.gate_proj.weight.data += lam * d
                ffn_mod_count += 1
            if "dUp_proj" in ffn_data:
                d = ffn_data["dUp_proj"].to(dtype)
                layer.mlp.up_proj.weight.data += lam * d
                ffn_mod_count += 1
            if "dDown_T_proj" in ffn_data:
                d = ffn_data["dDown_T_proj"].to(dtype)
                w = layer.mlp.down_proj.weight
                if d.shape == w.shape:
                    w.data += lam * d
                else:
                    w.data += lam * d.T
                ffn_mod_count += 1

    lam_str = f"{lam:.2f}".replace(".", "_")
    fmt_str = f"{cfg['delta_fmt']:.2f}".replace(".", "_")
    name = f"merged_lambda_{lam_str}_delta_fmt_{fmt_str}"
    u_scale = cfg.get("user_head_scale", 1.0)
    i_scale = cfg.get("item_head_scale", 1.0)
    if u_scale != 1.0 or i_scale != 1.0:
        u_str = f"{u_scale:.2f}".replace(".", "_")
        i_str = f"{i_scale:.2f}".replace(".", "_")
        name += f"_uscale_{u_str}_iscale_{i_str}"
    out_dir = os.path.join(cfg["checkpoint_dir"], name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Saving merged model to {out_dir}...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    with open(os.path.join(out_dir, "config_snapshot.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    print(f"\nApplied {attn_mod_count} attention head deltas, {ffn_mod_count} FFN deltas")


if __name__ == "__main__":
    main()
