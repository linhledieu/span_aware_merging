#!/usr/bin/env python3
"""
Build a Stage-3-compatible projected_task_vectors pickle from two raw checkpoints.

This is intended for ablations where we want to skip Stage 1 nullspace projection and
instead use the direct checkpoint delta:

    delta = instruct_model - base_model

The output mimics the Stage-1 container format closely enough for Stage 3 to consume.
"""

import argparse
import json
import os
import pickle
import time
from typing import Any, Dict, List

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def _parse_selected_heads(heads_arg: str, n_heads: int) -> List[int]:
    if heads_arg == "all":
        return list(range(n_heads))
    selected = [int(x.strip()) for x in heads_arg.split(",") if x.strip()]
    if any(head < 0 or head >= n_heads for head in selected):
        raise ValueError(f"--heads contains indices outside [0, {n_heads - 1}]")
    return selected


def _parse_selected_layers(layers_arg: str, layers_tail: int, num_layers: int) -> List[int]:
    if layers_arg:
        return [int(x.strip()) for x in layers_arg.split(",") if x.strip()]
    if layers_tail <= 0 or layers_tail > num_layers:
        raise ValueError(f"Invalid --layers_tail={layers_tail} for num_layers={num_layers}")
    return list(range(num_layers - layers_tail, num_layers))


def _save_json_config(projected_data: Dict[str, Any], output_path: str) -> None:
    def _json_safe(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return {
                "__tensor__": True,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_json_safe(v) for v in value]
        return value

    config_path = output_path.replace(".pkl", "_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        config_data = projected_data["config"].copy()
        config_data["stats"] = projected_data["projection_stats"]
        json.dump(_json_safe(config_data), handle, ensure_ascii=False, indent=2)
    print(f"📋 Config info: {config_path}")


def build_raw_delta_projected(
    base_model_path: str,
    instruct_model_path: str,
    merge_types: str,
    selected_layers: List[int],
    selected_heads: List[int],
) -> Dict[str, Any]:
    print("📥 Loading base model on CPU...")
    model_base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()
    print("📥 Loading instruct model on CPU...")
    model_instruct = AutoModelForCausalLM.from_pretrained(
        instruct_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()

    base_cfg = model_base.config
    inst_cfg = model_instruct.config
    if base_cfg.hidden_size != inst_cfg.hidden_size:
        raise ValueError("Base and instruct models have different hidden sizes")
    if base_cfg.num_hidden_layers != inst_cfg.num_hidden_layers:
        raise ValueError("Base and instruct models have different layer counts")
    if base_cfg.num_attention_heads != inst_cfg.num_attention_heads:
        raise ValueError("Base and instruct models have different attention head counts")

    d_model = base_cfg.hidden_size
    n_heads = base_cfg.num_attention_heads
    head_dim = d_model // n_heads
    kv_heads = getattr(base_cfg, "num_key_value_heads", n_heads)

    merge_q = "q" in merge_types.lower()
    merge_k = "k" in merge_types.lower()
    merge_v = "v" in merge_types.lower()
    merge_o = "o" in merge_types.lower()
    merge_f = "f" in merge_types.lower()

    kv_share_counts = {kvh: 0 for kvh in range(kv_heads)}
    for head in selected_heads:
        kv_share_counts[head % kv_heads] += 1

    projected_task_vectors: Dict[str, Any] = {
        "qk": {},
        "vo": {},
        "ffn": {},
    }
    stats = {
        "total_cg_iterations": 0,
        "total_constraint_residual": 0.0,
        "layer_stats": {},
    }

    print("🔬 Building raw-delta projected task vectors...")
    with torch.no_grad():
        for layer_idx in selected_layers:
            base_layer = model_base.model.layers[layer_idx]
            inst_layer = model_instruct.model.layers[layer_idx]
            layer_stats = {"heads": {}}

            if merge_q or merge_k:
                projected_task_vectors["qk"][layer_idx] = {}
                delta_q = None
                delta_k = None
                if merge_q:
                    delta_q = (
                        inst_layer.self_attn.q_proj.weight.detach().to(torch.float32)
                        - base_layer.self_attn.q_proj.weight.detach().to(torch.float32)
                    )
                if merge_k:
                    delta_k = (
                        inst_layer.self_attn.k_proj.weight.detach().to(torch.float32)
                        - base_layer.self_attn.k_proj.weight.detach().to(torch.float32)
                    )

                for head in selected_heads:
                    head_data: Dict[str, torch.Tensor] = {}
                    if merge_q and delta_q is not None:
                        q_rows = slice(head * head_dim, (head + 1) * head_dim)
                        head_data["dQ_proj"] = delta_q[q_rows, :].T.contiguous().cpu()
                    if merge_k and delta_k is not None:
                        kvh = head % kv_heads
                        k_rows = slice(kvh * head_dim, (kvh + 1) * head_dim)
                        denom = max(1, kv_share_counts[kvh])
                        head_data["dK_proj"] = (delta_k[k_rows, :] / denom).T.contiguous().cpu()
                    if head_data:
                        projected_task_vectors["qk"][layer_idx][head] = head_data
                        layer_stats["heads"][head] = {}

            if merge_v or merge_o:
                projected_task_vectors["vo"][layer_idx] = {}
                delta_v = None
                delta_o = None
                if merge_v:
                    delta_v = (
                        inst_layer.self_attn.v_proj.weight.detach().to(torch.float32)
                        - base_layer.self_attn.v_proj.weight.detach().to(torch.float32)
                    )
                if merge_o:
                    delta_o = (
                        inst_layer.self_attn.o_proj.weight.detach().to(torch.float32)
                        - base_layer.self_attn.o_proj.weight.detach().to(torch.float32)
                    )

                for head in selected_heads:
                    head_data = {}
                    if merge_v and delta_v is not None:
                        kvh = head % kv_heads
                        v_rows = slice(kvh * head_dim, (kvh + 1) * head_dim)
                        denom = max(1, kv_share_counts[kvh])
                        head_data["dV_proj"] = (delta_v[v_rows, :] / denom).T.contiguous().cpu()
                    if merge_o and delta_o is not None:
                        o_cols = slice(head * head_dim, (head + 1) * head_dim)
                        head_data["dO_proj"] = delta_o[:, o_cols].contiguous().cpu()
                    if head_data:
                        projected_task_vectors["vo"][layer_idx][head] = head_data
                        layer_stats["heads"].setdefault(head, {})

            if merge_f:
                delta_gate = (
                    inst_layer.mlp.gate_proj.weight.detach().to(torch.float32)
                    - base_layer.mlp.gate_proj.weight.detach().to(torch.float32)
                )
                delta_up = (
                    inst_layer.mlp.up_proj.weight.detach().to(torch.float32)
                    - base_layer.mlp.up_proj.weight.detach().to(torch.float32)
                )
                delta_down = (
                    inst_layer.mlp.down_proj.weight.detach().to(torch.float32)
                    - base_layer.mlp.down_proj.weight.detach().to(torch.float32)
                )
                projected_task_vectors["ffn"][layer_idx] = {
                    "dGate_proj": delta_gate.contiguous().cpu(),
                    "dUp_proj": delta_up.contiguous().cpu(),
                    "dDown_T_proj": delta_down.T.contiguous().cpu(),
                }
                layer_stats["ffn"] = {}

            stats["layer_stats"][layer_idx] = layer_stats

    return {
        "projected_task_vectors": projected_task_vectors,
        "config": {
            "merge_types": merge_types,
            "selected_layers": selected_layers,
            "selected_heads": selected_heads,
            "d_model": d_model,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "kv_heads": kv_heads,
            "compute_dtype": "torch.float32",
            "layer_lambda_max": {str(layer): 1.0 for layer in selected_layers},
            "active_layers": selected_layers,
            "lambda_search_done": False,
            "projection_format": "raw_delta",
            "projection_source": {
                "kind": "raw_checkpoint_delta",
                "base_model": base_model_path,
                "instruct_model": instruct_model_path,
            },
        },
        "projection_stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw checkpoint differences into a Stage-3-compatible projected_task_vectors pickle"
    )
    parser.add_argument("--base", type=str, required=True, help="Base model path")
    parser.add_argument("--instruct", type=str, required=True, help="Instruct model path")
    parser.add_argument("--output_file", type=str, required=True, help="Output pickle path")
    parser.add_argument("--merge_types", type=str, default="qkvof", help="Combination of q/k/v/o/f")
    parser.add_argument("--layers", type=str, default="", help="Comma-separated layer ids; overrides --layers_tail")
    parser.add_argument("--layers_tail", type=int, default=0, help="Use the last N layers when --layers is omitted")
    parser.add_argument("--heads", type=str, default="all", help="Heads to include: all or comma-separated")
    args = parser.parse_args()

    start_time = time.time()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    cfg_probe = AutoConfig.from_pretrained(
        args.base,
        trust_remote_code=True,
    )
    num_layers = cfg_probe.num_hidden_layers
    n_heads = cfg_probe.num_attention_heads

    selected_layers = _parse_selected_layers(args.layers, args.layers_tail or num_layers, num_layers)
    selected_heads = _parse_selected_heads(args.heads, n_heads)

    print("🚀 Raw checkpoint -> projected_task_vectors conversion")
    print("=" * 70)
    print(f"Base: {args.base}")
    print(f"Instruct: {args.instruct}")
    print(f"Output file: {args.output_file}")
    print(f"Merge types: {args.merge_types.upper()}")
    print(f"Layers: {selected_layers}")
    print(f"Heads: {selected_heads}")

    projected_data = build_raw_delta_projected(
        base_model_path=args.base,
        instruct_model_path=args.instruct,
        merge_types=args.merge_types,
        selected_layers=selected_layers,
        selected_heads=selected_heads,
    )
    projected_data["runtime_info"] = {
        "runtime_seconds": time.time() - start_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
    }

    with open(args.output_file, "wb") as handle:
        pickle.dump(projected_data, handle)

    file_size_mb = os.path.getsize(args.output_file) / 1024 / 1024
    print(f"✅ Saved: {args.output_file} ({file_size_mb:.1f} MB)")
    _save_json_config(projected_data, args.output_file)


if __name__ == "__main__":
    main()
