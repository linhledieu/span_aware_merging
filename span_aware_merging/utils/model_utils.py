import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def _dtype(cfg):
    return getattr(torch, cfg["dtype_str"])


def load_reasoning_model(cfg):
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_r_path"],
        torch_dtype=_dtype(cfg),
        device_map=cfg["device"],
    )
    model.requires_grad_(True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_r_path"])
    return model, tokenizer


def load_merged_model_at_alpha1(cfg):
    """Load θ_R and add Δ_I⊥ at coefficient 1.0.

    The Stage 1 delta is a .pkl file with structure:
      data['projected_task_vectors']['qk'][layer][head]['dQ_proj']  [d_model, head_dim]
      data['projected_task_vectors']['qk'][layer][head]['dK_proj']  [d_model, head_dim]
      data['projected_task_vectors']['vo'][layer][head]['dV_proj']  [d_model, head_dim]
      data['projected_task_vectors']['vo'][layer][head]['dO_proj']  [d_model, head_dim]
      data['projected_task_vectors']['ffn'][layer]['dGate_proj']    full matrix
      data['config']['kv_heads'], ['head_dim']

    Application convention (from unified_model_merge.py):
      q_proj: weight[h*hd:(h+1)*hd, :] += dQ.T
      k_proj: weight[kvh*hd:(kvh+1)*hd, :] += avg(dK.T)  (kvh = h // gqa_ratio, averaged over group)
      v_proj: same as k
      o_proj: weight[:, h*hd:(h+1)*hd] += dO          (column slice, no transpose)
    """
    import pickle

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_r_path"],
        torch_dtype=_dtype(cfg),
        device_map=cfg["device"],
        attn_implementation="eager",
    )

    with open(cfg["stage1_delta_path"], "rb") as f:
        raw = pickle.load(f)

    ptv = raw["projected_task_vectors"]
    delta_cfg = raw["config"]
    kv_heads = delta_cfg["kv_heads"]
    head_dim = delta_cfg["head_dim"]
    num_q_heads = model.config.num_attention_heads
    gqa_ratio = num_q_heads // kv_heads

    with torch.no_grad():
        for layer_module in model.model.layers:
            pass  # just to verify attribute exists

        for li, heads_data in ptv.get("qk", {}).items():
            layer = model.model.layers[li]
            kv_acc_k = {}
            kv_cnt_k = {}
            for h, hdata in heads_data.items():
                hd = head_dim
                if "dQ_proj" in hdata:
                    d = hdata["dQ_proj"].to(layer.self_attn.q_proj.weight)
                    layer.self_attn.q_proj.weight.data[h*hd:(h+1)*hd, :] += d.T
                if "dK_proj" in hdata:
                    kv_h = h // gqa_ratio
                    d = hdata["dK_proj"].float().T
                    kv_acc_k.setdefault(kv_h, torch.zeros_like(d))
                    kv_acc_k[kv_h] += d
                    kv_cnt_k[kv_h] = kv_cnt_k.get(kv_h, 0) + 1
            for kv_h, acc in kv_acc_k.items():
                hd = head_dim
                avg = (acc / kv_cnt_k[kv_h]).to(layer.self_attn.k_proj.weight)
                layer.self_attn.k_proj.weight.data[kv_h*hd:(kv_h+1)*hd, :] += avg

        for li, heads_data in ptv.get("vo", {}).items():
            layer = model.model.layers[li]
            kv_acc_v = {}
            kv_cnt_v = {}
            for h, hdata in heads_data.items():
                hd = head_dim
                if "dV_proj" in hdata:
                    kv_h = h // gqa_ratio
                    d = hdata["dV_proj"].float().T
                    kv_acc_v.setdefault(kv_h, torch.zeros_like(d))
                    kv_acc_v[kv_h] += d
                    kv_cnt_v[kv_h] = kv_cnt_v.get(kv_h, 0) + 1
                if "dO_proj" in hdata:
                    d = hdata["dO_proj"].to(layer.self_attn.o_proj.weight)
                    layer.self_attn.o_proj.weight.data[:, h*hd:(h+1)*hd] += d
            for kv_h, acc in kv_acc_v.items():
                hd = head_dim
                avg = (acc / kv_cnt_v[kv_h]).to(layer.self_attn.v_proj.weight)
                layer.self_attn.v_proj.weight.data[kv_h*hd:(kv_h+1)*hd, :] += avg
        for li, ffn_data in ptv.get("ffn", {}).items():
            layer = model.model.layers[li]
            if "dGate_proj" in ffn_data:
                d = ffn_data["dGate_proj"].to(layer.mlp.gate_proj.weight)
                layer.mlp.gate_proj.weight.data += d
            if "dUp_proj" in ffn_data:
                d = ffn_data["dUp_proj"].to(layer.mlp.up_proj.weight)
                layer.mlp.up_proj.weight.data += d
            if "dDown_T_proj" in ffn_data:
                d = ffn_data["dDown_T_proj"].to(layer.mlp.down_proj.weight)
                # dDown_T is stored transposed relative to down_proj.weight
                layer.mlp.down_proj.weight.data += d.T if d.shape != layer.mlp.down_proj.weight.shape else d

    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_r_path"])
    return model, tokenizer


def get_head_config(model):
    mc = model.config
    num_attention_heads = mc.num_attention_heads
    num_key_value_heads = getattr(mc, "num_key_value_heads", num_attention_heads)
    head_dim = mc.hidden_size // num_attention_heads
    return {
        "num_layers": mc.num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "kv_head_dim": head_dim,
    }


def get_qkvo_param_names(model):
    pattern = re.compile(r"^(.+\.layers\.)(\d+)(\.self_attn\.(q|k|v|o)_proj\.weight)$")
    layer_proj: dict[str, dict[int, str]] = {
        "q_proj": {}, "k_proj": {}, "v_proj": {}, "o_proj": {}
    }
    for key in model.state_dict().keys():
        m = pattern.match(key)
        if m:
            layer_idx = int(m.group(2))
            proj = m.group(4) + "_proj"
            layer_proj[proj][layer_idx] = key

    num_layers = max(max(d.keys()) for d in layer_proj.values() if d) + 1

    result = {}
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert len(layer_proj[proj]) == num_layers, (
            f"{proj}: found {len(layer_proj[proj])} layers, expected {num_layers}"
        )
        result[proj] = [layer_proj[proj][l] for l in range(num_layers)]
    return result
